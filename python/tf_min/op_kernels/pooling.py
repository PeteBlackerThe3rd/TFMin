"""
    TFMin v1.0 Minimal TensorFlow to C++ exporter
    ------------------------------------------

    Copyright (C) 2019 Pete Blacker, Surrey Space Centre & Airbus Defence and
    Space Ltd.
    Pete.Blacker@Surrey.ac.uk
    https://www.surrey.ac.uk/surrey-space-centre/research-groups/on-board-data-handling

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    in the LICENCE file of this software.  If not, see
    <http://www.gnu.org/licenses/>.

    ---------------------------------------------------------------------

    This module contains the base operation kernel for generating ansi-c
    code for the v2 architecture.
"""
import math as m
from .base_op_kernel import BaseOpKernel
import tf_min.types as types
import tf_min.activation_fns as act_fns


class PoolingOpKernel(BaseOpKernel):

    MIN_MAX_POOL_TEMPLATE = """
    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < depth; ++channel) {
            const int in_x_origin = (out_x * stride_width) - padding_width;
            const int in_y_origin = (out_y * stride_height) - padding_height;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = in_x_origin > 0 ? 0 : -in_x_origin;
            const int filter_x_end = (input_width - in_x_origin < filter_width) ? input_width - in_x_origin : filter_width;
            const int filter_y_start = in_y_origin > 0 ? 0 : -in_y_origin;
            const int filter_y_end = (input_height - in_y_origin < filter_height) ? input_height - in_y_origin : filter_height;
            
            D_TYPE value = MIN_MAX_INITIAL;
            for (int filter_y = filter_y_start; filter_y < filter_y_end;
                 ++filter_y) {
              for (int filter_x = filter_x_start; filter_x < filter_x_end;
                   ++filter_x) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                
                D_TYPE input_value = input_0[batch * input_d1_coeff + 
                                             in_y * input_d2_coeff +
                                             in_x * input_d3_coeff +
                                             channel * input_d4_coeff +
                                             input_d_base];
                if (MIN_MAX_COMPARISON)
                  value = input_value;
              }
            }
            // Fused activation function
            ACTIVATION_FN
            output_0[batch * output_d1_coeff +
                     out_y * output_d2_coeff +
                     out_x * output_d3_coeff +
                     channel * output_d4_coeff +
                     output_d_base] = value;
          }
        }
      }
    }
"""
    AVG_POOL_TEMPLATE = """
    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < depth; ++channel) {
            const int in_x_origin = (out_x * stride_width) - padding_width;
            const int in_y_origin = (out_y * stride_height) - padding_height;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = in_x_origin > 0 ? 0 : -in_x_origin;
            const int filter_x_end = (input_width - in_x_origin < filter_width) ? input_width - in_x_origin : filter_width;
            const int filter_y_start = in_y_origin > 0 ? 0 : -in_y_origin;
            const int filter_y_end = (input_height - in_y_origin < filter_height) ? input_height - in_y_origin : filter_height;
            
            SUM_DATA_TYPE sum = 0;
            for (int filter_y = filter_y_start; filter_y < filter_y_end;
                 ++filter_y) {
              for (int filter_x = filter_x_start; filter_x < filter_x_end;
                   ++filter_x) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                
                D_TYPE input_value = input_0[batch * input_d1_coeff + 
                                             in_y * input_d2_coeff +
                                             in_x * input_d3_coeff +
                                             channel * input_d4_coeff +
                                             input_d_base];
                sum += input_value;
              }
            }
            D_TYPE value = sum / ((filter_x_end - filter_x_start) * (filter_y_end - filter_y_start));
            
            // Fused activation function
            ACTIVATION_FN
            output_0[batch * output_d1_coeff +
                     out_y * output_d2_coeff +
                     out_x * output_d3_coeff +
                     channel * output_d4_coeff +
                     output_d_base] = value;
          }
        }
      }
    }
"""

    def __init__(self, operation):
        """

        :param operation:
        """
        super().__init__(operation)

    @staticmethod
    def matches(operation):
      return (operation.type == 'AvgPool' or
              operation.type == 'MaxPool' or
              operation.type == "MinPool")

    @staticmethod
    def description():
        return "Pooling op kernel, supports min, max & average pooling.\n" + \
               "Current supports float and integer data types"

    @staticmethod
    def status():
      """
      Development status of this op kernel.
      Either development, testing, production, or base.
      :return: String
      """
      return "testing"

    def get_safe_overlap(self, debug=False):
      """
      Pooling customised method used to compute the safe overlap
      of the input and output intermediate buffers of this operation.
      :return: size of the safe overlap in bytes
      """
      output = self.operation.outputs[0]
      input_shape = self.operation.inputs[0].shape
      output_shape = output.shape

      input_w = input_shape[1]
      input_h = input_shape[2]
      input_d = input_shape[3]

      output_w = output_shape[1]
      output_h = output_shape[2]

      kernel_w = self.operation.params['kernel_width']
      kernel_h = self.operation.params['kernel_width']

      stride_w = self.operation.params['stride_width']
      stride_h = self.operation.params['stride_height']

      dilation_w = 1
      dilation_h = 1

      padding_h = m.floor((output_h * stride_h - stride_h + kernel_h * dilation_h - dilation_h - input_h + 1) / 2)
      padding_w = m.floor((output_w * stride_w - stride_w + kernel_w * dilation_w - dilation_w - input_w + 1) / 2)

      a = (stride_h * input_w) / float(output_w)
      b = (output_w * stride_w - padding_h * input_w - stride_h * input_w - stride_w - padding_w) * input_d + 1
      ic = output_h * output_w * input_d

      if debug:
        print("a = %f, b = %d, ic = %d" % (a, b, ic))

      output_buffer_size = output.get_buffer_size()
      output_dtype_size = types.get_dtype_size(output.d_type)

      safe_overlap = (output_buffer_size +
                      (min(b / a, a * ic + b - ic) * output_dtype_size))
      if debug:
        print("safe_overlap = %d" % safe_overlap)

      return safe_overlap

    def get_dependencies(self):
        """
        Method which returns a dictionary of include dependencies where
        the keys are the strings of the files required
        :return: Dictionary of dependencies
        """
        dependencies = {}

        # if the data type is float then float.h is required for constants
        if self.operation.inputs[0].d_type in [types.TenDType.FLOAT64,
                                               types.TenDType.FLOAT32,
                                               types.TenDType.FLOAT16]:
            dependencies['float.h'] = True

        # if the tanh fused activation function is used then math.h is requred
        if (
          'fused_activation_fn' in self.operation.params and
          self.operation.params['fused_activation_fn'] == act_fns.ActType.TANH
        ):
            dependencies['math.h'] = True

        return dependencies

    def generate(self, batch_size=1, prefix="", fake_weights=None):
      """
      Overridable method to generate the ansi-c code of this operation.
      :return: String,
      """
      # prepare values for code generate
      input_shape = self.operation.inputs[0].get_tensor_shape(batch_size)
      output_shape = self.operation.outputs[0].get_tensor_shape(batch_size)
      min_max_comparison = ""
      min_max_initial = ""
      if self.operation.type == "MaxPool":
        min_max_comparison = "input_value > value"
        min_max_initial = types.get_dtype_lowest(
          self.operation.inputs[0].d_type
        )
      elif self.operation.type == "MinPool":
        min_max_comparison = "input_value < value"
        min_max_initial = types.get_dtype_highest(
          self.operation.inputs[0].d_type
        )
      sum_d_type = types.get_dtype_c_type(self.operation.inputs[0].d_type)
      if (self.operation.type == "AvgPool" and
              types.is_integer(self.operation.inputs[0].d_type)):
          higher_d_type = types.get_higher_range_type(
            self.operation.inputs[0].d_type
          )
          sum_d_type = types.get_dtype_c_type(higher_d_type)

      kernel_w = self.operation.params['kernel_width']
      kernel_h = self.operation.params['kernel_width']

      padding = super().compute_padding(
        filter_width=kernel_w,
        filter_height=kernel_h
      )

      # Get the offset function coefficients for the input and output tensors
      (input_d1_coeff,
       input_d2_coeff,
       input_d3_coeff,
       input_d4_coeff,
       input_d_base) = \
        self.operation.inputs[0].shape.get_layout_addressing_coeffs()
      (output_d1_coeff,
       output_d2_coeff,
       output_d3_coeff,
       output_d4_coeff,
       output_d_base) = \
        self.operation.outputs[0].shape.get_layout_addressing_coeffs()

      # populate template dictionary used to transform template into final code
      template_values = {
        'batches': input_shape[0],
        'depth': input_shape[3],
        'input_height': input_shape[1],
        'input_width': input_shape[2],
        'output_width': output_shape[1],
        'output_height': output_shape[2],
        'stride_width': self.operation.params['stride_width'],
        'stride_height': self.operation.params['stride_height'],
        'padding_width': padding['pad_width'],
        'padding_height': padding['pad_height'],
        'filter_width': kernel_w,
        'filter_height': kernel_h,
        'D_TYPE': types.get_dtype_c_type(self.operation.inputs[0].d_type),
        'SUM_DATA_TYPE': sum_d_type,
        'MIN_MAX_COMPARISON': min_max_comparison,
        'MIN_MAX_INITIAL': min_max_initial,
        'input_d1_coeff': input_d1_coeff,
        'input_d2_coeff': input_d2_coeff,
        'input_d3_coeff': input_d3_coeff,
        'input_d4_coeff': input_d4_coeff,
        'input_d_base': input_d_base,
        'output_d1_coeff': output_d1_coeff,
        'output_d2_coeff': output_d2_coeff,
        'output_d3_coeff': output_d3_coeff,
        'output_d4_coeff': output_d4_coeff,
        'output_d_base': output_d_base,
        'ACTIVATION_FN': super().gen_act_code()
      }

      # generate buffer declarations
      code = super().generate(batch_size, prefix)

      # merge template to generate c implementation of pooling layer
      if self.operation.type == 'AvgPool':
        code += BaseOpKernel.process_template(
            PoolingOpKernel.AVG_POOL_TEMPLATE,
            template_values
        )
      else:
        code += BaseOpKernel.process_template(
            PoolingOpKernel.MIN_MAX_POOL_TEMPLATE,
            template_values
        )

      return code
