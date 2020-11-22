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
from .base_op_kernel import BaseOpKernel
import tf_min.types as types


class DepthwiseConv2DOpKernel(BaseOpKernel):

    DEPTHWISECONV_2D_TEMPLATE = """
  const D_TYPE *filter_data = input_1; 
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int out_channel = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - padding_width;
            const int in_y_origin = (out_y * stride_height) - padding_height;
            D_TYPE value = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value = input_0[batch * input_d1_coeff + 
                                              in_y * input_d2_coeff +
                                              in_x * input_d3_coeff +
                                              ic * input_d4_coeff +
                                              input_d_base];
                  float filter_value = filter_data[0 * filter_d1_coeff +
                                                   filter_y * filter_d2_coeff +
                                                   filter_x * filter_d3_coeff +
                                                   out_channel * filter_d4_coeff +
                                                   filter_d_base];
                  value += input_value * filter_value;
                }
              }
            }
            BIAS_ADD
            // Fused activation function
            ACTIVATION_FN
            output_0[batch * output_d1_coeff +
                     out_y * output_d2_coeff +
                     out_x * output_d3_coeff +
                     out_channel * output_d4_coeff +
                     output_d_base] = value;
          }
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
      return operation.type == 'DepthwiseConv2D'

    @staticmethod
    def description():
        return "DepthwiseConv2D kernel,\n" + \
               "Currently supports float and integer data types"

    @staticmethod
    def status():
      """
      Development status of this op kernel.
      Either development, testing, production, or base.
      :return: String
      """
      return "testing"

    def generate(self, batch_size=1, prefix="", fake_weights=None):
      """
      Overridable method to generate the ansi-c code of this operation.
      :return: String,
      """
      # prepare values for code generate
      input_shape = self.operation.inputs[0].get_tensor_shape(batch_size)
      filter_shape = self.operation.inputs[1].get_tensor_shape(batch_size)
      output_shape = self.operation.outputs[0].get_tensor_shape(batch_size)

      padding = super().compute_padding(
        filter_width=filter_shape[0],
        filter_height=filter_shape[1]
      )

      bias_add = ""
      if len(self.operation.inputs) > 2:
        bias_add = "// Add Bias\nvalue += input_2[out_channel];"

      # Get the offset function coefficients for the input and output tensors
      (input_d1_coeff,
       input_d2_coeff,
       input_d3_coeff,
       input_d4_coeff,
       input_d_base) = \
        self.operation.inputs[0].shape.get_layout_addressing_coeffs()
      (filter_d1_coeff,
       filter_d2_coeff,
       filter_d3_coeff,
       filter_d4_coeff,
       filter_d_base) = \
        self.operation.inputs[1].shape.get_layout_addressing_coeffs()
      (output_d1_coeff,
       output_d2_coeff,
       output_d3_coeff,
       output_d4_coeff,
       output_d_base) = \
        self.operation.outputs[0].shape.get_layout_addressing_coeffs()

      # populate template dictionary used to transform template into final code
      template_values = {
        'batches': input_shape[0],
        'input_height': input_shape[1],
        'input_width': input_shape[2],
        'input_depth': input_shape[3],
        'output_width': output_shape[1],
        'output_height': output_shape[2],
        'output_depth': output_shape[3],
        'stride_width': self.operation.params['stride_width'],
        'stride_height': self.operation.params['stride_height'],
        'dilation_width_factor': self.operation.params['dilation_width_factor'],
        'dilation_height_factor': self.operation.params['dilation_height_factor'],
        'depth_multiplier': self.operation.params['depth_multiplier'],
        'padding_width': padding['pad_width'],
        'padding_height': padding['pad_height'],
        'filter_width': filter_shape[0],
        'filter_height': filter_shape[1],
        'D_TYPE': types.get_dtype_c_type(self.operation.inputs[0].d_type),
        'input_d1_coeff': input_d1_coeff,
        'input_d2_coeff': input_d2_coeff,
        'input_d3_coeff': input_d3_coeff,
        'input_d4_coeff': input_d4_coeff,
        'input_d_base': input_d_base,
        'filter_d1_coeff': filter_d1_coeff,
        'filter_d2_coeff': filter_d2_coeff,
        'filter_d3_coeff': filter_d3_coeff,
        'filter_d4_coeff': filter_d4_coeff,
        'filter_d_base': filter_d_base,
        'output_d1_coeff': output_d1_coeff,
        'output_d2_coeff': output_d2_coeff,
        'output_d3_coeff': output_d3_coeff,
        'output_d4_coeff': output_d4_coeff,
        'output_d_base': output_d_base,
        'BIAS_ADD': bias_add,
        'ACTIVATION_FN': super().gen_act_code()
      }

      # generate buffer declarations
      code = super().generate(batch_size, prefix)

      # merge template to generate c implementation of conv 2D layer
      code += BaseOpKernel.process_template(
          DepthwiseConv2DOpKernel.DEPTHWISECONV_2D_TEMPLATE,
          template_values
      )

      return code