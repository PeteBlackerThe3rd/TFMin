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
import tf_min.v2_kernels.base_op_kernel as base
import tf_min.types as types


class FullyConnectedOpKernel(base.BaseOpKernel):

    FC_TEMPLATE = """  
  // accum_depth is the last dimension of the input tensor
  // output_depth is the last dimension of the output tensor
  const D_TYPE *weightsData = input_1;
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      D_TYPE value = zero_literal;
      for (int d = 0; d < accum_depth; ++d) {
        D_TYPE inputVal = input_0[(b * accum_depth) + d];
        // D_TYPE weightVal = weightsData[(out_c * accum_depth) + d];
        // TODO need to work out why the weights are transposed so this
        // line has needed altering.
        D_TYPE weightVal = weightsData[(out_c + (output_depth * d)) fake_modifier];
        value += inputVal * weightVal;
      }
      BIAS_ADD
      // fused activation function
      ACTIVATION_FN
      output_0[out_c + (output_depth * b)] = value;
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
      return operation.type == 'MatMul'

    @staticmethod
    def description():
        return "MatMul kernel,\n" + \
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
      input_shape = self.operation.inputs[0].shape
      output_shape = self.operation.outputs[0].shape

      bias_add = ""
      if len(self.operation.inputs) > 2:
        bias_add = "// Add Bias\nvalue += input_2[out_c];"

      fake_modifier = ""
      if fake_weights is not None:
        weights_d_type_size = types.get_dtype_size(
          self.operation.inputs[1].d_type)
        fake_modifier = " %% %d" % int(fake_weights / weights_d_type_size)

      # populate template dictionary used to transform template into final code
      template_values = {
        'batches': batch_size,
        'accum_depth': input_shape.last_dim(),
        'output_depth': output_shape.last_dim(),
        'D_TYPE': types.get_dtype_c_type(self.operation.inputs[0].d_type),
        'zero_literal': types.get_dtype_zero(self.operation.inputs[0].d_type),
        'BIAS_ADD': bias_add,
        'ACTIVATION_FN': super().gen_act_code(),
        'fake_modifier': fake_modifier
      }

      # generate buffer declarations
      code = super().generate(batch_size, prefix)

      # merge template to generate c implementation of conv 2D layer
      code += base.BaseOpKernel.process_template(
        FullyConnectedOpKernel.FC_TEMPLATE,
          template_values
      )

      return code
