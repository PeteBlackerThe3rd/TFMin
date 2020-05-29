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


class Conv2DOpKernel(base.BaseOpKernel):

    CONV_2D_TEMPLATE = """
    // TODO
    """

    def __init__(self, operation):
        """

        :param operation:
        """
        super().__init__(operation)

    @staticmethod
    def matches(operation):
      return (operation.type == 'Conv2D')

    @staticmethod
    def description():
        return "Conv2D kernel,\n" + \
               "Currently supports float and integer data types"

    @staticmethod
    def status():
      """
      Development status of this op kernel.
      Either development, testing, production, or base.
      :return: String
      """
      return "testing"

    def generate(self, batch_size=1, prefix=""):
      """
      Overridable method to generate the ansi-c code of this operation.
      :return: String,
      """
      input_shape = self.operation.inputs[0].get_tensor_shape(batch_size)
      output_shape = self.operation.outputs[0].get_tensor_shape(batch_size)
      (input_d1_coeff,
       input_d2_coeff,
       input_d3_coeff,
       input_d4_coeff) = [1, 2, 3, 4]  # TODO make layout object and use here
      (output_d1_coeff,
       output_d2_coeff,
       output_d3_coeff,
       output_d4_coeff) = [1, 2, 3, 4]  # TODO make layout object and use here
      template_values = {
        'batches': input_shape[0],
        'depth': input_shape[3],
        'input_height': input_shape[1],
        'input_width': input_shape[2],
        'output_width': output_shape[1],
        'output_height': output_shape[2],
        'stride_width': self.operation.params['stride_width'],
        'stride_height': self.operation.params['stride_height'],
        'padding_width': 0,  # TODO comp actual values
        'padding_height': 0,
        'filter_width': 0,
        'filter_height': 0,
        'D_TYPE': types.get_dtype_c_type(self.operation.inputs[0].d_type),
        'lowest_possible': types.get_dtype_lowest(self.operation.inputs[0].d_type),
        'input_d1_coeff': input_d1_coeff,
        'input_d2_coeff': input_d2_coeff,
        'input_d3_coeff': input_d3_coeff,
        'input_d4_coeff': input_d4_coeff,
        'output_d1_coeff': output_d1_coeff,
        'output_d2_coeff': output_d2_coeff,
        'output_d3_coeff': output_d3_coeff,
        'output_d4_coeff': output_d4_coeff
      }

      # generate buffer declarations
      code = super().generate(batch_size, prefix)
      code += base.BaseOpKernel.process_template(
          Conv2DOpKernel.CONV_2D_TEMPLATE,
          template_values
      )

      return code
