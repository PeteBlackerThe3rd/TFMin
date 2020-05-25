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


class PoolingOpKernel(base.BaseOpKernel):

    MAX_POOL_TEMPLATE = """
    //const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    //const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    //const int input_height = input_shape.Dims(1);
    //const int input_width = input_shape.Dims(2);
    //const int output_height = output_shape.Dims(1);
    //const int output_width = output_shape.Dims(2);
    //const int stride_height = params.stride_height;
    //const int stride_width = params.stride_width;
    //const int padding_width = ??
    //const int padding_height = ??
    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < depth; ++channel) {
            const int in_x_origin = (out_x * stride_width) - padding_width;
            const int in_y_origin = (out_y * stride_height) - padding_height;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = std::max(0, -in_x_origin);
            const int filter_x_end =
                std::min(params.filter_width, input_width - in_x_origin);
            const int filter_y_start = std::max(0, -in_y_origin);
            const int filter_y_end =
                std::min(params.filter_height, input_height - in_y_origin);
            float max = std::numeric_limits<float>::lowest();
            for (int filter_y = filter_y_start; filter_y < filter_y_end;
                 ++filter_y) {
              for (int filter_x = filter_x_start; filter_x < filter_x_end;
                   ++filter_x) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                max = std::max(
                    max,
                    input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
              }
            }
            output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
                ActivationFunctionWithMinMax(max, params.float_activation_min,
                                             params.float_activation_max);
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

    def generate(self, batch_size=1):
      """
      Overridable method to generate the ansi-c code of this operation.
      :return: String,
      """
      input_shape = self.operation.inputs[0].get_tensor_shape(batch_size)
      output_shape = self.operation.outputs[0].get_tensor_shape(batch_size)
      template_values = {'batches': input_shape[0],
                         'depth': input_shape[3],
                         'input_height': input_shape[1],
                         'input_width': input_shape[2],
                         'output_width': output_shape[1],
                         'output_height': output_shape[2],
                         'stride_width': self.operation.params['stride_width'],
                         'stride_height': self.operation.params['stride_height'],
                         'padding_width': 0,  # TODO comp actual values
                         'padding_height': 0
                         }

      return base.BaseOpKernel.process_template(
          PoolingOpKernel.MAX_POOL_TEMPLATE,
          template_values
      )
