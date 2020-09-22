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
import tf_min.graph as tg
import tf_min.types as types
from tf_min import cpp_code_gen as c_gen
import tf_min.activation_fns as act_fns


class BaseOpKernel:

    def __init__(self, operation):
        """
        create an instance of this op kernel which will generate the code
        for the given operation instance
        :param operation: tf_min.Operation object
        """
        assert self.matches(operation), "Error, cannot instantiate %s op " \
                                        "kernel from a %s operation" % (
            type(self),
            operation.type
        )

        self.operation = operation
        self.tags = []

    @staticmethod
    def matches(operation):
        print("Error call to matches on base class with op [%s]!" %
              operation.type)
        return False

    @staticmethod
    def description():
        return "Error description called on base class!"

    @staticmethod
    def status():
        """
        Development status of this op kernel.
        Either development, testing, production, or base.
        :return: String
        """
        return "base"

    def get_safe_overlap(self):
        """
        Base op safe overlap method, this always returns zero
        unless its overriden with a specialised overlap calculation method
        :return: zero bytes
        """
        return 0

    def gen_act_code(self):
        """
        Function to generate the activation function code for this opeation,
        if no activation function is specified then a single newline is
        returned.
        :return: String, the c code implementation of the activation function
        """
        # TODO need to add code to extract all op params which are prefixed
        #   with act_ and pass them as a list to get_act_code.
        act_fn = act_fns.ActType.NONE
        if 'fused_activation_fn' in self.operation.params.keys():
          act_fn = self.operation.params['fused_activation_fn']
        return act_fns.gen_act_code(act_fn, self.operation.inputs[0].d_type)

    def compute_single_padding(self,
                               stride, dilation, in_size,
                               filter_size, out_size):
        """
        Method to compute the padding in a single dimension
        :param stride:
        :return:
        """
        effective_filter_size = (filter_size - 1) * dilation + 1
        padding = int(((out_size - 1) * stride +
                       effective_filter_size - in_size) / 2)
        if padding > 0:
          return padding
        else:
          return 0

    def compute_padding(self, filter_width, filter_height):
        """
        Method to compute the padding width and height of the operation from
        its stride and input and output sizes.
        :return: dictionary of the form {'pad_width': int, 'pad_height': int}
        """
        stride_width = 1
        stride_height = 1
        if 'stride_width' in self.operation.params:
          stride_width = self.operation.params['stride_width']
        if 'stride_height' in self.operation.params:
          stride_height = self.operation.params['stride_height']

        dilation_width_factor = 1
        dilation_height_factor = 1
        if 'dilation_width_factor' in self.operation.params:
          dilation_width_factor = \
            self.operation.params['dilation_width_factor']
        if 'dilation_height_factor' in self.operation.params:
          dilation_height_factor = \
            self.operation.params['dilation_height_factor']

        pad_width = self.compute_single_padding(stride_width,
                                                dilation_width_factor,
                                                self.operation.inputs[0].shape[2],
                                                filter_width,
                                                self.operation.outputs[0].shape[2])
        pad_height = self.compute_single_padding(stride_height,
                                                 dilation_height_factor,
                                                 self.operation.inputs[0].shape[1],
                                                 filter_height,
                                                 self.operation.outputs[0].shape[1])

        return {'pad_width': pad_width, 'pad_height': pad_height}

    def generate(self, batch_size=1, prefix=""):
        """
        Overridable method to generate the ansi-c code of this operation.
        This super class method generates the input and output pointers
        mapped onto th tensor arena
        :return: String,
        """
        buffer_declarations = ""
        for idx, input in enumerate(self.operation.inputs):
          # print("Generating buff declarations for tensor type [%s]" %
          #       input.type)
          if input.type == tg.TenType.INPUT:
              buffer_declarations += "    const {0} *input_{1} = p_{2};\n".format(
                  types.get_dtype_c_type(input.d_type),
                  idx,
                  c_gen.c_safe_identifier(input.label)
              )
          elif input.type == tg.TenType.CONSTANT:
              buffer_declarations += \
                "    const {0} *input_{1} = ({0}*){2}{3};\n".format(
                  types.get_dtype_c_type(input.d_type),
                  idx,
                  prefix,
                  c_gen.c_safe_identifier(input.label)
                )
          else:
              buffer_declarations += \
                "    const {0} *input_{1} = ({0}*)tensor_arena + {2};\n".format(
                  types.get_dtype_c_type(input.d_type),
                  idx,
                  int(input.memory_offset / types.get_dtype_size(input.d_type))
                )
        for idx, output in enumerate(self.operation.outputs):
          if output.type == tg.TenType.OUTPUT:
            buffer_declarations += "    {0} *output_{1} = p_{2};\n".format(
              types.get_dtype_c_type(output.d_type),
              idx,
              c_gen.c_safe_identifier(output.label)
            )
          else:
            buffer_declarations += \
              "    {0} *output_{1} = ({0}*)tensor_arena + {2};\n".format(
                types.get_dtype_c_type(output.d_type),
                idx,
                int(output.memory_offset / types.get_dtype_size(output.d_type))
              )
        return buffer_declarations

    def get_dependencies(self):
        """
        Method which returns a dictionary of include dependencies where
        the keys are the strings of the files required
        :return: Dictionary of dependencies
        """
        return {}

    @staticmethod
    def process_template(template, values):
        """
        For now a crude template function which simply replaces a list of
        (original -> new) pairs in the input value.
        No chain update avoidance is performed!
        TODO Need to make this more robust at some point.
        :param template:
        :param values:
        :return:
        """
        output = template
        for key, value in values.items():
            output = output.replace(key, str(value))
        return output
