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

    def generate(self, batch_size=1, prefix=""):
        """
        Overridable method to generate the ansi-c code of this operation.
        This super class method generates the input and output pointers
        mapped onto th tensor arena
        :return: String,
        """
        buffer_declarations = ""
        for idx, input in enumerate(self.operation.inputs):
          print("Generating buff declarations for tensor type [%s]" % input.type)
          if input.type == tg.TenType.INPUT:
              buffer_declarations += "    const {0} *input_{1} = {2};\n".format(
                  types.get_dtype_c_type(input.d_type),
                  idx,
                  c_gen.c_safe_identifier(input.label)
              )
          elif input.type == tg.TenType.CONSTANT:
              buffer_declarations += "    const {0} *input_{1} = ({0}*){2}{3};\n".format(
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
            buffer_declarations += "    {0} *output_{1} = {2};\n".format(
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

    @staticmethod
    def process_template(template, values):
        """

        :param template:
        :param values:
        :return:
        """
        output = template
        for key, value in values.items():
            output = output.replace(key, str(value))
        return output
