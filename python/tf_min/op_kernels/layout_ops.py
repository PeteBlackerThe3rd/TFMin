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

    Layout operation kernels of the TFMin library

    Operations supported
    -------------------------------
    reshape     (prod)
    concatV2    (prod)

"""
import tensorflow as tf
import numpy as np
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils
import tf_min.op_kernels.base_op as base_op
import tf_min.cpp_gen.cpp_gen as cpp_gen


class ReshapeOpKernel(base_op.BaseOpKernel):

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Reshape"

    @staticmethod
    def description():
        return "Reshape operation, fully supports the Reshape " \
               "tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_reference():
      return True

    @classmethod
    def gen_code(cls, tf_op, inputs):

        output_shape = tf_utils.np_tensor_shape(tf_op.outputs[0])
        input_identifier = code_gen.c_safe_identifier(inputs[0].name)

        code = "%s %s.reshape(Eigen::array<int, %d>(%s));" % \
               (base_op.BaseOpKernel.output_assignment(
                 tf_op, base_op.BaseOpKernel.evaluate_all
               ),
                input_identifier,
                len(output_shape),
                code_gen.ndarray_1d_to_literal(output_shape))
        return code


class FillOpKernel(base_op.BaseOpKernel):

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Fill"

    @staticmethod
    def description():
        return "Fill operation, fully supports the Reshape " \
               "tensorflow operation."

    @staticmethod
    def status():
        return "Testing"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        # output_shape = tf_utils.np_tensor_shape(tf_op.outputs[0])
        output_identifier = code_gen.c_safe_identifier(tf_op.outputs[0].name)

        # print("Fill operation looks like this. . .")
        # super().print_operation_details(tf_op)

        type = code_gen.get_c_dtype(tf_op.outputs[0].dtype.base_dtype)
        constant_value = tf_utils.get_const_scalar(
          tf_utils.get_parent_of_tensor(inputs[1])
        )

        code = cpp_gen.CodeBlock()
        code.add_statement(cpp_gen.Statement(
            base_op.BaseOpKernel.output_assignment(
                tf_op,
                eval=True,
                assignment=False
            )
        ))

        code.add_statement(cpp_gen.Statement("%s.setConstant((%s)%f)" %
                                             (output_identifier,
                                              type,
                                              constant_value)))
        return code


class SliceOpKernel(base_op.BaseOpKernel):

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Slice"

    @staticmethod
    def description():
        return "Slice operation, fully supports the Reshape " \
               "tensorflow operation."

    @staticmethod
    def status():
        return "Testing"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input_identifier = code_gen.c_safe_identifier(inputs[0].name)

        begin_type = tf_utils.get_parent_of_tensor(inputs[1]).type
        if begin_type != "Const":
            print("Error generating 'Slice' Operation: op_kernel only "
                  "supports Constant begin tensors.")
            return "// Error cannot generate Slice operation with " \
                   "non-const begin tensor!"

        size_type = tf_utils.get_parent_of_tensor(inputs[2]).type
        if size_type != "Const":
            print("Error generating 'Slice' Operation: op_kernel only "
                  "supports Constant size tensors.")
            return "// Error cannot generate Slice operation with " \
                   "non-const size tensor!"

        begin = tf_utils.get_const_tensor(
          tf_utils.get_parent_of_tensor(inputs[1])
        )
        size = tf_utils.get_const_tensor(
          tf_utils.get_parent_of_tensor(inputs[2])
        )

        # if -1 was given for any size dimensions then set them to the size
        # required to fill the remainder of the input
        for si in range(len(size)):
            if size[si] == -1:
                size[si] = inputs[0].dim_size(si) - begin[si]

        code = "%s %s.slice(Eigen::array<int, 2>(%s), " \
               "Eigen::array<int, 2>({%s}));" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input_identifier,
                code_gen.ndarray_1d_to_literal(begin),
                code_gen.ndarray_1d_to_literal(size))

        # print("Slice operation looks like this. . .")
        # super().print_operation_details(tf_op)

        return code


class CastOpKernel(base_op.BaseOpKernel):

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Cast"

    @staticmethod
    def description():
        return "Cast operation, fully supports the Cast tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        # super().print_operation_details(tf_op)

        input_identifier = code_gen.c_safe_identifier(inputs[0].name)
        type = code_gen.get_c_dtype(tf_op.get_attr("DstT"))

        code = "%s %s.cast<%s>();" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, eval=False),
                input_identifier,
                type)
        return code


class SplitOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Split, convolutional operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Split"

    @staticmethod
    def description():
        return "Split operation, fully supports the tensorflow operation."

    @staticmethod
    def status():
        return "Development"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        # base_op.BaseOpKernel.print_operation_details(tf_op)

        num_split = tf_op.get_attr("num_split")

        # This development version only supports the form where axis is
        # provided by a rank 0 constant operation
        if tf_utils.get_parent_of_tensor(inputs[0]).type != "Const":
            print("Error : Split operation doesn't support computed values "
                  "for axis yet!")
            return "// Error : Couldn't produce split operation with a " \
                   "computed axis dimension."

        # axis is provided by the first input tensor
        axis = tf_utils.get_const_scalar(
          tf_utils.get_parent_of_tensor(inputs[0])
        )

        # if there is an undefined batch dimension that has been collapsed
        # reduce the axis index by 1
        reduced_rank = len(tf_utils.np_tensor_shape(tf_op.outputs[0]))
        if reduced_rank != tf_op.outputs[0].shape.ndims:
            axis -= (tf_op.outputs[0].shape.ndims - reduced_rank)

        code = ""

        # if num_split is an integer then generate form 1 of this
        # operation where the input tensor is split into
        # num_split tensors, divided evenly along axis
        if type(num_split) is int:

            # verify that the size of dimenions 'axis' is a muliple of num_split
            input_axis_size = tf_utils.np_tensor_shape(inputs[1])[axis]
            if input_axis_size % num_split != 0:
                print("Error : Split operation trying to split dimenson of "
                      "size %d into %d parts, leaves remainder." %
                      (input_axis_size, num_split))
                return "// Error : Couldn't produce split operation where " \
                       "tensor doesn't divide into num_split parts"

            # Calculate the size in 'axis' of each output slice
            size = input_axis_size / num_split

            input1_identifier = code_gen.c_safe_identifier(inputs[1].name)
            rank = len(tf_utils.np_tensor_shape(inputs[1]))

            offset = np.zeros(rank, dtype=int)
            extents = tf_utils.np_tensor_shape(inputs[1])
            extents[axis] = size

            # generate code for each output tensor
            for idx in range(num_split):
                code += base_op.BaseOpKernel.output_assignment(tf_op, idx=idx)

                offset[axis] = idx * size

                code += " %s.slice(Eigen::array<int, %d>(%s), " \
                        "Eigen::array<int, %d>(%s));" % \
                        (input1_identifier,
                         rank,
                         code_gen.ndarray_1d_to_literal(offset),
                         rank,
                         code_gen.ndarray_1d_to_literal(extents)
                         )

        else:  # TODO need to implement this
            code = "// Error Split operation does not currently " \
                   "support arbitrary sized splits"

        return code


class ConcatV2OpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the ConcatV2, convolutional operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "ConcatV2"

    @staticmethod
    def description():
        return "ConcatV2 operation, fully supports the tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input1_identifier = code_gen.c_safe_identifier(inputs[1].name)

        axis = tf_utils.get_const_scalar(
          tf_utils.get_parent_of_tensor(inputs[2])
        )

        # if there is an undefined batch dimension that has been collapsed
        # reduce the axis index by 1
        reduced_rank = len(tf_utils.np_tensor_shape(tf_op.outputs[0]))
        if reduced_rank != tf_op.outputs[0].shape.ndims:
            axis -= (tf_op.outputs[0].shape.ndims - reduced_rank)

        code = "%s %s.concatenate(%s, %d);" % \
               (base_op.BaseOpKernel.output_assignment(tf_op),
                input0_identifier,
                input1_identifier,
                axis)

        return code
