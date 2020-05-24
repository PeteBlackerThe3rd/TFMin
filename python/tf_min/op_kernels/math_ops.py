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

    Math operation kernels of the TFMin library

    Operations supported
    -------------------------------
    Add         (prod)
    Sub         (prod)
    Sqrt        (prod)
    Rsqrt       (prod)
    Mul         (prod)
    Max         (prod)
    Min         (prod)
    RealDiv     (prod)
    FloorDiv    (prod)

"""
import tensorflow as tf
import numpy as np
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils
import tf_min.op_kernels.base_op as base_op
import tf_min.cpp_gen.cpp_gen as cpp_gen
import math


class AddOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Add, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Add"

    @staticmethod
    def description():
        return "Add operation, fully supports the elementwise addition tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input1_identifier = code_gen.c_safe_identifier(inputs[1].name)

        # If the input tensor sizes match then this is a simple elementwise addition
        # however if one of th tensors is smaller than the other then it will attempt to
        # `broadcast' the smaller tensor upto the size of the larger one
        input0_expression = input0_identifier
        input1_expression = input1_identifier

        input0_shape = tf_utils.np_tensor_shape(inputs[0])
        input1_shape = tf_utils.np_tensor_shape(inputs[1])

        if not np.array_equal(input0_shape, input1_shape):
            # print("Broadcasting needed in Add operation!")

            # print("Old input_0 (%s) input_1 (%s)" %
            #      (input0_shape, input1_shape))

            smaller = None
            # if one shape has lower rank than the other then pad the smaller rank
            # with size 1 dimensions
            if input1_shape.size < input0_shape.size:
                smaller = 1
                padding = np.ones(int(input0_shape.size - input1_shape.size), np.int)
                input1_shape = np.concatenate((padding, input1_shape))
                input1_expression += ".reshape(Eigen::array<int, %d>(%s))" % \
                                       (input1_shape.size,
                                        code_gen.ndarray_1d_to_literal(input1_shape))
            elif input0_shape.size < input1_shape.size:
                smaller = 0
                padding = np.ones(int(input1_shape.size - input0_shape.size), np.int)
                input0_shape = np.concatenate((padding, input0_shape))
                input0_expression += ".reshape(Eigen::array<int, %d>(%s))" % \
                                       (input0_shape.size,
                                        code_gen.ndarray_1d_to_literal(input0_shape))

            # print("New input_0 (%s) input_1 (%s)" %
            #      (input0_shape, input1_shape))

            broadcast_multiplier = np.ones(input1_shape.size, dtype=np.int)

            for d in range(input0_shape.size):

                if input0_shape[d] != input1_shape[d]:

                    # check error cases where dimensions are not universally smaller on one side
                    if (smaller == 0 and input0_shape[d] > input1_shape[d]) or\
                            (smaller == 1 and input1_shape[d] > input0_shape[d]):
                        print("Error: Add operation with non-broadcastable sized input tensors!")
                        return "// Error generating Add operation, non-broadcastable sized input tensors."

                    # check error case where dimenions are not equal or one of them is 1
                    if (input0_shape[d] < input1_shape[d] and input0_shape[d] != 1) or \
                            (input1_shape[d] < input0_shape[d] and input1_shape[d] != 1):
                        print("Error: Add operation with non-broadcastable sized input tensors!")
                        return "// Error generating Add operation, non-broadcastable sized input tensors."

                    # check if this dimension defines the smallest tensor
                    if smaller is None and input0_shape[d] < input1_shape[d]:
                        smaller = 0
                    elif smaller is None and input1_shape[d] < input0_shape[d]:
                        smaller = 1

                    # update the broadcast multiplier for this dimension
                    if smaller == 0:
                        broadcast_multiplier[d] = input1_shape[d]
                    else:
                        broadcast_multiplier[d] = input0_shape[d]

            broadcast_expression = ".broadcast(Eigen::array<int, %d>(%s))" % \
                                   (broadcast_multiplier.size,
                                    code_gen.ndarray_1d_to_literal(broadcast_multiplier))

            # update the expression for the smaller tensor
            if smaller == 0:
                input0_expression += broadcast_expression
            elif smaller == 1:
                input1_expression += broadcast_expression

        code = "%s %s + %s;" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input0_expression,
                input1_expression)

        return code


class SubOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Sub, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Sub"

    @staticmethod
    def description():
        return "Sub operation, fully supports the elementwise addition tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input1_identifier = code_gen.c_safe_identifier(inputs[1].name)

        code = "%s %s - %s;" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input0_identifier,
                input1_identifier)

        return code


class RealDivOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the RealDiv, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "RealDiv"

    @staticmethod
    def description():
        return "RealDiv elementwise floating point divide operation, "\
                "supports floating point elementwise division tensorflow operation. "\
                "TODO, add support for integer tensors."

    @staticmethod
    def status():
        return "Testing"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input1_identifier = code_gen.c_safe_identifier(inputs[1].name)

        # if the second argument is a scalar tensor
        input1_shape = tf_utils.np_tensor_shape(inputs[1])
        if len(input1_shape) == 0 or (len(input1_shape) == 1 and input1_shape[0] == 1):
            code = "%s %s / %s.constant(%s(0));" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input0_identifier,
                input0_identifier,
                input1_identifier)
        else:
            code = "%s %s / %s;" % \
                   (base_op.BaseOpKernel.output_assignment(tf_op, True),
                    input0_identifier,
                    input1_identifier)

        return code


class FloorDivOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the FloorDiv, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "FloorDiv"

    @staticmethod
    def description():
        return "FloorDiv performs integer division rounding towards negative infinity. " \
               "It should work with integer operands as well but this is not yet implemented"

    @staticmethod
    def status():
        return "Testing"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        output_identifier = code_gen.c_safe_identifier(tf_op.outputs[0].name)
        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input1_identifier = code_gen.c_safe_identifier(inputs[1].name)

        # if the second argument is a scalar tensor
        input1_shape = tf_utils.np_tensor_shape(inputs[1])
        if len(input1_shape) == 0 or (len(input1_shape) == 1 and input1_shape[0] == 1):

            input0_shape = tf_utils.np_tensor_shape(inputs[0])
            input0_size = np.prod(input0_shape)
            type = code_gen.get_c_dtype(inputs[0].dtype.base_dtype)

            code = cpp_gen.CodeBlock()
            target = "%s" % base_op.BaseOpKernel.output_assignment(tf_op, True, assignment=False)
            code.add_statement(cpp_gen.Statement(target.replace(";","").replace('\n', '')))

            # determine the type of expression to use. Either a division by the value of
            # a rank zero tensor, a division by a constant or a shift by a constant
            # in the case of power of two denominators

            if inputs[1].op.type == 'Const':
                const_value = tf_utils.get_const_scalar(inputs[1].op)

                if math.log2(const_value).is_integer():
                    expression = ">> %d" % int(math.log2(const_value))
                else:
                    expression = "/ (%s)%f" % (type, const_value)

            else:

                expression = "/ %s(0)" % input1_identifier

            for_loop = cpp_gen.LoopStatement("for", "int i=0; i<%d; ++i" % input0_size)
            for_loop.code.add_statement(cpp_gen.Statement(
                "((%s*)%s.data())[i] = ((%s*)%s.data())[i] %s" %
                    (type,
                     output_identifier,
                     type,
                     input0_identifier,
                     expression)
            ))

            code.add_statement(for_loop)
        else:
            code = "%s %s / %s;" % \
                   (base_op.BaseOpKernel.output_assignment(tf_op, True),
                    input0_identifier,
                    input1_identifier)

        return code


class SqrtOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Rsqrt, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Sqrt"

    @staticmethod
    def description():
        return "Sqrt operation, fully supports the square" \
               "root tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)

        code = "%s %s.sqrt();" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input0_identifier)

        return code


class TanhOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Sigmoid, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Tanh"

    @staticmethod
    def description():
        return "Tanh operation, supports float32 and float64 version of" \
               "the hyperbolic tangent tensorflow operation."

    @staticmethod
    def status():
        return "Testing"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        output_identifier = code_gen.c_safe_identifier(tf_op.outputs[0].name)

        type = code_gen.get_c_dtype(inputs[0].dtype.base_dtype)

        # calculate the number of elements in the input tensor
        input_shape = tf_utils.np_tensor_shape(inputs[0])
        element_count = 1
        for dim in input_shape:
            element_count *= dim

        # generate code to define the output tensor
        code = cpp_gen.CodeBlock()
        code.add_statement(cpp_gen.Statement(base_op.BaseOpKernel.output_assignment(tf_op,
                                                                                    eval=True,
                                                                                    assignment=False)))

        # generate a loop to perform a hyperbolic tan on each element, placing the result in the output tensor
        for_loop = cpp_gen.LoopStatement("for", "int i=0; i<%d; ++i" % element_count)
        for_loop.code.add_statement(cpp_gen.Statement(
            "((%s*)%s.data())[i] = std::tanh(((%s*)%s.data())[i])" %
            (type,
             output_identifier,
             type,
             input0_identifier)
        ))
        code.add_statement(for_loop)

        return code


class SigmoidOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Sigmoid, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Sigmoid"

    @staticmethod
    def description():
        return "Sigmoid operation, supports float32 and float64 version of" \
               "the sigmoid tensorflow operation."

    @staticmethod
    def status():
        return "Testing"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)

        dtype_string = code_gen.get_c_dtype(inputs[0].dtype.base_dtype)

        code ="%s (%s)1.0 / ((%s)1.0 + ((%s)0.0 - %s).exp())" %\
                (base_op.BaseOpKernel.output_assignment(tf_op, True),
                 dtype_string,
                 dtype_string,
                 dtype_string,
                 input0_identifier)

        return code


class RsqrtOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Rsqrt, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Rsqrt"

    @staticmethod
    def description():
        return "Rsqrt operation, fully supports the reciprical square" \
               "root tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)

        code = "%s %s.rsqrt();" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input0_identifier)

        return code


class MulOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Mul, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Mul"

    @staticmethod
    def description():
        return "Mul (elementwise multiplication) operation, fully supports" \
               "the tensorflow Mul operation."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        type = code_gen.get_c_dtype(tf_op.outputs[0].dtype.base_dtype)

        param_a_is_scalar = (inputs[0].shape.ndims == 0 or
                          (inputs[0].shape.ndims == 1 and
                           inputs[0].shape.dims[0] == 1))
        param_b_is_scalar = (inputs[1].shape.ndims == 0 or
                          (inputs[1].shape.ndims == 1 and
                           inputs[1].shape.dims[1] == 1))

        param_a_is_const = tf_utils.operation_is_constant(inputs[0].op)
        param_b_is_const = tf_utils.operation_is_constant(inputs[1].op)

        # if one of the inputs is a constant scalar then implement form 2
        if param_a_is_const and param_a_is_scalar:
            tensor_identifier = code_gen.c_safe_identifier(inputs[1].name)
            const_value = tf_utils.get_const_scalar(tf_utils.get_parent_of_tensor(tf_op.inputs[0]))

            return "%s %s * (%s)%s;" % \
                   (base_op.BaseOpKernel.output_assignment(tf_op, True),
                    tensor_identifier,
                    type,
                    str(const_value))

        if param_b_is_const and param_b_is_scalar:
            tensor_identifier = code_gen.c_safe_identifier(inputs[0].name)
            const_value = tf_utils.get_const_scalar(tf_utils.get_parent_of_tensor(tf_op.inputs[1]))

            return "%s %s * (%s)%s;" % \
                   (base_op.BaseOpKernel.output_assignment(tf_op, True),
                    tensor_identifier,
                    type,
                    str(const_value))

        # if both inputs are either tensors or not constants then generate form 1
        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input1_identifier = code_gen.c_safe_identifier(inputs[1].name)

        code = "%s %s * %s;" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, eval),
                input0_identifier,
                input1_identifier)
        return code


class MaximumOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Relu Operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Maximum"

    @staticmethod
    def description():
        return "Maximum (Elementwise) operation, fully supports all tensorflow parameters of this op."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input1_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input2_identifier = code_gen.c_safe_identifier(inputs[1].name)

        code = "%s %s.cwiseMax(%s);" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input1_identifier,
                input2_identifier)
        return code


class MinimumOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Relu Operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Minimum"

    @staticmethod
    def description():
        return "Minimum (Elementwise) operation, fully supports all tensorflow parameters of this op."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(tf_op):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input1_identifier = code_gen.c_safe_identifier(inputs[0].name)
        input2_identifier = code_gen.c_safe_identifier(inputs[1].name)

        code = "%s %s.cwiseMin(%s);" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input1_identifier,
                input2_identifier)
        return code
