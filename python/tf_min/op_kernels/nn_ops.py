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

    Neural Network operation kernels of the TFMin library

    Operations supported
    -------------------------------
    BiasAdd     (prod)
    SoftMax     (prod)
    Conv2D      (prod)
    MatMul      (prod)

"""
import subprocess as subproc
import tensorflow as tf
import numpy as np
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils
import tf_min.op_kernels.base_op as base_op
import tf_min.cpp_gen.cpp_gen as cpp_gen


class BiasAddOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the BiasAdd, convolutional operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "BiasAdd"

    @staticmethod
    def description():
        return "BiasAdd operation, fully supports the tensorflow operation."

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

        # If the bias tensor needs to be cast into the same time as the input
        bias_cast = ""

        # if the bias tensor needs to be broadcast into the same shape as the input
        bias_broadcast = ""
        input0_shape = tf_utils.np_tensor_shape(inputs[0])
        input1_shape = tf_utils.np_tensor_shape(inputs[1])
        shapes_match = False
        if len(input0_shape) == len(input1_shape):
            shapes_match = True
            for i in range(len(input0_shape)):
                if input0_shape[i] != input1_shape[i]:
                    shapes_match = False
        if not shapes_match:

            broadcast_shape = tf_utils.np_tensor_shape(inputs[0])
            broadcast_shape[len(broadcast_shape)-1] = 1

            reshape_shape = np.array(([1] * (len(broadcast_shape)-1)) + [input1_shape[0]])
            bias_broadcast = "\n    .reshape(Eigen::array<int, %d>(%s))" % \
                             (len(reshape_shape),
                              code_gen.ndarray_1d_to_literal(reshape_shape))
            bias_broadcast += "\n        .broadcast(Eigen::array<int, %d>(%s))" % \
                              (len(broadcast_shape),
                               code_gen.ndarray_1d_to_literal(broadcast_shape))

        code = "%s %s + %s%s%s;" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, False),
                input0_identifier,
                input1_identifier,
                bias_cast,
                bias_broadcast)
        return code


class SoftmaxOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Softmax, elementwise operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Softmax"

    @staticmethod
    def description():
        return "Softmax operation, fully supports the softmax tensorflow operation."

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
        type = code_gen.get_c_dtype(inputs[0].dtype.base_dtype)

        code = "\nauto %sExp = %s.exp();" % \
               (input0_identifier,
                input0_identifier)

        code += "\nEigen::Tensor<%s, 0, %s> %sExpSum = %sExp.sum();" % \
                (type,
                 base_op.BaseOpKernel.data_layout,
                 input0_identifier,
                 input0_identifier)

        code += "%s %sExp / %sExp.constant(%sExpSum(0));" % \
                (base_op.BaseOpKernel.output_assignment(tf_op, True),
                 input0_identifier,
                 input0_identifier,
                 input0_identifier)
        return code


class ArgMaxOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the ArgMax, elementwise operation."""

    @staticmethod
    def matches(tf_op):

        if tf_op.type != "ArgMax":
            return False

        return True

    @staticmethod
    def description():
        return "ArgMax operation, supports ArgMax of 1 dimensional tensors. TODO extend to N dimensions."

    @staticmethod
    def status():
        return "Development"

    @staticmethod
    def requires_concrete_inputs():
      return True

    @classmethod
    def gen_code(cls, tf_op, inputs):

        # super().print_operation_details(tf_op)

        identifier = code_gen.c_safe_identifier(tf_op.outputs[0].name)
        input0_identifier = code_gen.c_safe_identifier(inputs[0].name)
        type = code_gen.get_c_dtype(inputs[0].dtype.base_dtype)
        input_shape = tf_utils.np_tensor_shape(inputs[0])

        code = cpp_gen.CodeBlock()

        assignment = base_op.BaseOpKernel.output_assignment(tf_op, True, assignment=False)
        if assignment[-1] == ';':
            assignment = assignment[:-1]
        assignment = assignment.replace('\n', '')
        code.add_statement(cpp_gen.Statement(str(assignment)))

        code.add_statement(cpp_gen.Statement("%s %s_max = std::numeric_limits<%s>::min()" %
                                             (type, identifier, type)))

        code.add_statement(cpp_gen.Statement("%s(0) = 0" % identifier))

        if_statement = cpp_gen.IfStatement("%s(%s_it) > %s_max" % (input0_identifier, identifier, identifier))
        if_statement.if_code.add_statement(cpp_gen.Statement("%s_max = %s(%s_it)" %
                                                             (identifier,
                                                              input0_identifier,
                                                              identifier)))
        if_statement.if_code.add_statement(cpp_gen.Statement("%s(0) = %s_it" % (identifier, identifier)))

        for_loop = cpp_gen.LoopStatement("for", "long %s_it=0; %s_it<%d; ++%s_it" %
                                         (identifier,
                                          identifier,
                                          input_shape[0],
                                          identifier))
        for_loop.code.add_statement(if_statement)

        code.add_statement(for_loop)

        return code


class Conv2DOpKernel(base_op.BaseOpKernel):
  """Code generator kernel for the Conv2D, convolutional operation.
     New version based upon the reference implmenetation from TF lite.
  """

  @staticmethod
  def matches(tf_op):
    return tf_op.type == "Conv2D"

  @staticmethod
  def description():
    return "Conv2D operation, [TF lite based] supports no-padding form of the Conv2D tensorflow operation."

  @staticmethod
  def status():
    return "Production"

  @staticmethod
  def requires_concrete_inputs():
    return True

  # overridable method which calculates the maximum safe overlap of the
  # input and output buffers of this operation. Defaults to None meaning
  # no safe overlap.
  @staticmethod
  def get_safe_overlap(tf_op):

    input_shape = tf_op.inputs[0].shape
    filter_shape = tf_op.inputs[1].shape
    output_shape = tf_op.outputs[0].shape

    get_buffer_overlap_cmd = "tfmin_buffer_overlap OpName=Conv2d"
    get_buffer_overlap_cmd += " TensorSize=input[%d,%d,%d]" %\
                              (input_shape.dims[1],
                               input_shape.dims[2],
                               input_shape.dims[3])
    get_buffer_overlap_cmd += " TensorSize=filter[%d,%d,%d,%d]" %\
                              (filter_shape.dims[0],
                               filter_shape.dims[1],
                               filter_shape.dims[2],
                               filter_shape.dims[3])
    get_buffer_overlap_cmd += " TensorSize=output[%d,%d,%d]" %\
                              (output_shape.dims[1],
                               output_shape.dims[2],
                               output_shape.dims[3])
    stride = np.array(tf_op.get_attr("strides"))
    get_buffer_overlap_cmd += " Param=stride_width,%d" % stride[1]
    get_buffer_overlap_cmd += " Param=stride_height,%d" % stride[2]
    get_buffer_overlap_cmd += " Param=padding,%s" %\
                              tf_op.get_attr("padding").decode("utf-8")
    # TODO Pete Add support for dialated convolutions
    get_buffer_overlap_cmd += " Param=dilation_width_factor,1"
    get_buffer_overlap_cmd += " Param=dilation_height_factor,1"

    print("About to run shell command [%s]" % get_buffer_overlap_cmd)

    safe_overlap = None
    try:
        safe_overlap_bytes = subproc.check_output(get_buffer_overlap_cmd,
                                                  shell=True)
        safe_overlap = int(safe_overlap_bytes.decode("utf-8"))
    except subproc.CalledProcessError as e:
        print("Error running external tfmin_buffer_overlap tool!")

    return safe_overlap

  @classmethod
  def gen_code(cls, tf_op, inputs):

    input_identifier = code_gen.c_safe_identifier(inputs[0].name)
    filter_identifier = code_gen.c_safe_identifier(inputs[1].name)
    output_identifier = code_gen.c_safe_identifier(tf_op.outputs[0].name)

    padding = "Eigen::"
    if tf_op.get_attr("padding") == b'SAME':
      padding += "PADDING_SAME"
    elif tf_op.get_attr("padding") == b'VALID':
      padding += "PADDING_VALID"

    filter_stride = np.array(tf_op.get_attr("strides"))
    row_stride = filter_stride[1]
    col_stride = filter_stride[2]

    row_dilation = 1
    col_dilation = 1

    code = base_op.BaseOpKernel.output_assignment(tf_op, eval=True, idx=0, assignment=False)

    code += "TFMin::ConvTFL::conv(%s, %s, %s, %s, %d, %d, %d, %d)" % \
                (input_identifier,
                 filter_identifier,
                 output_identifier,
                 padding,
                 col_stride,
                 row_stride,
                 col_dilation,
                 row_dilation
                 )

    return code

"""
class Conv2DOpKernel(base_op.BaseOpKernel):
    
    Code generator kernel for the Conv2D, convolutional operation.
    Original Eigen based version of this operation.

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Conv2D"

    @staticmethod
    def description():
        return "Conv2D operation, [Eigen based] supports no-padding form of the Conv2D tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input_identifier = code_gen.c_safe_identifier(inputs[0].name)
        filter_identifier = code_gen.c_safe_identifier(inputs[1].name)

        # generate filters reshape (dim0, product of dims1,2,3)
        filter_shape_in = tf_utils.np_tensor_shape(inputs[1])
        filter_count = filter_shape_in[3]
        filter_channels = filter_shape_in[2]
        filter_reshape_d0 = filter_shape_in[0] * filter_shape_in[1] * filter_shape_in[2]
        filter_reshape = "Eigen::array<int, 2>({%d, %d})" % (filter_reshape_d0, filter_count)

        output_shape = code_gen.ndarray_1d_to_literal(tf_utils.np_tensor_shape(tf_op.outputs[0]))
        output_rank = len(tf_utils.np_tensor_shape(tf_op.outputs[0]))
        output_reshape = "Eigen::array<int, %d>(%s)" % \
                         (output_rank,
                          output_shape)

        filter_rows = filter_shape_in[0]
        filter_cols = filter_shape_in[1]
        filter_stride = np.array(tf_op.get_attr("strides"))
        filter_row_stride = filter_stride[1]
        filter_col_stride = filter_stride[2]

        padding = "Eigen::"
        if tf_op.get_attr("padding") == b'SAME':
            padding += "PADDING_SAME"
        elif tf_op.get_attr("padding") == b'VALID':
            padding += "PADDING_VALID"

        image_patches = "%d, %d, %d, %d, 1, 1, %s" % \
                        (filter_rows,
                         filter_cols,
                         filter_row_stride,
                         filter_col_stride,
                         padding)

        output_rows = tf_utils.np_tensor_shape(tf_op.outputs[0])[0]
        output_cols = tf_utils.np_tensor_shape(tf_op.outputs[0])[1]

        pre_contract_dims = "Eigen::array<int, 2>({%d, %d})" % \
                            (output_rows * output_cols,
                             filter_channels * filter_rows * filter_cols)

        code = ""

        if base_op.BaseOpKernel.data_layout == 'Eigen::ColMajor':

            contraction = "%s.reshape(%s), matMulDims" % \
                          (filter_identifier,
                           filter_reshape)

            code += "%s %s\n    .extract_image_patches(%s)\n    .reshape(%s)\n    .contract(%s)\n    .reshape(%s);" % \
                    (base_op.BaseOpKernel.output_assignment(tf_op, True),
                     input_identifier,
                     image_patches,
                     pre_contract_dims,
                     contraction,
                     output_reshape)

        else:  # data_layout == 'Eigen::RowMajor':

            contraction = "%s.reshape(%s), matMulDims" % \
                          (filter_identifier,
                           filter_reshape)

            code += "%s %s\n    .extract_image_patches(%s)\n    .reshape(%s)\n    .contract(%s)\n    .reshape(%s);" % \
                    (base_op.BaseOpKernel.output_assignment(tf_op, True),
                     input_identifier,
                     image_patches,
                     pre_contract_dims,
                     contraction,
                     output_reshape)

        return code"""


class MatMulOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Conv2D, convolutional operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "MatMul"

    @staticmethod
    def description():
        return "MatMul operation, fully supports the tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        # generate source information used to generate MatMul statement
        input0_statement = code_gen.c_safe_identifier(inputs[0].name)
        input1_statement = code_gen.c_safe_identifier(inputs[1].name)
        input0_shape = tf_utils.np_tensor_shape(inputs[0])
        input1_shape = tf_utils.np_tensor_shape(inputs[1])

        # if the inputs include vectors then reshape them to rank 2
        reshaped = False
        if len(input0_shape) == 1:
            input0_statement += ".reshape(Eigen::array<int,2>({1,%d}))" % input0_shape[0]
            reshaped = True
        if len(input1_shape) == 1:
            input1_statement += ".reshape(Eigen::array<int,2>({%d,1}))" % input1_shape[1]
            reshaped = True

        final_reshape = ""
        if reshaped:
            output_shape = tf_utils.np_tensor_shape(tf_op.outputs[0])
            final_reshape = ".reshape(Eigen::array<int,1>({%d}))" % output_shape[0]

        code = "%s %s.contract(%s, matMulDims)%s;" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input0_statement,
                input1_statement,
                final_reshape)
        return code
