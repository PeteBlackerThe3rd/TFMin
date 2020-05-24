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

    Beta Testing operation kernels of the TFMin library

    Operations under development
    -------------------------------
    FakeQuantWithMinMaxVars         (dev)
    DepthwiseConv2dNative           (dev)

"""
import subprocess as subproc
import tensorflow as tf
import numpy as np
import math
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils
import tf_min.op_kernels.base_op as base_op
import tf_min.cpp_gen.cpp_gen as cpp_gen


class FakeQuantWithMinMaxVarsOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the FakeQuantWithMinMaxVars, elementwise linear range quantization."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "FakeQuantWithMinMaxVars"

    @staticmethod
    def description():
        return "FakeQuantWithMinMaxVars operation, fully supports the tensorflow quantization operation."

    @staticmethod
    def status():
        return "Development"

    @classmethod
    def gen_code(cls, tf_op, inputs):

        narrow_range = tf_op.get_attr("narrow_range")
        num_bits = tf_op.get_attr("num_bits")

        range_min = tf_utils.get_const_scalar(
            tf_utils.get_parent_of_tensor(inputs[1]))
        range_max = tf_utils.get_const_scalar(
            tf_utils.get_parent_of_tensor(inputs[2]))

        print("narrow [%s] num_bits [%d] range [%f - %f]" %
              (narrow_range,
               num_bits,
               range_min,
               range_max))

        # output_shape = tf_utils.np_tensor_shape(tf_op.outputs[0])
        # input_identifier = code_gen.c_safe_identifier(tf_op.inputs[0].name)

        # code = "%s %s.reshape(Eigen::array<int, %d>(%s));" % \
        #       (core_ops.output_assignment(tf_op, eval),
        #        input_identifier,
        #        len(output_shape),
        #        code_gen.ndarray_1d_to_literal(output_shape))

        return "// TODO"


class DepthwiseConv2dNativeOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the DepthwiseConv2dNative,
       convolutional operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "DepthwiseConv2dNative"

    @staticmethod
    def description():
        return "DepthwiseConv2dNative operation, fully " \
               "supports softmax tensorflow operation."

    @staticmethod
    def status():
        return "Development"

    # overridable method which calculates the maximum safe overlap of the
    # input and output buffers of this operation. Defaults to None meaning
    # no safe overlap.
    @staticmethod
    def get_safe_overlap(tf_op):

        base_op.BaseOpKernel.print_operation_details(tf_op)

        input_shape = tf_op.inputs[0].shape
        filter_shape = tf_op.inputs[1].shape
        output_shape = tf_op.outputs[0].shape

        input_w = input_shape[1].value
        input_h = input_shape[2].value
        input_d = input_shape[3].value

        output_w = output_shape[1].value
        output_h = output_shape[2].value
        output_d = output_shape[3].value

        kernel_w = filter_shape[1].value
        kernel_h = filter_shape[2].value
        kernel_c = filter_shape[0].value

        stride_w = tf_op.get_attr("strides")[0]
        stride_h = tf_op.get_attr("strides")[1]
        dilation_w = tf_op.get_attr("dilations")[0]
        dilation_h = tf_op.get_attr("dilations")[1]


        print("input_w is a : %s" % str(type(input_w)))

        print("Stride_w is a : %s" % str(type(stride_w)))
        print("dilation_w is a : %s" % str(type(dilation_w)))


        padding_h = math.floor((output_h * stride_h - stride_h + kernel_h *
                                dilation_h - dilation_h - input_h + 1) / 2)
        padding_w = math.floor((output_w * stride_w - stride_w + kernel_w *
                                dilation_w - dilation_w - input_w + 1) / 2)

        """print("Using anlytic solution to depthwise conv safe overlap.")
  
        print("input whc %d %d %d" % (input_w, input_h, input_d))
        print("output wh %d %d" % (output_w, output_h))
        print("kernel whc %d %d %d" % (kernel_w, kernel_h, kernel_c))
        print("stride wh %d %d" % (stride_w, stride_h))
        print("dilation wh %d %d" % (dilation_w, dilation_h))
        print("padding wh %d %d" % (padding_w, padding_h))"""

        a = (stride_h * input_w) / float(output_w * kernel_c)
        b = (output_w * stride_w - padding_h * input_w - stride_h *
             input_w - stride_w - padding_w + 1) * input_d
        ic = output_h * output_w * input_d * kernel_c

        # print("a = %f, b = %d, ic = %d" % (a, b, ic))

        # output_buffer_size = model.tensor_memory_sizes[output_tensor_ids[0]]
        # output_dtype = model.tensor_dtype_idxs[output_tensor_ids[0]]
        output_dtype_size = tf_op.outputs[0].dtype.size
        print("Datatype size of output is %d [%s]" %
              (output_dtype_size,
               str(tf_op.outputs[0].dtype)))

        output_buffer_size = output_w * output_h * output_d * output_dtype_size

        safe_overlap = output_buffer_size + (
                  min(b / a, a * ic + b - ic) * output_dtype_size)

        """get_buffer_overlap_cmd = "tfmin_buffer_overlap OpName=DepthwiseConv"
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

        print("About to run shell command [%s]" % get_buffer_overlap_cmd)

        safe_overlap = None
        try:
            safe_overlap_bytes = subproc.check_output(get_buffer_overlap_cmd,
                                                      shell=True)
            print("depthwise conv overlap was [%s]" % safe_overlap_bytes.decode())
            safe_overlap = int(safe_overlap_bytes.decode("utf-8"))
        except subproc.CalledProcessError as e:
            print("Error running external tfmin_buffer_overlap tool!")

        return safe_overlap"""

        return safe_overlap

    @classmethod
    def gen_code(cls, tf_op, inputs):

        output_identifier = code_gen.c_safe_identifier(tf_op.outputs[0].name)
        input_identifier = code_gen.c_safe_identifier(inputs[0].name)
        filter_identifier = code_gen.c_safe_identifier(inputs[1].name)

        filter_stride = np.array(tf_op.get_attr("strides"))
        row_stride = filter_stride[1]
        col_stride = filter_stride[2]

        code = base_op.BaseOpKernel.output_assignment(tf_op, eval=True, idx=0,
                                                      assignment=False)

        code += "TFMin::DepthwiseConvFloatTFL::depthwiseConv(" \
                "%s, %s, %s, %d, %d)" % \
                (input_identifier,
                 filter_identifier,
                 output_identifier,
                 col_stride,
                 row_stride
                 )

        return code
