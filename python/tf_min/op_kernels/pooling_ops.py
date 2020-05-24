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

    Pooling operation kernels of the TFMin library

    Operations supported
    -------------------------------
    MaxPool     (prod)
    AvgPool     (prod)

"""
import subprocess as subproc
import tensorflow as tf
import numpy as np
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils
import tf_min.op_kernels.base_op as base_op


class MaxPoolOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the MaxPool, convolutional operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "MaxPool" and \
               (tf_op.get_attr("padding") == b'SAME' or tf_op.get_attr("padding") == b'VALID')

    @staticmethod
    def description():
        return "MaxPool operation, fully supports the tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    # overridable method which calculates the maximum safe overlap of the
    # input and output buffers of this operation. Defaults to None meaning
    # no safe overlap.
    @staticmethod
    def get_safe_overlap(tf_op):

      input_shape = tf_op.inputs[0].shape
      output_shape = tf_op.outputs[0].shape

      get_buffer_overlap_cmd = "tfmin_buffer_overlap OpName=MaxPool"
      get_buffer_overlap_cmd += " TensorSize=input[%d,%d,%d]" %\
                                (input_shape.dims[1],
                                 input_shape.dims[2],
                                 input_shape.dims[3])
      get_buffer_overlap_cmd += " TensorSize=output[%d,%d,%d]" %\
                                (output_shape.dims[1],
                                 output_shape.dims[2],
                                 output_shape.dims[3])
      stride = np.array(tf_op.get_attr("strides"))
      get_buffer_overlap_cmd += " Param=stride_width,%d" % stride[1]
      get_buffer_overlap_cmd += " Param=stride_height,%d" % stride[2]
      pool_size = np.array(tf_op.get_attr("ksize"))
      get_buffer_overlap_cmd += " Param=pool_width,%d" % pool_size[1]
      get_buffer_overlap_cmd += " Param=pool_height,%d" % pool_size[2]
      get_buffer_overlap_cmd += " Param=padding,%s" %\
                                tf_op.get_attr("padding").decode("utf-8")

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

        k_size = np.array(tf_op.get_attr("ksize"))
        patch_rows = k_size[1]
        patch_cols = k_size[2]

        strides = np.array(tf_op.get_attr("strides"))
        row_stride = strides[1]
        col_stride = strides[2]

        padding = "Eigen::"
        if tf_op.get_attr("padding") == b'SAME':
            padding += "PADDING_SAME"
        elif tf_op.get_attr("padding") == b'VALID':
            padding += "PADDING_VALID"

        output_shape = code_gen.ndarray_1d_to_literal(tf_utils.np_tensor_shape(tf_op.outputs[0]))
        output_rank = len(tf_utils.np_tensor_shape(tf_op.outputs[0]))

        code = "\nauto %sPatches = %s.extract_image_patches(%d, %d, %d, %d, 1, 1, %s);" % \
               (input_identifier,
                input_identifier,
                patch_rows, patch_cols,
                row_stride, col_stride,
                padding)

        code += "%s %sPatches.maximum(Eigen::array<int, 2>({1, 2}))" % \
                (base_op.BaseOpKernel.output_assignment(tf_op, True),
                 input_identifier)
        code += "\n    .reshape(Eigen::array<int, %d>(%s));" % \
                (output_rank,
                 output_shape)

        return code


class AvgPoolOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the AvgPool, convolutional operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "AvgPool" and \
               (tf_op.get_attr("padding") == b'SAME' or tf_op.get_attr("padding") == b'VALID')

    @staticmethod
    def description():
        return "AvgPool operation, fully supports the tensorflow operation."

    @staticmethod
    def status():
        return "Production"

    @classmethod
    def gen_code(cls, tf_op, inputs):
        input_identifier = code_gen.c_safe_identifier(inputs[0].name)

        k_size = np.array(tf_op.get_attr("ksize"))
        patch_rows = k_size[1]
        patch_cols = k_size[2]

        strides = np.array(tf_op.get_attr("strides"))
        row_stride = strides[1]
        col_stride = strides[2]

        padding = "Eigen::"
        if tf_op.get_attr("padding") == b'SAME':
            padding += "PADDING_SAME"
        elif tf_op.get_attr("padding") == b'VALID':
            padding += "PADDING_VALID"

        output_shape = code_gen.ndarray_1d_to_literal(tf_utils.np_tensor_shape(tf_op.outputs[0]))
        output_rank = len(tf_utils.np_tensor_shape(tf_op.outputs[0]))

        code = "\nauto %sPatches = %s.extract_image_patches(%d, %d, %d, %d, 1, 1, %s);" % \
               (input_identifier,
                input_identifier,
                patch_rows, patch_cols,
                row_stride, col_stride,
                padding)

        code += "%s %sPatches.mean(Eigen::array<int, 2>({1, 2}))" % \
                (base_op.BaseOpKernel.output_assignment(tf_op, True),
                 input_identifier)
        code += "\n    .reshape(Eigen::array<int, %d>(%s));" % \
                (output_rank,
                 output_shape)

        return code
