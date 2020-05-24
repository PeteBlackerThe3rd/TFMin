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

    Activation operation kernels of the TFMin library

    Operations supported
    -------------------------------
    Relu        (prod)
    LeakyRelu   (prod)
    Relu6       (prod)

"""
import tensorflow as tf
import numpy as np
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils
import tf_min.op_kernels.base_op as base_op


class ReluOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Relu Operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Relu"

    @staticmethod
    def description():
        return "Relu (Rectified Linear Unit) operation, fully " \
               "supports all tensorflow parameters of this op."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(_):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input_identifier = code_gen.c_safe_identifier(inputs[0].name)
        type = code_gen.get_c_dtype(tf_op.outputs[0].dtype.base_dtype)

        code = "%s %s.cwiseMax((%s)0);" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input_identifier,
                type)
        return code


class LeakyReluOpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Relu Operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "LeakyRelu"

    @staticmethod
    def description():
        return "LeakyRelu (Leaky rectified Linear Unit) operation, fully " \
               "supports all tensorflow parameters of this op."

    @staticmethod
    def status():
        return "Testing"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(_):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        # super().print_operation_details(tf_op)
        alpha = tf_op.get_attr("alpha")

        input_identifier = code_gen.c_safe_identifier(inputs[0].name)
        type = code_gen.get_c_dtype(tf_op.outputs[0].dtype.base_dtype)

        code = "%s %s.cwiseMax(%s * (%s)%f);" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input_identifier,
                input_identifier,
                type,
                alpha)
        return code


class Relu6OpKernel(base_op.BaseOpKernel):
    """Code generator kernel for the Relu6 Operation."""

    @staticmethod
    def matches(tf_op):
        return tf_op.type == "Relu6"

    @staticmethod
    def description():
        return "Relu6 [min(max(features, 0), 6)] operation, " \
               "fully supports tensorflow op."

    @staticmethod
    def status():
        return "Production"

    @staticmethod
    def can_inplace_clobber():
      return True

    @staticmethod
    def get_safe_overlap(_):
      return 0

    @classmethod
    def gen_code(cls, tf_op, inputs):

        input_identifier = code_gen.c_safe_identifier(inputs[0].name)
        type = code_gen.get_c_dtype(tf_op.outputs[0].dtype.base_dtype)

        six_constant = 6

        # if this is operating on quantised data then the
        # six_constant will need multplying by the correct power of 2.

        code = "%s %s.cwiseMax((%s)0).cwiseMin((%s)%d);" % \
               (base_op.BaseOpKernel.output_assignment(tf_op, True),
                input_identifier,
                type,
                type,
                six_constant)
        return code
