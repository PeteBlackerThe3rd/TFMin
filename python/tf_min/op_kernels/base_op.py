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

    Base operation kernels object of the TFMin library

    This object is used to derive all concrete op_kernels for generating C++
    equivalent code for TensorFlow operations.
"""
import tensorflow as tf
import numpy as np
import pprint as pp
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils


class BaseOpKernel:

    # static members, these are set by the Exporter
    # object when it is instantiated
    data_layout = "Eigen::ColMajor"
    use_memory_map = False
    evaluate_all = False

    def __init__(self):
        print("")

    @staticmethod
    def matches(tf_op):
        print("Error call to matches on base class with op [%s]!" % tf_op.type)
        return False

    @staticmethod
    def description():
        return "Error description called on base class!"

    @staticmethod
    def status():
        return "Base"

    # overridable methods which define if this operation can be define its
    # output using the address space of the input, either as a reference or
    # by overwriting the original value. Note these options are mutually
    # exclusive.
    @staticmethod
    def can_inplace_reference():
      return False

    @staticmethod
    def can_inplace_clobber():
      return False

    # overridable method which calculates the maximum safe overlap of the
    # input and output buffers of this operation. Defaults to None meaning
    # no safe overlap.
    @staticmethod
    def get_safe_overlap(_):
      return None

    # overridable method which indicates if this operation requires concrete
    # inputs or not. For example most Eigen based operation can take TensorMaps
    # or Eigen Operation objects, however operations which work directly on
    # data buffers require concrete TensorMap objects.
    @staticmethod
    def requires_concrete_inputs():
      return False

    @classmethod
    def gen_code(cls, tf_op, inputs):
        return "// Error gen_code called on base class with " \
               "op [%s - %d]!" % (tf_op.type, len(inputs))

    @classmethod
    def generate(cls, tf_op):

        # find input tensor list, leap-frogging any identity operations
        input_tensors = []
        for op_input in tf_op.inputs:
            if op_input.op.type == "Identity":
                input_tensors += [op_input.op.inputs[0]]
            else:
                input_tensors += [op_input]

        return cls.gen_code(tf_op, input_tensors)

    @staticmethod
    def output_assignment(tf_op, eval=True, idx=0, assignment=True):
        """ Words."""

        identifier = code_gen.c_safe_identifier(tf_op.outputs[idx].name)
        type = code_gen.get_c_dtype(tf_op.outputs[idx].dtype.base_dtype)
        rank = len(tf_utils.np_tensor_shape(tf_op.outputs[idx]))
        shape_np = tf_utils.np_tensor_shape(tf_op.outputs[idx])
        shape = code_gen.ndarray_1d_to_literal(shape_np, open='', close='')

        # -- special case --
        # if the result of this operation is a model output then
        # create a tensor map to the output buffer
        if hasattr(tf_op.outputs[idx], 'tfmin_output_identifier'):
            code = "\nEigen::TensorMap<Eigen::Tensor<%s, %d, %s>>" % \
                    (type,
                     rank,
                     BaseOpKernel.data_layout)
            code += " %s((%s*)%s, %s);" % \
                    (identifier,
                     type,
                     tf_op.outputs[idx].tfmin_output_identifier,
                     shape)

            if assignment:
                code += "\n%s = " % identifier

            return code

        # if this operation needs to be concrete or all ops are being evaluated
        if BaseOpKernel.evaluate_all or tf_op.tfmin_concrete_needed:
          eval = True

        # if evaluate is true then create a concrete tensor or
        # map of the operations result
        if eval:

            if BaseOpKernel.use_memory_map:

                precalculated_offset = None
                if hasattr(tf_op.outputs[idx], '_tfmin_memory_offset'):
                  precalculated_offset = tf_op.outputs[idx]._tfmin_memory_offset

                tensor_map_pointer = "(%s*)(memoryBlock + %s)" % \
                                     (type,
                                      precalculated_offset)

                # if no precalculated_offset was found then assume it is
                # safe to use the memory space of the input to this operation.
                # NOTE this will be safe is most cases but this may well explode
                # in some rare cases!! I apologise in advance if this has just
                # happened to you.
                if precalculated_offset is None:
                  input = tf_op.inputs[0]
                  if input.op.type == "Identity":
                      input = input.op.inputs[0]
                  tensor_map_pointer = "%s.data()" % \
                                       code_gen.c_safe_identifier(input.name)

                code = ("\nEigen::TensorMap<Eigen::Tensor<%s, %d, %s>>" %
                        (type,
                         rank,
                         BaseOpKernel.data_layout))

                code += " %s(%s, %s);" % \
                        (identifier,
                         tensor_map_pointer,
                         shape)
            else:
                code = "\nEigen::Tensor<%s, %d, %s> %s =" % \
                        (type,
                         rank,
                         data_layout,
                         identifier)

            if assignment:
                code += "\n%s.device(d) =" % identifier

            return code

        # if this operation is not being evaluated then create
        # an auto type so that the Eigen library produces a evaluator
        # object instead of a concrete tensor.
        else:
            code = "\nauto %s = " % identifier

            return code

    @staticmethod
    def print_operation_details(tf_op):
        """ print_operation_details, shows operation attributes,
        inputs and outputs details Prints out all the attributes
        of an operation as well as the size and types all any
        input and output tensors
        """

        inputs = []
        for op_input in tf_op.inputs:
            if op_input.op.type == "Identity":
                inputs += [op_input.op.inputs[0]]
            else:
                inputs += [op_input]

        # Brutally hacky way of getting the list of attributes
        # from a tensorflow.core.framework.node_def_pb2.NodeDef
        lines = str(tf_op.node_def).split("\n")
        attr_keys = []
        for l in lines:
            if l.startswith("  key: \""):
                key = l[8:100].replace("\"", "")
                attr_keys += [key]

        print("Attr keys are : " + str(attr_keys))

        print("Details of operation \"%s\" "
              "type [%s] -------------------" % (tf_op.name, tf_op.type))

        if len(attr_keys) > 0:
            print("Attributes:")
            for key in attr_keys:
                value = tf_op.get_attr(key)
                print("   \"%s\"\t\ttype(%s)\t\tvalue(%s)" %
                      (key,
                       str(type(value)),
                       str(value)))

        print("%d inputs:" % len(inputs))
        for idx, input in enumerate(inputs):
            input_parent_op = tf_utils.get_parent_of_tensor(input)
            print("   [%2d] \"%s\" %s rank(%d) %s : source op (\"%s\" - %s)" %
                  (idx,
                   input.name,
                   code_gen.get_c_dtype(input.dtype.base_dtype),
                   len(tf_utils.np_tensor_shape(input)),
                   tf_utils.np_tensor_shape(input),
                   input_parent_op.name,
                   input_parent_op.type))

        print("%d outputs:" % len(tf_op.outputs))
        for idx, output in enumerate(tf_op.outputs):
            print("   [%2d] \"%s\" %s rank(%d) %s" %
                  (idx,
                   output.name,
                   code_gen.get_c_dtype(output.dtype.base_dtype),
                   len(tf_utils.np_tensor_shape(output)),
                   tf_utils.np_tensor_shape(output)))
        print("--------------------------------------------------")
