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

    Module containing TensorFlow Utilities used by the TFMin library
"""
import tensorflow as tf
from tensorflow.python.framework.ops import get_gradient_function
import numpy as np
import struct
import tf_min.cpp_code_gen as code_gen


active_session = None


def get_parent_of_tensor(tensor):
    """Returns the parent operation of the given tensor object."""

    for opr in tensor.graph.get_operations():
        if tensor in opr.outputs:
            return opr

    raise ValueError("Error couldn't find an operation which was the parent of [%s] in the graph" % tensor.name)


def tensor_is_constant(tensor):
    """
    function to test if this operation is or depends solely on Const and Variable
    operations, and can therefore by reduced to a constant for inference.
    :param tensor: The tensor to check the dependecies of
    :return: True of the tensor depends on only Const and Variable ops, False otherwise
    """
    operation = get_parent_of_tensor(tensor)
    return operation_is_constant(operation)

def get_output_ops(op, ops_list, debug=False):
    """
    Function to return a list of the immediate output operations of the
    given operation. Skips identity operations
    :param op: operation to find outputs of
    :param ops_list: list of operations to search for outputs within
    :return: list of output operations
    """
    output_ops = []
    for list_op in ops_list:
      for input in list_op.inputs:
        input_op = input.op
        if input_op.type == "Identity":
          input_op = input_op.inputs[0].op
        if input_op == op:
          output_ops += [list_op]

    if debug:
      print("Found [%d] outputs of %s op \"%s\"" %
            (len(output_ops),
             op.type,
             op.name))
      for out_op in output_ops:
        print("  -> %s : %s" %
              (out_op.type,
               out_op.name))

    return output_ops


def operation_is_constant(operation):
    """
    function to test if this tensor depends solely on Const and Variable
    operations, and can therefore by reduced to a constant for inference.
    :param operation: The Operation to check the dependecies of
    :return: True of the operation depends on only Const and Variable ops, False otherwise
    """
    constant_op_types = ["Const", "Variable", "VariableV2"]

    # if this is a root op
    if len(operation.inputs) == 0:

        if operation.type in constant_op_types:
            return True
        else:
            return False

    # if not a root op then check all parent operations
    for input_tensor in operation.inputs:
        if not tensor_is_constant(input_tensor):
            return False

    # if the last loop didn't return then all inputs must be constant
    return True

def get_const_scalar(op):
    """Returns the scalar value of a scalar constant operation."""

    # enforce that this operation must resolve to a constant
    # i.e. no dependant upon input Placeholders.
    assert(operation_is_constant(op) == True)
    [value] = active_session.run([op.outputs[0]], {})

    # if the result was a tensor with a single element, then return the scalar
    # element instead
    if type(value) == np.ndarray and value.size == 1:
      value = value.flatten[0]

    return value

def get_const_tensor(op):
    """
    Returns a numpy.ndarray of a constant tensor stored in the graph
    :param op: Constant TF.Operation to evaluate
    :return: numpy.ndarray of constant tensor value
    """

    # enforce that this operation must resolve to a constant
    # i.e. no dependant upon input Placeholders.
    assert(operation_is_constant(op) == True)
    [value] = active_session.run([op.outputs[0]], {})

    return value

def np_tensor_shape(tensor):
    """Returns a 1 dimensional numpy array describing the shape of this Tensor object.

        Ignores undefined dimensions, so for example a tensor of shape ~,51,51,1
        would return a shape of 51,51,1
    """

    # if the tensor is a single scalar then return [ 1 ]. One D tensor of size 1
    if tensor.shape.dims is None:
        return np.array([1])

    shape = []
    for d in tensor.shape.dims:
        if d.value is not None:
            shape += [d.value]

    # if this tensor was variable batch_size only, it will have reduced to a rank zero tensor.
    # so promote it to a unary rank 1 tensor.
    if len(shape) == 0:
        shape += [1]

    return np.array(shape)


print_ops_printed = []


def show_parent_of_tensor(tensor, prefix=""):

    global print_ops_printed

    for opr in active_session.graph.get_operations():
        if tensor in opr.outputs:

            # if the parent operating is Indentity then leapfrog this to it's parent
            if opr.type == "Identity":
                show_parent_of_tensor(opr.inputs[0], prefix)

            else:   # not an Identity op so recurse normally

                # if this op has inputs then find if this operation has a gradients function
                grad_string = ''
                if len(opr.inputs) > 0:
                    grad_string = " \033[92mGrads\033[0m"
                    try:
                        if get_gradient_function(opr) is None:
                            grad_string = ' \033[91mNo Grads\033[0m'
                    except LookupError:
                        grad_string = ' \033[91mlookup Error\033[0m'

                print("%s  [\033[35m%s\033[0m \033[34m\"%s\"\033[0m] %s"
                      % (prefix, opr.type, opr.node_def.name, grad_string))

                # test if this operation has already been printed as prior to a previous op
                if opr in print_ops_printed:
                    print("%s   . . ." % prefix)
                    return opr

                print_ops_printed += [opr]

                for i in range(len(opr.inputs)):
                    parent_tensor = opr.inputs[i]
                    print("%s  |<\"%s\" with size %s and type %s>" %
                          (prefix,
                           parent_tensor.name,
                           str(parent_tensor.shape),
                           str(parent_tensor.dtype.base_dtype)))
                    if i == len(opr.inputs)-1:
                        show_parent_of_tensor(parent_tensor, (prefix + "   "))
                    else:
                        show_parent_of_tensor(parent_tensor, (prefix + "  |"))

            return opr


def write_numpy_array_data(file, array, indent, type='int'):

    if array.ndim == 1:
        format_str = "%d"  # % (precision, precision)

        file.write("{ ")
        for d in range(array.shape[0]):
            file.write(format_str % array[d])
            if d < array.shape[0]-1:
                file.write(", ")
        file.write(" }")

    else:
        file.write("{")
        condition = [True]
        for d in range(array.shape[0]):
            selection = array.compress(condition, axis=0).reshape(array.shape[1:])
            write_numpy_array_data(file, selection, indent+" ")
            if d < array.shape[0]-1:
                file.write(",\n%s " % indent)
            condition = [False] + condition
        file.write("}")


def write_numpy_array_c(file, identifier, array, data=True):

    c_data_type = code_gen.get_c_dtype(array.dtype)

    type = code_gen.get_c_dtype(array.dtype)
    is_float = type == "float" or type == "double" or type == "long double"

    if not is_float:
        declaration = "%s %s" % (c_data_type, identifier)

        for dim in range(array.ndim):
            declaration = ("%s[%d]" % (declaration, array.shape[dim]))

        if data:
            declaration = ("%s = " % declaration)
            indent = " " * len(declaration)
            file.write(declaration)

            write_numpy_array_data(file, array, indent, type)
            file.write(";\n\n")
        else:
            file.write(declaration + ";\n\n")

    # if the type of this numpy ndarray was float or double then also transcode this to hex and write it
    if is_float:
        array_flat = array.reshape(np.prod(array.shape))
        file.write("\n\nunsigned int %sHex[%d]" % (identifier, array_flat.shape[0]))

        if data:
            file.write(" = { ")

            for i in range(array_flat.shape[0]):
                int_of_float = struct.unpack('<I', struct.pack('<f', array_flat[i]))
                file.write("0x%X" % int_of_float)
                if i != array_flat.shape[0]-1:
                    file.write(", ")
            file.write("};\n\n")
        else:
            file.write(";\n\n")


def ensure_is_in_list(given_list, new_element):

    if new_element in given_list:
        return given_list
    else:
        return given_list + [new_element]


def ensure_operation_is_in_list(op_list, new_op):

    already_present = False

    for op in op_list:
        if op.name == new_op.name:
            already_present = True

    if already_present:
        return [op_list, False]
    else:
        return [op_list + [new_op], True]


def print_tensor(tensor, value, skip=False):

    print("--------------Tensor content [%s] --------------------" % tensor.name)

    if tensor.shape.ndims == 2:
        for d1 in range(tensor.shape.dims[1]):
            line = "D1 [ %2d ] " % d1
            for d0 in range(tensor.shape.dims[0]):
                line += "%10f, " % value.item((d0, d1))
            print(line)

    if tensor.shape.ndims == 3:
        for d2 in range(tensor.shape.dims[2]):
            print("D2 [ %2d ] --------------" % d2)
            for d1 in range(tensor.shape.dims[1]):
                line = "D1 [ %2d ] " % d1
                for d0 in range(tensor.shape.dims[0]):
                    line += "%10f, " % value.item((d0, d1, d2))
                print(line)

    if tensor.shape.ndims == 4 and not skip:
        for d3 in range(tensor.shape.dims[3]):
            print("D3 [ %2d ] ------------" % d3)
            for d2 in range(tensor.shape.dims[2]):
                print("D2 [ %2d ] --------------" % d2)
                for d1 in range(tensor.shape.dims[1]):
                    line = "D1 [ %2d ] " % d1
                    for d0 in range(tensor.shape.dims[0]):
                        line += "%10f, " % value.item((d0, d1, d2, d3))
                    print(line)

    if tensor.shape.ndims == 4 and skip:
        for d3 in range(tensor.shape.dims[3]):
            print("D2 [ %2d ] --------------" % d3)
            for d2 in range(tensor.shape.dims[2]):
                line = "D1 [ %2d ] " % d2
                if tensor.shape.dims[1] > 10:
                    for d1 in range(5):
                        line += "%10f, " % value.item((0, d1, d2, d3))
                    line += "..., "
                    for d1 in range(tensor.shape.dims[1] - 5, tensor.shape.dims[1]):
                        line += "%10f, " % value.item((0, d1, d2, d3))
                else:
                    for d1 in range(tensor.shape.dims[1]):
                        line += "%10f, " % value.item((0, d1, d2, d3))
                print(line)

    print("------------------------------------------------------")


list_input_placeholders = []
list_verification_tensors = []
list_training_tensors = []
list_operations = []


def build_graph_lists(tensors):

    global list_input_placeholders, list_verification_tensors, list_training_tensors, list_operations
    list_input_placeholders = []
    list_verification_tensors = []
    list_training_tensors = []
    list_operations = []

    for tensor in tensors:
        build_graph_lists_rec(tensor)

    return list_input_placeholders, list_verification_tensors, list_training_tensors, list_operations


def build_graph_lists_rec(tensor):

    global list_input_placeholders, list_verification_tensors, list_training_tensors, list_operations
    ignore_list = ["Const"]

    for opr in active_session.graph.get_operations():
        if opr not in list_operations:     # ignore if this operation has already been processed
            if tensor in opr.outputs:

                # if this is an input placeholder
                if opr.type == "Placeholder":
                    [list_input_placeholders,
                     _] = ensure_operation_is_in_list(list_input_placeholders, opr)
                    # if added:
                    #    print("-> Adding Placeholder %s" % opr.name)
                    list_verification_tensors =\
                        ensure_is_in_list(list_verification_tensors, opr.outputs[0])

                # if the parent operating is Indentity then leapfrog this to it's parent
                elif opr.type == "Identity":
                    build_graph_lists_rec(opr.inputs[0])

                # if this operation doesn't depend on placeholders it can be reduced to a constant
                # then add it as a weight and it will be resolved
                elif operation_is_constant(opr):
                    [list_training_tensors, _] = ensure_operation_is_in_list(list_training_tensors, tensor)

                # if this is a VariableV2 operation then evaluate and write the variable to the data file
                # elif opr.type == "VariableV2" or opr.type == "Const":
                #    [list_training_tensors,
                #     added] = ensure_operation_is_in_list(list_training_tensors, tensor)

                else:    # not an graph ending operation, so recurse normally
                    for i in range(len(opr.inputs)):
                        build_graph_lists_rec(opr.inputs[i])

                    # all inputs must now have been evaulated so add this operation to the list
                    if opr.type not in ignore_list:
                        [list_operations, added] = ensure_operation_is_in_list(list_operations, opr)
                        if added:
                            # print("--> Adding Operation [%s] %s" % (opr.type, opr.name))
                            if opr.type == "Switch":
                                # print("Switch op with %d inputs" % len(opr.inputs))
                                for idx, input in enumerate(opr.inputs):
                                    in_op = get_parent_of_tensor(input)
                                    if in_op.type == "Identity":
                                        in_op = get_parent_of_tensor(in_op.inputs[0])
                                    # print("[%2d] %s \"%s\"" % (idx, in_op.type, in_op.name))
