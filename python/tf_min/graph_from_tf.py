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

    This module contains the Tensorflow v1.x importer which generates a 
    TFMin Graph object from a Tensorflow interactive session. 
"""
import tensorflow as tf
import numpy as np
import tf_min.graph as tg
import tf_min.types as types
from tf_min.activation_fns import ActType


TF_OP_TRANSLATIONS = {'DepthwiseConv2DNative': 'DepthwiseConv2D'}


def tf_type_to_tfmin(tf_type):
  conversion = {tf.float32: types.TenDType.FLOAT32,
                tf.float64: types.TenDType.FLOAT64,
                tf.int8: types.TenDType.INT8,
                tf.int16: types.TenDType.INT16,
                tf.int32: types.TenDType.INT32,
                tf.int64: types.TenDType.INT64,
                tf.uint8: types.TenDType.UINT8,
                tf.uint16: types.TenDType.UINT16,
                tf.uint32: types.TenDType.UINT32,
                tf.uint64: types.TenDType.UINT64}
  if tf_type not in conversion.keys():
    raise TypeError("type %s not supported by TFMin" % tf_type)
  return conversion[tf_type]


def tf_shape_to_tfmin(tf_shape):
  np_shape = []
  for dim in tf_shape.dims:
    if dim.value is None:
      np_shape.append(-1)
    else:
      np_shape.append(dim.value)
  return tg.TensorShape(np_shape)


def get_opr_input_params(tf_opr, sess):
  """
  Function to extract effective parameterw which TF insists on passing as
  tensor inputs. This is really messey, so this function extracts their
  values adds them as const parameters instead and removes the link to the
  input tensor
  :param tf_opr:
  :param sess: tf.InteractiveSession used to resolve contant values
  :return:
  """
  additional_params = {}
  # crude copy of inputs array to create a genunine python list instead.
  valid_inputs = []
  for input in tf_opr.inputs:
    valid_inputs.append(input)

  if tf_opr.type == "ArgMax":
    [value] = sess.run([tf_opr.inputs[1]], {})
    # print("Got a value of [%s] type [%s]" % (value, str(type(value))))
    additional_params["dim"] = value
    valid_inputs.remove(valid_inputs[1])
  elif tf_opr.type == "Reshape":
    [shape] = sess.run([tf_opr.inputs[1]], {})
    additional_params['shape'] = shape
    valid_inputs.remove(valid_inputs[1])

  return [additional_params, valid_inputs]


def add_tf_tensor(graph, tf_tensor, sess):
    # skip if this tensor has already been added
    for tensor in graph.tensors:
        if tensor.label == tf_tensor.name:
            return tensor

    new_tensor = tf_to_tensor(tf_tensor, sess)
    new_tensor.creating_op = add_tf_op(graph, tf_tensor.op, sess)
    new_tensor.creating_op.outputs.append(new_tensor)
    graph.tensors.append(new_tensor)
    return new_tensor


def add_tf_op(graph, tf_op, sess):
  # skip if this op has already been added
  for opr in graph.ops:
    if opr.label == tf_op.name:
      return opr

  # add this operation and input tensors
  new_op = tf_to_operation(tf_op, sess)
  [_, valid_inputs] = get_opr_input_params(tf_op, sess)
  for tensor in valid_inputs:
    new_tensor = add_tf_tensor(graph, tensor, sess)
    new_op.inputs.append(new_tensor)
    new_tensor.dependent_ops.append(new_op)
  graph.ops.append(new_op)

  return new_op


def mark_inputs(self):
  ops_to_remove = []
  for opr in self.ops:
    if opr.type == "Placeholder":
      input_tensor = opr.outputs[0]
      input_tensor.creating_op = None
      input_tensor.type = tg.TenType.INPUT
      ops_to_remove.append(opr)

  for opr in ops_to_remove:
    self.ops.remove(self.ops[self.get_opr_idx(opr)])


def mark_weights(self, sess):
  weight_op_types = ["Const", "Variable", "VariableV2"]
  ops_to_remove = []
  for opr in self.ops:
    if opr.type in weight_op_types:
      weights_tensor = opr.outputs[0]
      weights_tensor.creating_op = None
      weights_tensor.type = tg.TenType.CONSTANT
      tf_tensor = sess.graph.get_tensor_by_name(weights_tensor.label)
      value = sess.run([tf_tensor], {})
      weights_tensor.value = value
      ops_to_remove.append(opr)

  for opr in ops_to_remove:
    self.ops.remove(self.ops[self.get_opr_idx(opr)])


def get_parent_of_tensor(tensor):
    """Returns the parent operation of the given tensor object."""

    for opr in tensor.graph.get_operations():
      if tensor in opr.outputs:
        return opr

    raise ValueError("Error couldn't find an operation which was "
                     "the parent of [%s] in the graph" % tensor.name)


def tf_to_tensor(tf_tensor, sess):
    new_tensor = tg.Tensor()
    new_tensor.label = tf_tensor.name
    # numpy_type = np.dtype(tf_tensor.dtype.as_numpy_dtype)
    # print("tf_to_tensor np_type is [%s]" % type(numpy_type))
    new_tensor.d_type = tf_type_to_tfmin(tf_tensor.dtype.base_dtype)
    new_tensor.shape = tf_shape_to_tfmin(tf_tensor.shape)
    new_tensor.type = tg.TenType.INTERMEDIATE

    # if the tensor is a constant or variable then capture it's value
    constant_op_types = ["Const", "Variable", "VariableV2"]
    parent_op_type = get_parent_of_tensor(tf_tensor).type
    if parent_op_type in constant_op_types:
        [tensor_value] = sess.run([tf_tensor], feed_dict={})
        print("Getting value of tensor [%s] : \n%s" %
              (tf_tensor.name, tensor_value))
        print("type of value [%s]" % type(tensor_value))
        new_tensor.value = tensor_value
        new_tensor.type = tg.TenType.CONSTANT
        print("type of tensor.value [%s]" % type(new_tensor.value))

    return new_tensor


def tf_to_operation(tf_opr, sess):
    opr = tg.Operation()
    opr.type = tf_opr.type
    opr.label = tf_opr.name

    # translate operation type from tflite to TFMin
    if opr.type in TF_OP_TRANSLATIONS:
      opr.type = TF_OP_TRANSLATIONS[self.type]

    # Brutally hacky way of getting the list of attributes
    # from a tensorflow.core.framework.node_def_pb2.NodeDef
    lines = str(tf_opr.node_def).split("\n")
    for line in lines:
      if line.startswith("  key: \""):
        key = line[8:100].replace("\"", "")

        # add recognised parameter types
        if key == "dtype":
          opr.params['dtype'] = tf_type_to_tfmin(
            tf_opr.get_attr(key).base_dtype
          )
        elif key == "dilations":
          opr.params['dilation_height_factor'] = tf_opr.get_attr(key)[1]
          opr.params['dilation_width_factor'] = tf_opr.get_attr(key)[2]
        elif key == "padding":
          opr.params['padding'] = tf_opr.get_attr(key).decode('utf-8')
        elif key == "strides":
          opr.params['stride_height'] = tf_opr.get_attr(key)[1]
          opr.params['stride_width'] = tf_opr.get_attr(key)[2]
        elif key == "ksize":
          opr.params['kernel_height'] = tf_opr.get_attr(key)[1]
          opr.params['kernel_width'] = tf_opr.get_attr(key)[2]

    [add_params, _] = get_opr_input_params(tf_opr, sess)
    opr.params.update(add_params)
    return opr


def remove_identity_ops(graph):
    """
    Function to remove all pointless identity operations from this graph.
    For some reason tensorflow puts them all over the place to join things
    together.
    :param graph: The input graph which is modified in place.
    :return: The number of identity operations removed
    """
    removed_count = 0

    items_to_remove = []

    for opr in graph.ops:
        # if an identity operation is found link it's input tensor to the
        # operation its output tensor links to, and add the op and it's output
        # to the items to remove list
        if opr.type == "Identity":

            # if the output of this identity is an intermediate tensor
            # then remove the output tensor
            if opr.outputs[0].type == tg.TenType.INTERMEDIATE:
                input_tensor = opr.inputs[0]
                output_ops = opr.outputs[0].dependent_ops
                for output_op in output_ops:
                    idx = output_op.inputs.index(opr.outputs[0])
                    output_op.inputs[idx] = input_tensor
                input_tensor.dependent_ops = output_ops

                items_to_remove.extend([opr, opr.outputs[0]])
                removed_count += 1

            # otherwise if the input to this identity op is an intermediate
            # tensor then remove the input tensor.
            elif opr.inputs[0].type == tg.TenType.INTERMEDIATE:
                input_op = opr.inputs[0].creating_op
                input_op_output_idx = input_op.outputs.index(opr.inputs[0])
                output_tensor = opr.outputs[0]
                input_op.outputs[input_op_output_idx] = output_tensor
                output_tensor.creating_op = input_op

                items_to_remove.extend([opr, opr.inputs[0]])
                removed_count += 1

            # if neither the input or output tensors are intermediate then
            # do not remove this identity op

    graph.remove(items_to_remove)
    return removed_count


def fuse_bias_additions(graph):
    """
    Function which detects bias add operations in a graph and fuses them
    into the preceding layer operation
    :param graph:
    :return:
    """
    fusable_op_types = ['MatMul', 'Conv2D', 'DepthwiseConv2D']
    items_to_remove = []
    bias_additions_fused = 0

    for opr in graph.ops:
        if opr.type in fusable_op_types:

            # For this fusion to be possible the following op must be an Add
            # operation with a vector input the same size as the last dimension
            # of the output tensor. There must also only be one
            # operation which consumes the output tensor of this op.
            output_tensor = opr.outputs[0]
            if (len(output_tensor.dependent_ops) == 1 and
                    output_tensor.dependent_ops[0].type == 'Add'):
                last_dim = output_tensor.shape.last_dim()
                add_op = output_tensor.dependent_ops[0]
                if add_op.inputs[0] == output_tensor:
                    bias_tensor = add_op.inputs[1]
                else:
                    bias_tensor = add_op.inputs[0]

                if (bias_tensor.shape.len() == 1 and
                        bias_tensor.shape[0] == last_dim):

                    items_to_remove.extend([add_op, output_tensor])

                    # fusable bias add detected
                    opr.inputs.append(bias_tensor)
                    bias_tensor.dependent_ops = [opr]
                    opr.outputs = add_op.outputs
                    for output in opr.outputs:
                        output.creating_op = opr
                    bias_additions_fused += 1

    graph.remove(items_to_remove)
    return bias_additions_fused


def fuse_activations(graph):
    """
    function to detect and fuse layer activation functions
    :param graph:
    :return:
    """
    fusable_op_types = ['MatMul', 'Conv2D', 'DepthwiseConv2D']
    activation_fns = {'Relu': ActType.RELU,
                      'Relu6': ActType.RELU6,
                      'ReluN1To1': ActType.RELUN1TO1,
                      'TanH': ActType.TANH}

    items_to_remove = []
    activations_fused = 0

    for opr in graph.ops:
        if opr.type in activation_fns.keys():
            act_input_tensor = opr.inputs[0]
            if len(act_input_tensor.dependent_ops) == 1:
                input_op = act_input_tensor.creating_op
                if input_op.type in fusable_op_types:
                    output_tensors = opr.outputs

                    items_to_remove.extend([opr, act_input_tensor])
                    input_op.outputs = output_tensors
                    input_op.params['fused_activation_fn'] = \
                        activation_fns[opr.type]
                    for output in input_op.outputs:
                        output.creating_op = input_op
                    activations_fused += 1

    graph.remove(items_to_remove)
    return activations_fused


def remove_reshape_ops(graph):
    """
    Function to remove rehsape operations from the graph.
    Pointless reshapes which have the same input and output are simply
    removed. While reshapes which can be performed by re-addressing the input
    buffer are converted into a super/sub tensor pair.
    :param graph: 
    :return: 
    """
    items_to_remove = []
    pointless_reshapes_removed = 0

    for opr in graph.ops:
        if opr.type == "Reshape":
            old_shape = opr.inputs[0].shape.get_shape()
            new_shape = opr.outputs[0].shape.get_shape()
            if len(old_shape) == len(new_shape):
                shapes_match = True
                for idx, dim in enumerate(old_shape):
                    if dim != new_shape[idx]:
                        shapes_match = False
                if shapes_match:
                    if opr.outputs[0].type == tg.TenType.INTERMEDIATE:
                        # remove this op and the ouput tensor
                        items_to_remove.extend([opr, opr.outputs[0]])
                        input_tensor = opr.inputs[0]
                        output_ops = opr.outputs[0].dependent_ops
                        input_tensor.dependent_ops = output_ops
                        for output in output_ops:
                            idx = output.inputs.index(opr.outputs[0])
                            output.inputs[idx] = input_tensor
                        pointless_reshapes_removed += 1

                    elif opr.inputs[0].type == tg.TenType.INTERMEDIATE:
                        # remove this op and the input tensor
                        items_to_remove.extend([opr, opr.inputs[0]])
                        input_op = opr.inputs[0].creating_op
                        output_tensor = opr.outputs[0]
                        input_op.outputs = [output_tensor]
                        output_tensor.creating_op = input_op
                        pointless_reshapes_removed += 1

    graph.remove(items_to_remove)

    # TODO remove reshape ops with can be replaced by re-indexing

    return pointless_reshapes_removed


def graph_from_tf_sess(sess, outputs):
    """
    method to populate this grah from the given session and list of output
    tensors
    :param sess: TF interactive session which includes the source flow graph
    :param outputs: list of output tensors (either string of their
                    name or objects)
    :return: True on success, False on failure
    """
    new_graph = tg.Graph()

    # Convert and strings in the outputs list to their tensor objects
    output_tensors = []
    for out in outputs:
        if isinstance(out, tf.Tensor):
            output_tensors.append(out)
        else:
            try:
                output_tensors.append(
                    sess.graph.get_tensor_by_name(out))
            except KeyError:
                print("Error: No tensor named \"%s\" found in graph!" % out)
                for opr in sess.graph.get_operations():
                    for ten in opr.outputs:
                        print("Did you mean : %s" % ten.name)

    # add each output tensor and recurse to add all preceding operations and
    # tensors. Marking them as outputs.
    for output_tensor in output_tensors:
        new_tensor = add_tf_tensor(new_graph, output_tensor, sess)
        new_tensor.type = tg.TenType.OUTPUT

    mark_inputs(new_graph)
    mark_weights(new_graph, sess)

    new_graph.find_orphans(print_debug=True)
    print("op count %d" % len(new_graph.ops))

    remove_identity_ops(new_graph)
    fused_adds = fuse_bias_additions(new_graph)
    fused_acts = fuse_activations(new_graph)
    reshapes_removed = remove_reshape_ops(new_graph)
    print("Fused %d additions" % fused_adds)
    print("Fused %d activations" % fused_acts)
    print("Removed %d reshapes" % reshapes_removed)

    new_graph.find_orphans(print_debug=True)
    print("new op count %d" % len(new_graph.ops))

    return new_graph
