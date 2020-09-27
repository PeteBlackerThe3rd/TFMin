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

    This module defines a set of graph simplification graph translators, which
    can be used to reduce the number of operations in a grpah without
    changing the processing it performs. The following graph translators are
    defined:

    RemoveIdentityOps : Tensorflow adds these operations all over the place
                        since they don't actually do anything, this
                        translator can be used remove them.

    FuseBiasAdds : Convolution operations as defined in TFMin include fused
                   bias adds. So if a graph contains bias adds performed as
                   separate operations they can be merged. This is a more
                   memory and speed efficient way of performing this
                   common calculation.

    FuseActivations : Many common layer operators in TFMin include optional
                      fused activation functions. This translator detects
                      separate activation functions and fuses them into the
                      single layer operation.
"""
import xml.dom as xmldom
import copy
from enum import Enum
import numpy as np
import operator
import tf_min.graph as tg
import tf_min.types as types
from tf_min.activation_fns import ActType
from .graph_translator import GraphTranslator


class RemoveIdentityOps(GraphTranslator):
  """
  SequenceOps graph translator. Adds operation order used either a laze or
  greedy approach.
  """

  DEFAULT_PARAMS = {}
  TYPE = 'RemoveIdentityOps'
  DESCRIPTION = 'Remove Identity Ops graph translator, removes these ' \
                'operations from the graph.'

  def __call__(self, input_graph, inplace=False):
    """
    Remove Identity from the graph, or cloned and returned.
    :param input_graph:
    :return:
    """
    if inplace:
        output_graph = input_graph
    else:
        output_graph = input_graph.clone()

    removed_count = 0
    items_to_remove = []

    for opr in output_graph.ops:
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

    output_graph.remove(items_to_remove)
    self.summary = ("Removed %d identity ops from the graph." % removed_count)

    if inplace:
        return
    else:
        return output_graph


class FuseBiasAdds(GraphTranslator):
  """
  FuseBiasAdds graph translator. Fuse any separare bias add operations
  with their preceding convolution operations.
  """

  DEFAULT_PARAMS = {}
  TYPE = 'FuseBiasAdds'
  DESCRIPTION = 'Fuse Bias Adds graph translator, fuses any separare bias ' \
                'add operations with their preceding convolution operations.'

  def __call__(self, input_graph, inplace=False):
    """
    Fuse any separare bias add operations with their preceding convolution
    operations.
    :param input_graph:
    :return:
    """
    if inplace:
        output_graph = input_graph
    else:
        output_graph = input_graph.clone()

    fusable_op_types = ['MatMul', 'Conv2D', 'DepthwiseConv2D',
                        'TransposedConv2D']
    bias_add_op_types = ['Add', 'BiasAdd']
    items_to_remove = []
    bias_additions_fused = 0

    for opr in output_graph.ops:
      if opr.type in fusable_op_types:

        # For this fusion to be possible the following op must be an Add
        # or BiasAdd operation with a vector input the same size as the
        # last dimension of the output tensor. There must also only be one
        # operation which consumes the output tensor of this op.
        output_tensor = opr.outputs[0]
        if (len(output_tensor.dependent_ops) == 1 and
                output_tensor.dependent_ops[0].type in bias_add_op_types):
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

    output_graph.remove(items_to_remove)
    self.summary = ("Fused %d bias adds." % bias_additions_fused)

    if inplace:
        return
    else:
        return output_graph


class FuseActivations(GraphTranslator):
  """
  FuseBiasActivations graph translator. Fuse any separare activations
  operations with their preceding operations.
  """

  DEFAULT_PARAMS = {}
  TYPE = 'FuseActivations'
  DESCRIPTION = 'Fuse Activations graph translator, fuses any separare ' \
                'activations operations with their preceding operations.'

  def __call__(self, input_graph, inplace=False):
    """
    Fuse any separare activation operations with their preceding operations.
    :param input_graph:
    :return:
    """
    if inplace:
        output_graph = input_graph
    else:
        output_graph = input_graph.clone()

    fusable_op_types = ['MatMul', 'Conv2D', 'DepthwiseConv2D',
                        'TransposedConv2D']
    activation_fns = {'Relu': ActType.RELU,
                      'Relu6': ActType.RELU6,
                      'ReluN1To1': ActType.RELUN1TO1,
                      'TanH': ActType.TANH,
                      'LeakyRelu': ActType.LEAKY_RELU}

    items_to_remove = []
    activations_fused = 0

    for opr in output_graph.ops:
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
            for act_param in opr.params.keys():
              input_op.params['act_' + act_param] = opr.params[act_param]
            for output in input_op.outputs:
              output.creating_op = input_op
            activations_fused += 1

    output_graph.remove(items_to_remove)
    self.summary = ("Fused %d activation functions." % activations_fused)

    if inplace:
        return
    else:
        return output_graph


class RemoveReshapeOps(GraphTranslator):
  """
  RemoveReshapeOps graph translator. removes reshape operations either by
  removing them or replacing them with re-mapped sub-tensors
  """

  DEFAULT_PARAMS = {}
  TYPE = 'RemoveReshapeOps'
  DESCRIPTION = 'Remove Reshape Ops graph translator, removed reshapce ops ' \
                'either byremoving them or replacing them with ' \
                're-mapped sub-tensors.'

  def __call__(self, input_graph, inplace=False):
    """
    remove reshape operations either by
    removing them or replacing them with re-mapped sub-tensors
    :param input_graph:
    :return:
    """
    if inplace:
        output_graph = input_graph
    else:
        output_graph = input_graph.clone()

    items_to_remove = []
    pointless_reshapes_removed = 0
    reshapes_remapped = 0

    for opr in output_graph.ops:
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

    output_graph.remove(items_to_remove)

    # TODO find reshape operations which can be replaced by re-mappings

    self.summary = ("Removed %d pointless reshape operations\n"
                    "Removed %d reshape operations via re-mapping" %
                    (pointless_reshapes_removed,
                     reshapes_remapped))

    if inplace:
        return
    else:
        return output_graph
