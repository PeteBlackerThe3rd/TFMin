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

    This module defines the graph translator base object from which all
    graph translators are descended.
"""
import xml.dom as xmldom
import copy
from enum import Enum
import numpy as np
import operator
import tf_min.types as types
from ..graph import Graph, Tensor, Operation, TenType, TenMetaType


class GraphTranslator:
  """
  GraphTranslator functionoid base class, defines the mechanism for
  loading settings and serializing and deserializing to and from XML
  """

  DEFAULT_PARAMS = {}
  TYPE = 'BaseTranslator'
  DESCRIPTION = 'Base Graph Translator object, should not be' \
                'instantiated or used!'

  def __init__(self, source={}):
    """
    Constructor which configures this graph translator, either deserializing
    from XML or adding settings from a dictionary.
    :param source: Optional, either a dictionary of settings or an
                   xml.dom.Node to read settings from.
    """
    # If a dictionary was given then copy it to settings and merge with
    # and undefined default settings
    self.parameters = copy.copy(self.DEFAULT_PARAMS)
    self.summary = ""

    # if a dictionary was passed to source then add this parameters
    if isinstance(source, dict):
      self.parameters.update(source)

    # if an XML node was passed to source then read attributes from the
    # XML element
    elif isinstance(source, xmldom.Node):
      self.parameters.update(dict(source.attributes.items()))

    # if any other type was passed to source then throw and assertion
    else:
      assert False, "Error cannot instantiate GraphTranslator from a " \
                    "\"%s\" type." % type(source)

  @classmethod
  def get_type(cls):
      return cls.TYPE

  @classmethod
  def call(cls, graph, params={}, inplace=True):
    """
    Convenience function so this translator can be used with a single
    function call, without the need to setup the functionoid.
    :param graph: Original graph object
    :param params: Dictionary of parameters
    :param inplace: Boolean, if true the input graph is modified,
                    if false the input graph is cloned and modified and the
                    new unique graph is returned.
    :return: None, or new graph depending on inplace
    """
    translator = cls(params)

    if inplace:
      translator(graph, inplace=True)
      return
    else:
      return translator(graph, inplace=False)

  def to_xml(self, doc):
    """
    Function to create an XML node representing this translator and
    its current settings.
    :param doc: an xml.dom.Document object used to create this node
    :return:
    """
    xml_node = doc.createNode(self.get_type())
    for setting, value in self.settings:
      xml_node.setAttribute(setting, value)
    return xml_node

  @staticmethod
  def clone(graph):
    cloned_graph = Graph()

    # copy tensors and ops
    for tensor in graph.tensors:
      cloned_graph.tensors.append(Tensor(tensor))
    for opr in graph.ops:
      cloned_graph.ops.append(Operation(opr))

    # update internal references within new tensors
    for idx, tensor in enumerate(cloned_graph.tensors):

      # update creating_op reference
      if tensor.creating_op is not None:
        creating_op_idx = (
          graph.get_opr_idx(graph.tensors[idx].creating_op)
        )
        tensor.creating_op = cloned_graph.ops[creating_op_idx]

      # update output references
      for i, output in enumerate(graph.tensors[idx].dependent_ops):
        output_idx = graph.get_opr_idx(output)
        tensor.dependent_ops[i] = cloned_graph.ops[output_idx]

      # update super tensor reference
      if tensor.super_tensor is not None:
        super_tensor_idx = (
          graph.get_tensor_idx(graph.tensors[idx].super_tensor)
        )
        tensor.super_tensor = cloned_graph.tensors[super_tensor_idx]

      # update sub_tensor references
      if tensor.meta_type == TenMetaType.SUPER:
        for i, sub_tensor in enumerate(graph.tensors[idx].sub_tensors):
          sub_tensor_idx = graph.get_tensor_idx(sub_tensor)
          tensor.sub_tensors[i] = cloned_graph.tensors[sub_tensor_idx]

      # update safe overlap preceding tensor reference
      if tensor.safe_overlap_preceding_tensor is not None:
        preceding_tensor_idx = (graph.get_tensor_idx(
          graph.tensors[idx].safe_overlap_preceding_tensor
        ))
        tensor.safe_overlap_preceding_tensor = \
          cloned_graph.tensors[preceding_tensor_idx]

      """if len(graph.tensors[idx].sub_tensors) > 0:
          print("Cloning a tensor with %d sub tensors of type [%s] "
                "(first sub tensor is [%s] idx [%s])" %
                (len(graph.tensors[idx].sub_tensors),
                 tensor.meta_type,
                 graph.tensors[idx].sub_tensors[0],
                 graph.get_tensor_idx(
                  graph.tensors[idx].sub_tensors[0])))"""

    # update internal references within new operations
    for idx, opr in enumerate(cloned_graph.ops):
      for i, input in enumerate(graph.ops[idx].inputs):
        input_idx = graph.get_tensor_idx(input)
        opr.inputs[i] = cloned_graph.tensors[input_idx]
      for i, output in enumerate(graph.ops[idx].outputs):
        output_idx = graph.get_tensor_idx(output)
        opr.outputs[i] = cloned_graph.tensors[output_idx]

    # update sequence references if they exist
    if graph.op_sequence is not None:
      cloned_graph.update_sequence_references()

    # cloned_graph = copy.deepcopy(graph)

    return cloned_graph

  def operate(self, input_graph):
    """
    Overloadable base class method which will perform the actual operation
    of this graph translator. This is the only method which much be
    overriden by child classes.
    :param input_graph: tf_min.Graph object to operate on
    :return: None
    """
    self.summary = "Base Graph Translator Object called. " \
                   "Returning unchanged graph."

  def __call__(self, input_graph, inplace=True):
    """
    Overload making this object callable, base class implementation
    takes an input Graph clones it and returns it unchanged.
    :param input_graph: tf_min.Graph object to operate on
    :param inplace: Boolean, if True the graph is modified inplace if
                    False then the graph is cloned modified and returned.
    :return:
    """
    if inplace:
        output_graph = input_graph
    else:
        output_graph = self.clone(input_graph)

    self.operate(output_graph)

    if inplace:
        return
    else:
        return output_graph
