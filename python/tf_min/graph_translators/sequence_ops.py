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
from tf_min.graph_translators.graph_translator import GraphTranslator


class SequenceOps(GraphTranslator):
  """
  SequenceOps graph translator. Adds operation order used either a laze or
  greedy approach.
  """

  DEFAULT_PARAMS = {'Execution': 'Greedy'}
  TYPE = 'SequenceOps'
  DESCRIPTION = 'Sequence Ops graph translator, adds operation ' \
                'ordering using either a greedy or lazy approach.'

  def __call__(self, input_graph, inplace=False):
    """
    Sequence the provided graph either in place, or cloned and returned.
    :param input_graph:
    :return:
    """
    if inplace:
        output_graph = input_graph
    else:
        output_graph = input_graph.clone()

    # clear any previous sequence
    for opr in output_graph.ops:
      opr.sequence_index = None

    next_index = 0
    opr_added = True
    while opr_added:
      opr_added = False
      # find an operation which has not been sequenced but where all
      # input tensors are.
      for opr in self.output_graph.ops:
        if opr.sequence_index is None:
          inputs_ready = True
          for input in opr.inputs:
            if input.creating_op is not None and \
                    input.creating_op.sequence_index is None:
              inputs_ready = False
              break
          if inputs_ready:
            opr.sequence_index = next_index
            next_index += 1
            opr_added = True

    self.output_graph.update_sequence_references()

    self.summary = ("Sequenced %d operations using greedy execution." %
                    next_index)

    if inplace:
        return
    else:
        return output_graph


def sequence_ops(input_graph, params={}, inplace=False):
    """
    Convenience function so this translator can be used with a single
    function call, without the need to setup the functionoid.
    :param input_graph: Original graph object
    :param params: Dictionary of parameters
    :param inplace: Boolean, if true the input graph is modified,
                    if false the input graph is cloned and modified and the
                    new unique graph is returned.
    :return: None, or new graph depending on inplace
    """
    translator = SequenceOps(params)

    if inplace:
        translator(input_graph, inplace=True)
        return
    else:
        return translator(input_graph, inplace=False)