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

    Base memory optimiser class. Provides a set of convenience functions
    useful for memory optimisation algorithms.
"""
import xml.dom as xmldom
import copy
from enum import Enum
import numpy as np
import operator
import tf_min.types as types
import tf_min.graph as tg
from ..graph import TenType, TenMetaType, Tensor, Operation, Graph
from tf_min.mem_opt.memory_region import MemoryRegion
from ..graph_translators.graph_translator import GraphTranslator


class MemoryOptimiser(GraphTranslator):
  """
  Base MemoryOptimiser object
  """

  DEFAULT_PARAMS = {'Order': 'Backwards',
                    'BatchSize': 1,
                    'UseOverlaps': True}
  TYPE = 'BaseMemoryOptimiser'
  DESCRIPTION = "---"

  def reset_memory_layout(self, graph):
    """
    Method which clears the buffer offsets all intermediate buffers of the
    given graph, and computes all buffer sizes.
    :param graph: tf_min.Graph, graph to reset
    :return: None
    """
    # reset offset and tensor buffer sizes
    for tensor in graph.tensors:
      if (tensor.type == tg.TenType.INTERMEDIATE and
              tensor.meta_type != TenMetaType.SUB):
        tensor.memory_offset = None
        tensor.buffer_size = \
            tensor.get_buffer_size(self.parameters['BatchSize'])
        tensor.creation_idx = None
        tensor.last_use_idx = None

  @staticmethod
  def scopes_overlap(tensor_a, tensor_b):
    """
    Method which returns true if the scopes of tensor_a and tensor_b overlap
    :param tensor_a: tf_min.Tensor
    :param tensor_b: tf_min.Tensor
    :return: True if their scopes overlap, False otherwise.
    """
    if tensor_a.creation_idx > tensor_b.last_use_idx:
      return False
    if tensor_b.creation_idx > tensor_a.last_use_idx:
      return False
    return True

  @staticmethod
  def populate_tensor_scopes(graph):
    """
    Method which updates the scope (first and last uses) of every tensor
    with an underlying buffer (not sub tensors) in the graph.
    :param graph: tf_min.Graph to update scopes of
    :return: None
    """
    for opr in graph.ops:
      idx = opr.sequence_index
      for output in opr.outputs:
        # if this output is a mapped sub-tensor then promote the
        # scope up-to the containing super-tensor
        if output.meta_type == tg.TenMetaType.SUB:
          output = output.super_tensor
        if output.creation_idx is None:
          output.creation_idx = idx
        else:
          output.creation_idx = min(output.creation_idx, idx)
      for input in opr.inputs:
        # if this input is a mapped sub-tensor then promote the
        # scope up-to the containing super-tensor
        if input.meta_type == tg.TenMetaType.SUB:
          input = input.super_tensor
        if input.last_use_idx is None:
          input.last_use_idx = idx
        else:
          input.last_use_idx = max(input.last_use_idx, idx)

  def get_heap_allocated_offset(self, graph, new_tensor):
    """
    Find the offset to place the tensor in the allocated block
    pattern using a heap allocation method. I.e. the first free space.
    :param graph:
    :param new_tensor: the tensor object to allocate
    :return: the offset to place this tensor at in bytes.
    """

    # create a list of all free regions of memory around the
    # tensors currently allocated which overlap with this tensors scope
    free_regions = [MemoryRegion(0, None)]
    for tensor in graph.tensors:
      if tensor.allocated() and self.scopes_overlap(tensor, new_tensor):
        new_free_regions = []
        tensor_region = MemoryRegion(tensor.memory_offset,
                                     (tensor.memory_offset +
                                      tensor.buffer_size))

        # if a safe overlap between this tensor and the tensor that's
        # being allocated has been defined then reduce the size of this
        # tensor region.
        if (tensor.safe_overlap_preceding_tensor == new_tensor and
                tensor.safe_overlap_bytes is not None):
          tensor_region.end -= tensor.safe_overlap_bytes

        for region in free_regions:
          new_free_regions.extend(region.get_carve_result(tensor_region))
        free_regions = new_free_regions

    # add the new tensor buffer to the first region it fits into
    new_tensor_region = MemoryRegion(0, new_tensor.buffer_size)
    for region in free_regions:
      if new_tensor_region.can_fit_inside(region):
        return region.start
    asset(False and "Error reached impossible point in heap "
                    "allocate algorithm!")
    return None

  @staticmethod
  def get_concrete_tensors(tensors):
      """
      Method return return a list of the concrete tensors from a list of
      tensors operation. this is either the single or super tensors or if it
      outputs a sub-tensor this is transformed to the corresponding
      super-tensor.
      :param tensors: list of tensors to filter
      :return: list of concrete tensors.
      """
      concrete_tensors = []
      for tensor in tensors:
          # if this output is a mapped sub-tensor then promote the
          # scope up-to the containing super-tensor
          if tensor.meta_type == tg.TenMetaType.SUB:
              tensor = tensor.super_tensor
          if tensor.type == tg.TenType.INTERMEDIATE:
            concrete_tensors.append(tensor)
      return concrete_tensors
