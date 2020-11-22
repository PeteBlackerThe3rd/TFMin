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

    Heap Smart Order memory optimiser. This extends the basic Heap memory
    optimiser by placing the tensors in an order such that the tensor
    which will be placed at the lowest memory address is placed first.
"""
import xml.dom as xmldom
import copy
from enum import Enum
import numpy as np
import operator
import tf_min.types as types
import tf_min.graph as tg
from ..graph import TenType, TenMetaType, Tensor, Operation, Graph
from .memory_region import MemoryRegion
from .memory_optimiser import MemoryOptimiser
# from ..graph_translators.graph_translator import GraphTranslator


class HeapSmartOrder(MemoryOptimiser):
  """
  Heap Smart Order memory optimiser
  """

  DEFAULT_PARAMS = {'Order': 'Backwards',
                    'BatchSize': 1,
                    'UseOverlaps': True}
  TYPE = 'HeapSmartOrder'
  DESCRIPTION = "Allocated intermediate tensors using a " \
                "heap approach. With smart ordering."

  def operate(self, graph):
    """
    Sequence the provided graph either in place, or cloned and returned.
    :param graph: tf_min.Graph object to operate on
    :return: None
    """
    self.heap_allocate(graph)

  '''@staticmethod
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
      return concrete_tensors'''

  def heap_allocate(self, graph):

      # print("-- starting to heap allocate tensors --")

      # reset offset and tensor buffer sizes
      self.reset_memory_layout(graph)
      '''for tensor in graph.tensors:
          if (tensor.type == tg.TenType.INTERMEDIATE and
                  tensor.meta_type != TenMetaType.SUB):
              tensor.memory_offset = None
              # element_count = np.prod(tensor.get_tensor_shape(batch_size))
              tensor.buffer_size = \
                  tensor.get_buffer_size(self.parameters['BatchSize'])
              tensor.creation_idx = None
              tensor.last_use_idx = None'''

      # populate creation and final use of all tensors to allocate
      self.populate_tensor_scopes(graph)
      '''for opr in graph.ops:
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
                  input.last_use_idx = max(input.last_use_idx, idx)'''

      # initialise the set of tensors to start allocating
      tensors_to_allocate = []
      if self.parameters['Order'] == 'Forwards':
          first = graph.op_sequence[0]
          tensors_to_allocate.extend(self.get_concrete_tensors(first.outputs))
      elif self.parameters['Order'] == 'Backwards':
          last = graph.op_sequence[-1]
          tensors_to_allocate.extend(self.get_concrete_tensors(last.inputs))
      else:
          assert False, ("Error invalid order parameter \"%s\"" %
                         self.parameters['Order'])

      # While there are still tensors to allocate, allocate the one which
      # is placed such that it's offset+size is lowest
      tensors_allocated = 0
      while tensors_to_allocate:
          lowest_tensor = None
          lowest_tensor_offset = None

          # find the lowest tensor of the current tensors to allocate
          for tensor in tensors_to_allocate:
              offset = (self.get_heap_allocated_offset(graph, tensor) +
                        tensor.buffer_size)
              if (lowest_tensor_offset is None or
                      offset < lowest_tensor_offset):
                  lowest_tensor = tensor
                  lowest_tensor_offset = offset

          # allocate this tensor and remove it from the to allocate list
          lowest_tensor.memory_offset = (lowest_tensor_offset -
                                         lowest_tensor.buffer_size)
          tensors_to_allocate.remove(lowest_tensor)
          tensors_allocated += 1

          # add any un-allocated tensors with overlapping scopes with
          # the lowest tensor to the tensors to allocate list
          for tensor in graph.tensors:
              if (not tensor.allocated() and
                      tensor.type == tg.TenType.INTERMEDIATE and
                      tensor.meta_type != tg.TenMetaType.SUB and
                      tensor not in tensors_to_allocate and
                      tensor.scope_overlaps(lowest_tensor)):
                  tensors_to_allocate.append(tensor)

      self.summary = ("Allocated %d of %d intermediate tensor buffers,"
                      "taking %s bytes (%d KB)" %
                      (tensors_allocated,
                       graph.count_tensors_by_type(
                         TenType.INTERMEDIATE,
                         meta_type=[TenMetaType.SINGLE,
                                    TenMetaType.SUPER]
                       ),
                       graph.get_peak_memory(),
                       0))  # graph.get_peak_memory() / 1024))

  '''def heap_allocate_tensor(self, graph, new_tensor):
    """
    Add a single buffer to the allocated block pattern using
     a heap allocation method. I.e. the first free space.
    :param new_tensor: the tensor object to allocate
    :return: None
    """
    offset = self.get_heap_allocated_offset(graph, new_tensor)
    new_tensor.memory_offset = offset'''


'''def heap_smart_order(input_graph, params={}, inplace=False):
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
    translator = HeapSmartOrder(params)

    if inplace:
        translator(input_graph, inplace=True)
        return
    else:
        return translator(input_graph, inplace=False)'''