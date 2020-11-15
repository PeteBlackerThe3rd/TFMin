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

    Heap memory optimiser. Memory pre-allocation algorithm which iterates
    through the tensors to allocate either forwards or backwards and places
    each tensor buffer into the lowest area of free space.
"""
import copy
from ..graph import TenType, TenMetaType, Tensor, Operation, Graph
from .memory_optimiser import MemoryOptimiser


class HeapMemOpt(MemoryOptimiser):
  """
  Heap memory optimiser
  """

  DEFAULT_PARAMS = {'Order': 'Backwards',
                    'BatchSize': 1,
                    'UseOverlaps': True}
  TYPE = 'Heap'
  DESCRIPTION = "Allocated intermediate tensors using a " \
                "simple heap approach. Each tensor is places at the " \
                "lowest offset in free memory in order."

  def operate(self, graph):
    """
    Sequence the provided graph either in place, or cloned and returned.
    :param graph: tf_min.Graph object to operate on
    :return: None
    """
    self.heap_allocate(graph)

  def heap_allocate(self, graph):

      # prepare graph for memory pre-allocation
      self.reset_memory_layout(graph)
      self.populate_tensor_scopes(graph)

      # create list of tensors to allocate in order
      allocation_op_order = copy.copy(graph.op_sequence)
      if self.parameters['Order'] == 'Backwards':
        allocation_op_order.reverse()

      # iterate through the allocation order allocating all
      # generated tensors which are unallocated
      tensors_allocated = 0
      for opr in allocation_op_order:
        tensors_to_allocate = self.get_concrete_tensors(opr.outputs)
        for tensor in tensors_to_allocate:
          if not tensor.allocated():
            offset = self.get_heap_allocated_offset(graph, tensor)
            tensor.memory_offset = offset
            tensors_allocated += 1

      self.summary = ("Completed Heap memory allocator\n"
                      "Allocated %d of %d intermediate tensor buffers,"
                      "taking %s bytes (%d KB)" %
                      (tensors_allocated,
                       graph.count_tensors_by_type(
                         TenType.INTERMEDIATE,
                         meta_type=[TenMetaType.SINGLE,
                                    TenMetaType.SUPER]
                       ),
                       graph.get_peak_memory(),
                       graph.get_peak_memory() / 1024))
