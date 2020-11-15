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

    Heap Reorder memory optimiser. This extends the basic Heap memory
    optimiser attempting a set of allocations in order except one tensor
    is moved from it's original place in the order to the end. In some cases
    this has the effect on 'un-hooking' large tensors which could be placed
    at lower offsets.

    This allocator is particularly effective on sequential models utilising
    diagonal memory overlaps.
"""
import copy
from ..graph import TenType, TenMetaType, Tensor, Operation, Graph
from .memory_optimiser import MemoryOptimiser


class HeapReorderMemOpt(MemoryOptimiser):
  """
  Heap reorder memory optimiser
  """

  DEFAULT_PARAMS = {'Order': 'Backwards',
                    'BatchSize': 1,
                    'UseOverlaps': True}
  TYPE = 'HeapReorder'
  DESCRIPTION = "Allocated intermediate tensors using a " \
                "simple heap approach. Each tensor is places at the " \
                "lowest offset in free memory in order."

  def operate(self, graph):
    """
    Sequence the provided graph either in place, or cloned and returned.
    :param graph: tf_min.Graph object to operate on
    :return: None
    """

    # get the default tensor allocation order
    tensor_order = self.get_tensor_order_to_allocate(graph)

    best_idx = None
    best_memory_size = None

    # for every tensor except the last reorder the tensors, perform heap
    # pre-allocation and record the tensor arena size
    for idx in range(len(tensor_order) - 1):
      new_order = copy.copy(tensor_order)
      new_order.append(new_order[idx])
      del new_order[idx]
      self.heap_allocate(graph, new_order)
      memory_size = graph.get_peak_memory()
      if best_memory_size is None or memory_size < best_memory_size:
        best_idx = idx
        best_memory_size = memory_size

    # re-allocate the graph using the optimal final tensor index
    new_order = copy.copy(tensor_order)
    new_order.append(new_order[best_idx])
    del new_order[best_idx]
    self.heap_allocate(graph, new_order)

    self.summary = ("Completed Heap reorder memory allocator\n"
                    "Allocated %d intermediate tensor buffers,"
                    "taking %s bytes (%d KB)\n"
                    "Tensor %d moved to final locations" %
                    (graph.count_tensors_by_type(
                       TenType.INTERMEDIATE,
                       meta_type=[TenMetaType.SINGLE,
                                  TenMetaType.SUPER]
                     ),
                     graph.get_peak_memory(),
                     graph.get_peak_memory() / 1024,
                     best_idx))

  def get_tensor_order_to_allocate(self, graph):
    """
    Method to return a list of tensors in the order they will be allocated
    :param graph: tf_min.Graph, to extract tensors and order from
    :return: List of tf_min.Tensor
    """
    allocation_op_order = copy.copy(graph.op_sequence)
    if self.parameters['Order'] == 'Backwards':
      allocation_op_order.reverse()

    # iterate through the allocation order adding tensors to the list
    tensor_order = []
    for opr in allocation_op_order:
      tensors_to_allocate = self.get_concrete_tensors(opr.outputs)
      for tensor in tensors_to_allocate:
        if tensor not in tensor_order:
          tensor_order.append(tensor)

    return tensor_order

  def heap_allocate(self, graph, tensor_order):

    # prepare graph for memory pre-allocation
    self.reset_memory_layout(graph)
    self.populate_tensor_scopes(graph)

    # iterate through the tensor order allocating all
    # generated tensors which are unallocated
    for tensor in tensor_order:
      offset = self.get_heap_allocated_offset(graph, tensor)
      tensor.memory_offset = offset
