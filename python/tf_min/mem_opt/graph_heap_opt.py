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

    This module contains the basic Heap tensor pre-allocation allocation
    algorithm.
"""
import numpy as np
import tf_min.graph as tg
from tf_min.mem_opt.memory_region import MemoryRegion


class HeapAllocateGraph(tg.GraphTranslator):

    def __init__(self, graph, params={}):
        super().__init__(graph)
        self.label = "Heap Allocator"
        self.description = "Allocated intermediate tensors using a " \
                           "heap approach."
        self.summary = ""
        batch_size = 1
        order = "forwards"
        use_overlaps = False
        if 'batch_size' in params:
            batch_size = params['batch_size']
        if 'order' in params:
            order = params['order']
        if 'use_overlaps' in params:
            use_overlaps = params['use_overlaps']
        if self.output_graph.op_sequence is None:
            print("Error: Cannot heap allocate a graph which "
                  "hasn't been sequenced!")
        self.heap_allocate(order, batch_size)

    def heap_allocate(self, order, batch_size):

        print("-- starting to heap allocate tensors --")

        # test safe overlap tensors are part of this graph
        for tensor in self.output_graph.tensors:
            if tensor.safe_overlap_preceding_tensor is not None:
                preceding_idx = self.output_graph.get_tensor_idx(
                  tensor.safe_overlap_preceding_tensor
                )
                if preceding_idx is None:
                    print("Error preceding tensor link referrs to wrong graph!")

        # reset offset and tensor buffer sizes
        for tensor in self.output_graph.tensors:
            if tensor.type == tg.TenType.INTERMEDIATE:
                tensor.memory_offset = None
                # element_count = np.prod(tensor.get_tensor_shape(batch_size))
                tensor.buffer_size = tensor.get_buffer_size(batch_size)
                tensor.creation_idx = None
                tensor.last_use_idx = None

        # populate creation and final use of all tensors to allocate
        for opr in self.output_graph.ops:
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

        if order == 'backwards':
          self.output_graph.op_sequence.reverse()

        tensors_allocated = 0
        for opr in self.output_graph.op_sequence:
            for output in opr.outputs:
                if output.meta_type is tg.TenMetaType.SUB:
                    output = output.super_tensor
                if output.type == tg.TenType.INTERMEDIATE and \
                        not output.allocated():
                    self.heap_allocate_tensor(output)
                    tensors_allocated += 1

        if order == 'backwards':
          self.output_graph.op_sequence.reverse()

        self.summary = ("Allocated %d of %d intermediate tensor buffers,"
                        "taking %s bytes" %
                        (tensors_allocated,
                         self.output_graph.count_tensors_by_type(
                           tg.TenType.INTERMEDIATE,
                           meta_type=[tg.TenMetaType.SINGLE,
                                      tg.TenMetaType.SUPER]
                         ),
                         self.output_graph.get_peak_memory()))

        #self.output_graph.find_peak_ops_and_tensors(highlight=(50, 50, 100))

    @staticmethod
    def scopes_overlap(tensor_a, tensor_b):

        if tensor_a.creation_idx > tensor_b.last_use_idx:
            return False
        if tensor_b.creation_idx > tensor_a.last_use_idx:
            return False
        return True

    def heap_allocate_tensor(self, new_tensor):
        """
        Add a single buffer to the allocated block pattern using
        a heap allocation method. I.e. the first free space.
        This function takes advantage of diagonal memory optimisation if the
        safe overlap attributes of the tensor objects have been populated.
        :param new_tensor: the tensor object to allocate
        :return: None
        """

        # create a list of all free regions of memory around the
        # tensors currently allocated which overlap with this tensors scope
        free_regions = [MemoryRegion(0, None)]
        for tensor in self.output_graph.tensors:
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
              # print("Found a safe overlap during heap alloc")
              # print("Reducing region size from %d to %d" %
              #      (tensor_region.get_size(),
              #       tensor_region.get_size() - tensor.safe_overlap_bytes))
              tensor_region.end -= tensor.safe_overlap_bytes

            for region in free_regions:
              new_free_regions.extend(region.get_carve_result(tensor_region))
            free_regions = new_free_regions

        # add the new tensor buffer to the first region it fits into
        new_tensor_region = MemoryRegion(0, new_tensor.buffer_size)
        for region in free_regions:
          if new_tensor_region.can_fit_inside(region):
            new_tensor.memory_offset = region.start
            break

    def translate(self, verbose=False):
        if verbose:
          print(self.summary)
        return self.output_graph
