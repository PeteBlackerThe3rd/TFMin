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

    This module contains the semi-brute force peak-reordering tensor
    buffer pre-allocation algorithm.
"""
import math as m
import numpy as np
import tf_min.graph as tg
from tf_min.mem_opt.memory_region import MemoryRegion
import tf_min.graph_mem_2_svg as mem_2_svg
from ..graph_translators.graph_translator import GraphTranslator


class PeakReorderAllocateGraph(GraphTranslator):

    def __init__(self, graph, params={}):
        super().__init__(graph)
        self.label = "Peak Reorder Allocator"
        self.description = "Allocate intermediate tensors using a " \
                           "heap approach combined with peak re-ordering. " \
                           "Here the peak memory operations of the allocated " \
                           "graph are allocated in all possible orders to " \
                           "attempt to find a more optimal memory use pattern."
        self.summary = ""
        batch_size = 1
        order = "forwards"
        if 'batch_size' in params:
            batch_size = params['batch_size']
        if 'initial-order' in params:
            order = params['initial-order']
        self.peak_reorder_allocate(order, batch_size)

    def allocate_in_order(self, tensor_order):
        for tensor in tensor_order:
            tensor.memory_offset = None
        for tensor in tensor_order:
            self.heap_allocate_tensor(tensor)

    def get_permutation(self, input_list, permutation):
        """
        Method to make a copy of list in a different order
        the order is described by a number between 0 and (n-1)!
        :param input_list: list of items to re-arrange
        :param permutation: permutation index
        :return: list of items in requested permutation
        """
        un_placed = input_list.copy()
        placed = []
        # debug = ""
        n = len(input_list)
        for i in range(n-1):

            repeat_len = int(m.factorial(n) / m.factorial(n - i - 1))
            scale = int(m.factorial(n) / m.factorial(n - i))
            index_to_place = int((permutation % repeat_len) / scale)
            #debug += " [%d : %d (%d) u(%d)] " % (repeat_len, scale, index_to_place, len(un_placed))
            placed.append(un_placed[index_to_place])
            un_placed.remove(un_placed[index_to_place])
        placed.append(un_placed[0])
        #print("Perm [%3d] is %s - %s" % (permutation, placed, debug))

        return placed

    def peak_reorder_allocate(self, order, batch_size):

        #print("-- starting to heap allocate tensors --")

        # reset offset and tensor buffer sizes
        for tensor in self.output_graph.tensors:
            if tensor.type == tg.TenType.INTERMEDIATE:
                tensor.memory_offset = None
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

        # create initial tensor allocation order
        tensor_order = []
        for opr in self.output_graph.op_sequence:
            for output in opr.outputs:
                if output.meta_type is tg.TenMetaType.SUB:
                    output = output.super_tensor
                if output not in tensor_order and \
                        output.type == tg.TenType.INTERMEDIATE:
                    tensor_order.append(output)
        if order == 'backwards':
          tensor_order.reverse()

        # initial heap allocation
        self.allocate_in_order(tensor_order)

        debug_idx = 0
        debug_log = ""

        # repeatedly find the set of peak memory defining operations
        # and shuffle them into all possible orders, re-allocating each
        # permutation. Repeat for as long that this apporach reduces
        # the peak memory
        factorial_limit = 6  # don't attempt more than 6! = 720 permutations
        while True:
            [_, peak_tensors] = \
              self.output_graph.find_peak_ops_and_tensors(highlight=(0, 0, 100))

            peak_reorder_limit = 6
            if len(peak_tensors) > peak_reorder_limit:
                print("Warning %d peak tensors reducing set to the "
                      "limit of %d" %
                      (len(peak_tensors),
                       peak_reorder_limit))
                peak_tensors = peak_tensors[peak_reorder_limit:]

            debug_log += ("Re-ordering a set of %d peak tensors\n" %
                          len(peak_tensors))

            mem_writer = mem_2_svg.SVGMemoryWriter(self.output_graph)
            mem_writer.write("peak_reader_step_%02d.svg" % debug_idx)
            debug_idx += 1

            if len(peak_tensors) > factorial_limit:
                break

            # get a list of the indices of each peak tensor in the current
            # tensor order, so that different orders of peak tensors can
            # be easily merged into the original tensor order
            peak_tensor_idxs = []
            for peak_tensor in peak_tensors:
                peak_tensor_idxs.append(tensor_order.index(peak_tensor))

            # permutations are described by an integer between 0 and (n-1)!
            best_permutation = None
            best_peak_mem = self.output_graph.get_peak_memory()
            for perm in range(m.factorial(len(peak_tensors))):
                new_peak_order = self.get_permutation(peak_tensors, perm)
                for i, peak_tensor in enumerate(new_peak_order):
                    tensor_order[peak_tensor_idxs[i]] = peak_tensor
                self.allocate_in_order(tensor_order)
                new_peak_mem = self.output_graph.get_peak_memory()
                if new_peak_mem < best_peak_mem:
                    best_permutation = perm
                    best_peak_mem = new_peak_mem

            # if an more optimal order has been found then use it
            if best_permutation is not None:
                debug_log += ("Permutation found which reduces "
                              "peak memory to : %d" % best_peak_mem)
                best_peak_order = self.get_permutation(peak_tensors,
                                                       best_permutation)
                for i, peak_tensor in enumerate(best_peak_order):
                    tensor_order[peak_tensor_idxs[i]] = peak_tensor
                self.allocate_in_order(tensor_order)
            else:  # if no permutations reduced peak memory
                break

        mem_writer = mem_2_svg.SVGMemoryWriter(self.output_graph)
        mem_writer.write("peak_reader_step_%02d_final.svg" % debug_idx)

        self.summary = ("Allocated tensors using peak re-ordering:\n%s"
                        "\nFinal peak memory %d (%d KB)" %
                        (debug_log,
                         self.output_graph.get_peak_memory(),
                         self.output_graph.get_peak_memory() / 1024))

        self.output_graph.find_peak_ops_and_tensors(highlight=(50, 50, 100))

    @staticmethod
    def scopes_overlap(tensor_a, tensor_b):

        if tensor_a.creation_idx is None:
            print("Warning tensor a creation idx is None [%s] (%s)" %
                  (tensor_a.label,
                   tensor_a.meta_type))
        if tensor_a.last_use_idx is None:
            print("Warning tensor a last use idx is None [%s] (%s) <%s>" %
                  (tensor_a.label,
                   tensor_a.meta_type,
                   tensor_a.type))
        if tensor_b.creation_idx is None:
            print("Warning tensor b creation idx is None [%s] (%s)" %
                  (tensor_b.label,
                   tensor_b.meta_type))
        if tensor_b.last_use_idx is None:
            print("Warning tensor b last use idx is None [%s] (%s)" %
                  (tensor_b.label,
                   tensor_b.meta_type))

        if tensor_a.creation_idx > tensor_b.last_use_idx:
            return False
        if tensor_b.creation_idx > tensor_a.last_use_idx:
            return False
        return True

    def heap_allocate_tensor(self, new_tensor):
        """
        Add a single buffer to the allocated block pattern using
         a heap allocation method. I.e. the first free space.
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
