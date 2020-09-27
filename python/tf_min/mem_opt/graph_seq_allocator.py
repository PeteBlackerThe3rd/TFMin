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

    This module defines a custom tensor buffer pre-allocation algorithm
    which can work with operation-split graphs without doing anything
    silly with the parallel split tensors.
"""
import numpy as np
import tf_min.graph as tg
from tf_min.mem_opt.memory_region import MemoryRegion
from ..graph_translators.graph_translator import GraphTranslator


class SeqAllocateGraph(GraphTranslator):

    def __init__(self, graph, params={}):
        super().__init__(graph)
        self.label = "Sequential Graph Allocator"
        self.description = "Allocated intermediate tensors using a " \
                           "an alternating approach unique to sequential and " \
                           "sequentual & split graphs only."
        self.summary = ""
        self.graph_allocated = False
        batch_size = 1
        # order = "forwards"
        if 'batch_size' in params:
            batch_size = params['batch_size']
        # if 'order' in params:
        #    order = params['order']
        if self.output_graph.op_sequence is None:
            print("Error: Cannot heap allocate a graph which "
                  "hasn't been sequenced!")
        if self.graph_is_sequential(self.output_graph):
            self.allocate(batch_size)
            self.graph_allocated = True
        elif 'debug' in params:
            print("Error: graph is not sequential cannot use SeqAllocateGraph"
                  "to allocated tensor buffers.")

    @staticmethod
    def graph_is_sequential(graph):

        # There must be one input and one output
        if (len(graph.get_inputs()) != 1 or len(graph.get_outputs()) != 1 or
                len(graph.get_inputs()[0].dependent_ops) != 1):
            return False
        input = graph.get_inputs()[0]

        # test if the graph is purely sequential
        purely_sequential = True
        cur_op = input.dependent_ops[0]
        while True:
            int_outputs = cur_op.get_outputs_by_type(tg.TenType.INTERMEDIATE)
            if len(int_outputs) != 1:
                if cur_op.outputs[0].type == tg.TenType.OUTPUT:
                    break
                else:
                    purely_sequential = False
                    break
            if len(int_outputs[0].dependent_ops) != 1:
                purely_sequential = False
                break
            cur_op = int_outputs[0].dependent_ops[0]

        if purely_sequential:
            print("Purely sequential graph found.")
            return True
        else:
            # continue testing for a sequential plus split graph
            sequential = True
            cur_op = input.dependent_ops[0]
            while True:
              int_outputs = cur_op.get_outputs_by_type(tg.TenType.INTERMEDIATE)

              # if this could be the start of a split section
              if (len(int_outputs) == 1 and
                      int_outputs[0].meta_type == tg.TenMetaType.SUPER):

                # get the final output super tensor if it exists
                first_int_op = int_outputs[0].sub_tensors[0].dependent_ops[0]
                if len(first_int_op.outputs) != 1:
                    sequential = False
                    print("#1")
                    break
                if len(first_int_op.outputs[0].dependent_ops) != 1:
                    sequential = False
                    print("#1.1")
                    break
                first_int_op2 = first_int_op.outputs[0].dependent_ops[0]
                if len(first_int_op2.outputs) != 1:
                    sequential = False
                    print("#1.2")
                    break
                if first_int_op2.outputs[0].meta_type != tg.TenMetaType.SUB:
                    sequential = False
                    print("#2")
                    break
                output_super_tensor = first_int_op2.outputs[0].super_tensor
                split_path_count = len(int_outputs[0].sub_tensors)

                # verify that all split paths are correct
                for path in range(split_path_count):
                    int_tensor = int_outputs[0].sub_tensors[path]
                    if len(int_tensor.dependent_ops) != 1:
                        sequential = False
                        print("#3")
                        break
                    int_op = int_tensor.dependent_ops[0]
                    if len(int_op.outputs) != 1:
                        sequential = False
                        print("#4")
                        break
                    if len(int_op.outputs[0].dependent_ops) != 1:
                        sequential = False
                        print("#4.1")
                        break
                    int_op2 = int_op.outputs[0].dependent_ops[0]
                    if len(first_int_op2.outputs) != 1:
                        sequential = False
                        print("#4.2")
                        break
                    if int_op2.outputs[0].meta_type != tg.TenMetaType.SUB:
                        sequential = False
                        print("#5")
                        break
                    if int_op2.outputs[0].super_tensor != output_super_tensor:
                        sequential = False
                        print("#6")
                        break

                # if this point is reached then it's a valid split section
                if output_super_tensor.type == tg.TenType.OUTPUT:
                    break
                if len(output_super_tensor.dependent_ops) != 1:
                    sequential = False
                    break
                cur_op = output_super_tensor.dependent_ops[0]

              else:  # if it's not the start of a split section
                if len(int_outputs) != 1:
                  if cur_op.outputs[0].type == tg.TenType.OUTPUT:
                    break
                  else:
                    purely_sequential = False
                    break
                if len(int_outputs[0].dependent_ops) != 1:
                  purely_sequential = False
                  break
                cur_op = int_outputs[0].dependent_ops[0]

            if sequential:
                print("Sequential graph with split sections found.")
                return True

        return False

    def allocate(self, batch_size):

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

        # Find the peak memory by finding the adjacent pair of buffers with
        # the largest combined size
        """peak_memory = 0
        for tensor in self.output_graph.tensors:
            if tensor.type == tg.TenType.INTERMEDIATE:
                following_tensor = tensor.dependent_ops[0].outputs[0]
                if following_tensor.type == tg.TenType.INTERMEDIATE:
                    required_memory = tensor.buffer_size + \
                                      following_tensor.buffer_size
                    if required_memory > peak_memory:
                        peak_memory = required_memory"""

        buffer_num = 0
        previous_tensor = None
        cur_tensor = \
            self.output_graph.get_inputs()[0].dependent_ops[0].outputs[0]
        while cur_tensor.type != tg.TenType.OUTPUT:

            # if this is the start of a split section allocate it specially
            if cur_tensor.meta_type == tg.TenMetaType.SUPER:
                input_tensor = cur_tensor
                path_count = len(input_tensor.sub_tensors)
                int_tensors = []
                for path in range(path_count):
                    int_tensor = \
                      input_tensor.sub_tensors[path].dependent_ops[0].outputs[0]
                    int_tensors.append(int_tensor)
                output_tensor = \
                  int_tensor.dependent_ops[0].outputs[0].super_tensor

                # int_tensors_offset = None
                if buffer_num % 2 == 0:
                    input_tensor.memory_offset = 0
                    int_tensors_offset = input_tensor.buffer_size
                else:
                    output_tensor.memory_offset = 0
                    int_tensors_offset = output_tensor.buffer_size

                outer_offset = 0
                for path in range(path_count):
                    int_tensors[path].memory_offset = int_tensors_offset
                    outer = int_tensors_offset + int_tensors[path].buffer_size
                    if outer > outer_offset:
                        outer_offset = outer
                if previous_tensor is not None:
                    outer_offset = max(outer_offset,
                                       previous_tensor.buffer_size)

                if buffer_num % 2 == 0:
                    output_tensor.memory_offset = outer_offset
                else:
                    input_tensor.memory_offset = outer_offset

                buffer_num += 2  # not really needed now but maybe in future?
                previous_tensor = output_tensor
                cur_tensor = output_tensor.dependent_ops[0].outputs[0]

            # if this is a regular tensor allocate it normally
            else:
                next_tensor = cur_tensor.dependent_ops[0].outputs[0]

                if next_tensor is None:
                    print("Warning next_tesnor is None!")

                if next_tensor.buffer_size is None:
                    print("Whats going on? type [%s] metatype [%s]" % (next_tensor.type, next_tensor.meta_type))

                if (previous_tensor is not None and
                        previous_tensor.buffer_size is None):
                    print("Previous tensor buffer size is None!!")

                if buffer_num % 2 == 0:
                    cur_tensor.memory_offset = 0
                else:
                    offset = 0
                    if previous_tensor is not None:
                        offset = previous_tensor.buffer_size
                    if next_tensor.type == tg.TenType.INTERMEDIATE:
                        offset = max(offset, next_tensor.buffer_size)

                    cur_tensor.memory_offset = offset

                buffer_num += 1
                previous_tensor = cur_tensor
                cur_tensor = next_tensor

        self.graph_allocated = True

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
            for region in free_regions:
              new_free_regions += region.get_carve_result(tensor_region)
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
