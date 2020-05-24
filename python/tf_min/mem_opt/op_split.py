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

    This module defines the operation-splitting meta algorithm used to
    pre-allocate tensor buffers. This algorithm uses a extra block level 
    optimiser to perform the allocation of a modified graphs which are
    generated by this algorithm.
"""
import sys
import math as m
import copy as copy
import numpy as np
import tf_min.graph as tg
import tf_min.mem_opt.graph_heap_opt as heap_opt
import tf_min.mem_opt.graph_seq_allocator as seq_alloc
import tf_min.graph_2_svg as graph_2_svg
import tf_min.graph_mem_2_svg as mem_2_svg

sys.setrecursionlimit(10000)

# from sys import getsizeof, stderr
# from itertools import chain
# from collections import deque
# try:
#    from reprlib import repr
# except ImportError:
#    pass

# def total_size(o, handlers={}, verbose=False):
""" Returns the approximate memory footprint an object and all of its contents.

Automatically finds the contents of the following builtin containers and
their subclasses:  tuple, list, deque, dict, set and frozenset.
To search other containers, add handlers to iterate over their contents:

    handlers = {SomeContainerClass: iter,
                OtherContainerClass: OtherContainerClass.get_elements}

"""
"""dict_handler = lambda d: chain.from_iterable(d.items())
all_handlers = {tuple: iter,
                list: iter,
                deque: iter,
                dict: dict_handler,
                set: iter,
                frozenset: iter,
                }
all_handlers.update(handlers)  # user handlers take precedence
seen = set()  # track which object id's have already been seen
default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

def sizeof(o):
  if id(o) in seen:  # do not double count the same object
    return 0
  seen.add(id(o))
  s = getsizeof(o, default_size)

  if verbose:
    print(s, type(o), repr(o), file=stderr)

  for typ, handler in all_handlers.items():
    if isinstance(o, typ):
      s += sum(map(sizeof, handler(o)))
      break
  return s

return sizeof(o)"""


class OprSplit(tg.GraphTranslator):

    def __init__(self, graph, params={}):
        super().__init__(graph)
        self.label = "Operation Splitting Optimiser"
        self.description = "Optimiser which splits large tensor operations" \
                           "into smaller parallel streams to reduce the peak " \
                           "memory requirement of the model."
        self.summary = ""
        self.split_options = []

        self.supported_conv_ops = ['Conv2D', 'DepthwiseConv2D']
        self.supported_pool_ops = ['MaxPool', 'MinPool', 'AvgPool']

        # set default parameters and update with given parameters
        self.tensor_allocator = heap_opt.HeapAllocateGraph
        self.allocator_params = {}
        self.output_debug = False
        self.save_graphs = True
        self.attempt_sequential = True
        if 'allocator' in params:
            self.tensor_allocator = params['allocator']
        if 'allocator-params' in params:
            self.allocator_params = params['allocator-params']
        if 'output-debug' in params:
            self.output_debug = params['output-debug']
        if 'save-graphs' in params:
            self.save_graphs = params['save-graphs']
        if 'attempt-seq' in params:
            self.attempt_sequential = params['attempt-seq']
        self.output_graph.clear_highlights()

        print("Starting op split optimisation. output-debug is [%s]" %
              self.output_debug)

        self.find_split_options()

    def split_possible(self, op_1):

        # check there is an operation following this one
        if not op_1.outputs or not op_1.outputs[0].dependent_ops:
            return False
        op_2 = op_1.outputs[0].dependent_ops[0]

        # check op types
        kernel_ops = self.supported_conv_ops + self.supported_pool_ops
        if op_1.type not in kernel_ops or op_2.type not in kernel_ops:
            return False

        # check op_1 -> op_2 is a chain
        if len(op_1.outputs) != 1:
            return False
        if len(op_1.outputs[0].dependent_ops) != 1:
            return False

        # if any of the input or output tensors are SUB or SUPER tensors
        #  then ignore
        if op_1.inputs[0].meta_type is not tg.TenMetaType.SINGLE or \
                op_1.outputs[0].meta_type is not tg.TenMetaType.SINGLE or \
                op_2.outputs[0].meta_type is not tg.TenMetaType.SINGLE:
            return False

        # check that the output tensor of op_1 is larger than both it's input
        # and the output tensor of op_2
        input_buffer_size = op_1.inputs[0].get_buffer_size(1)
        int_buffer_size = op_1.outputs[0].get_buffer_size(1)
        output_buffer_size = op_2.outputs[0].get_buffer_size(1)
        if int_buffer_size <= input_buffer_size or \
                int_buffer_size <= output_buffer_size:
            return False

        # it is theoretically possible to split these operations and reduce the
        # peak memory, in practice you need to do the split and re-allocate
        # the graph to find out for sure.
        return True

    @staticmethod
    def calc_split_sizes(size, n_splits):
        """
        Function to compute the sizes of tensor required to split the input
        size into n_splits approximately equal sized chunks
        Throws error if any of the output sizes are zero.
        :param size:
        :param n_splits:
        :return: list of split tensor sizes
        """
        big_size = m.ceil(size / float(n_splits))
        small_size = m.floor(size / float(n_splits))
        big_count = size % n_splits
        small_count = n_splits - big_count
        return [small_size] * small_count + [big_size] * big_count

    # @staticmethod
    def get_receptive_file_size(self, opr, split_dim):
        """
        Method to compute the size of the receptive field of ops one and two in
        the split dimension
        :param opr: operation to get receptive field of
        :param split_dim: the dimension to get the receptive field size in
        :returns: the size of the receptive field (int)
        """

        # if 'dilation_height_factor' not in opr.params:
        #    print("Error cannot find 'dilation_height_factor' parameters:")
        #    print(opr.params)

        if split_dim == 1:
            if 'dilation_height_factor' in opr.params:
                op_dilation_factor = opr.params['dilation_height_factor']
            else:
                op_dilation_factor = 1
        else:
            if 'dilation_width_factor' in opr.params:
                op_dilation_factor = opr.params['dilation_width_factor']
            else:
                op_dilation_factor = 1

        # If this is a pooling op then use the filter size parameters
        if opr.type in self.supported_pool_ops:
            if split_dim == 1:
                filter_size = opr.params['filter_height']
            else:
                filter_size = opr.params['filter_width']
        else: # if it's a convolution of any sort use the filter input
            filter_size = opr.inputs[1].shape[split_dim]

        op_receptive_size = (filter_size - 1) * op_dilation_factor + 1
        return op_receptive_size

    @staticmethod
    def get_stride(opr, split_dim):
        if split_dim == 1:
            return opr.params['stride_height']
        return opr.params['stride_width']

    # @staticmethod
    def get_left_right_padding(self, opr, split_dim):
        """
        Method to get the left and right padding of the given operation in the
        given dimension
        :param opr: operation to get padding of
        :param split_dim: dimension to get padding of
        :return: a tuple of the left and right padding (int, int)
        """
        if split_dim == 1:
            stride = opr.params['stride_height']
        else:
            stride = opr.params['stride_width']

        in_size = opr.inputs[0].shape[split_dim]
        out_size = opr.outputs[0].shape[split_dim]
        effective_filter_size = self.get_receptive_file_size(opr, split_dim)
        padding = (out_size - 1) * stride + effective_filter_size - in_size

        return 0, max(0, padding)

    @staticmethod
    def increase_sequence_indices(graph, threshold, increment):
        for opr in graph.ops:
            if opr.sequence_index >= threshold:
              opr.sequence_index += increment

    @staticmethod
    def get_split_dim(shape):
        if shape[2] > shape[1]:
            return 2
        else:
            return 1

    # @staticmethod
    def split_ops(self, graph, opr, n_splits):

        input_tensor = opr.inputs[0]
        int_tensor = opr.outputs[0]
        opr_2 = opr.outputs[0].dependent_ops[0]
        output_tensor = opr_2.outputs[0]

        print("Splitting: output tensor name [%s] into %d branches" %
              (output_tensor.label,
               n_splits))

        # split the output of opr_2 on the largest of its spatial dimensions
        opr_2_output = opr_2.outputs[0]
        split_dim = OprSplit.get_split_dim(opr_2_output.shape)

        opr_2_split_dim = opr_2_output.shape[split_dim]
        #if opr_2_output.shape[2] > opr_2_output.shape[1]:
            # split_dim = 2
            #opr_2_split_dim = opr_2_output.shape[2]

        # compute the size of each intermediate tensor and the
        # sub-tensors of the input and output
        opr_rec_size = self.get_receptive_file_size(opr, split_dim)
        opr_padding = self.get_left_right_padding(opr, split_dim)
        opr_stride = OprSplit.get_stride(opr, split_dim)
        opr_2_rec_size = self.get_receptive_file_size(opr_2, split_dim)
        opr_2_padding = self.get_left_right_padding(opr_2, split_dim)
        opr_2_stride = OprSplit.get_stride(opr_2, split_dim)
        input_split_sizes = []
        int_split_sizes = []
        output_split_sizes = OprSplit.calc_split_sizes(opr_2_split_dim,
                                                       n_splits)
        for i, spit_size in enumerate(output_split_sizes):
            # compute the size of each intermediate tensor in the split dim
            int_size = (spit_size - 1) * opr_2_stride + opr_2_rec_size
            if i == 0:
                int_size -= opr_2_padding[0]
            if i == n_splits - 1:
                int_size -= opr_2_padding[1]
            int_split_sizes.append(int_size)
            # compute the size of the input sub-tensor in the split dim
            input_size = (int_size - 1) * opr_stride + opr_rec_size
            if i == 0:
                input_size -= opr_padding[0]
            if i == n_splits - 1:
                input_size -= opr_padding[1]
            input_split_sizes.append(input_size)

        # print("opr seq idx %d" % opr.sequence_index)
        # print("opr_2 seq idx %d" % opr_2.sequence_index)
        # print("output tensor has %d dependent_ops" % len(output_tensor.dependent_ops))

        # increase sequence indices of operations following these two ops
        # so that the new operations within the parallel chains can
        # be ordered correctly
        OprSplit.increase_sequence_indices(
          graph,
          output_tensor.dependent_ops[0].sequence_index,
          (n_splits - 1) * 2
        )
        # print("increasing sequence indices greater or equal to %d by %d" %
        #      (output_tensor.dependent_ops[0].sequence_index,
        #       (n_splits - 1) * 2))

        # create N split output tensors mapped onto opr_2_output
        output_sub_tensors = []
        opr_2_output.meta_type = tg.TenMetaType.SUPER
        opr_2_output.highlight_color = (0, 100, 0)
        opr_2_output.creating_op = None
        for i, new_size in enumerate(output_split_sizes):
            new_shape = opr_2_output.shape.copy()
            new_shape[split_dim] = new_size
            # sub_tensor = copy.deepcopy(opr_2_output)
            sub_tensor = tg.Tensor(opr_2_output)
            sub_tensor.highlight_color = (0, 100, 0)
            sub_tensor.shape = new_shape
            sub_tensor.label += "_%d" % i
            sub_tensor.meta_type = tg.TenMetaType.SUB
            sub_tensor.super_tensor = opr_2_output
            sub_tensor.dependent_ops = []
            sub_tensor.creating_op = opr_2
            opr_2_output.sub_tensors.append(sub_tensor)
            graph.tensors.append(sub_tensor)
            output_sub_tensors.append(sub_tensor)

        # create N split input tensors mapped onto input_tensor
        input_sub_tensors = []
        input_tensor.meta_type = tg.TenMetaType.SUPER
        input_tensor.highlight_color = (0, 100, 100)
        for i in range(n_splits):
            # sub_tensor = copy.deepcopy(input_tensor)
            sub_tensor = tg.Tensor(input_tensor)
            sub_tensor.meta_type = tg.TenMetaType.SUB
            sub_tensor.label += "_%d" % i
            # print("new input sub tensor [%s]" % sub_tensor.label)
            sub_tensor.super_tensor = input_tensor
            sub_tensor.creating_op = None

            sub_tensor.shape[split_dim] = input_split_sizes[i]

            input_tensor.sub_tensors.append(sub_tensor)
            graph.tensors.append(sub_tensor)
            input_sub_tensors.append(sub_tensor)

        # split opr, opr_2 and their intermediate tensor into N parallel chains
        opr_splits = []
        int_tensor_splits = []
        opr_2_splits = []
        input_tensor.dependent_ops = []
        opr.inputs[1].dependent_ops = []
        opr_weights = None
        if len(opr.inputs) > 1:
            opr_weights = opr.inputs[1]
            opr_weights.dependent_ops = []
        opr_biases = None
        if len(opr.inputs) > 2:
            opr_biases = opr.inputs[2]
            opr_biases.dependent_ops = []
        opr_2_weights = None
        if len(opr_2.inputs) > 1:
            opr_2_weights = opr_2.inputs[1]
            opr_2_weights.dependent_ops = []
        opr_2_biases = None
        if len(opr_2.inputs) > 2:
            opr_2_biases = opr_2.inputs[2]
            opr_2_biases.dependent_ops = []
        for i in range(n_splits):
            new_opr = tg.Operation(opr)
            new_int_tensor = tg.Tensor(int_tensor)
            new_opr_2 = tg.Operation(opr_2)

            # print("new sub opr has %d inputs" % len(new_opr.inputs))
            # print("new int tensor has %d deps" %
            # len(new_int_tensor.dependent_ops))

            new_opr.highlight_color = (100, 100, 0)
            new_int_tensor.highlight_color = (100, 100, 0)
            new_opr_2.highlight_color = (100, 100, 0)
            new_int_tensor.label += "_%d" % i

            new_int_tensor.shape[split_dim] = int_split_sizes[i]

            # define sequence of new operations
            new_opr.sequence_index = opr.sequence_index + (i*2)
            new_opr_2.sequence_index = opr.sequence_index + (i*2) + 1

            # connect new operations together
            input_sub_tensors[i].dependent_ops = [new_opr]
            new_opr.inputs[0] = input_sub_tensors[i]
            new_opr.outputs[0] = new_int_tensor
            new_int_tensor.creating_op = new_opr
            new_int_tensor.dependent_ops[0] = new_opr_2
            new_opr_2.inputs[0] = new_int_tensor
            new_opr_2.outputs[0] = output_sub_tensors[i]
            output_sub_tensors[i].creating_op = new_opr_2

            # connect weights and biases tensors to the new operations
            if opr_weights is not None:
                opr_weights.dependent_ops.append(new_opr)
            if opr_biases is not None:
                opr_biases.dependent_ops.append(new_opr)

            if opr_2_weights is not None:
                opr_2_weights.dependent_ops.append(new_opr_2)
            if opr_2_biases is not None:
                opr_2_biases.dependent_ops.append(new_opr_2)

            opr_splits.append(new_opr)
            int_tensor_splits.append(new_int_tensor)
            opr_2_splits.append(new_opr_2)
            graph.ops.extend([new_opr, new_opr_2])
            graph.tensors.append(new_int_tensor)

        graph.remove([opr, opr.outputs[0], opr_2])
        # print("Removed %d elements from the graph" % count)

        graph.update_sequence_references()

    # @staticmethod
    def get_recomputations(self, split_op, n_splits):
        """
        Compute the number of element recomputations produced by splitting
        the given op n_splits times
        :param split_op: the operation being split
        :param n_splits: the number of times to split it's calculation
        :return: int, the number of element recomputations
        """
        opr_2 = split_op.outputs[0].dependent_ops[0]
        opr_2_output = opr_2.outputs[0]
        split_dim = OprSplit.get_split_dim(opr_2_output.shape)
        int_tensor = split_op.outputs[0]
        slice_element_count = (np.prod(int_tensor.shape) /
                               int_tensor.shape[split_dim])
        receptive_field = self.get_receptive_file_size(opr_2, split_dim)
        stride = OprSplit.get_stride(opr_2, split_dim)

        recomputations = ((n_splits - 1) *
                          slice_element_count *
                          (receptive_field - stride))
        return recomputations

    def allocate_buffers(self, graph_to_alloc):
        """
        Function to allocate the intermediate buffers of this graph
        either using the sequential optimiser if possible, falling back
        to the provided allocation aglorithm if not.
        :return: a clone of the graph with allocated intermediate buffers
        """
        # graph_to_alloc.sequence_ops()

        if self.attempt_sequential:
            seq_opt = seq_alloc.SeqAllocateGraph(graph_to_alloc)
            if seq_opt.graph_allocated:
                return seq_opt.translate()

        heap_opt = self.tensor_allocator(graph_to_alloc,
                                         params=self.allocator_params)
        return heap_opt.translate()

    def find_split_options(self):
        """
        Method which finds the set of split options for this graph. updating the
        split_options list which contains the description of the split type
        memory saving, recomputation cost and optimised graph itself
        :return: None
        """
        # initial allocation of tensors to work from
        # heap_opt = self.tensor_allocator(self.output_graph,
        #                                 params=self.allocator_params)
        # self.output_graph = heap_opt.translate()
        self.output_graph = self.allocate_buffers(self.output_graph)
        # Add the non-optimised default option
        self.split_options.append(
          {'desc': 'original',
           'peak_mem': self.output_graph.get_peak_memory(),
           'mem_saved': 0,
           'recomputations': 0,
           'recheck': self.output_graph.elements_computed(),
           'graph': self.clone(self.output_graph),
           'op_splits': {}})
        test_graph_num = 0

        if self.save_graphs:
            graph_viz = graph_2_svg.SVGWriter(self.output_graph)
            graph_viz.write("test_graph_%02d.svg" % test_graph_num)
            mem_viz = mem_2_svg.SVGMemoryWriter(self.output_graph)
            mem_viz.write("test_mem_%02d.svg" % test_graph_num)

        print("This is test graph [%d]" % test_graph_num)
        test_graph_num += 1

        while True:
            optimisation_found = False
            current_option = self.split_options[-1]

            # find if there is a possible split which optimised the peak memory
            [peak_ops, _] = current_option['graph'].find_peak_ops_and_tensors()

            # first if there are any split ops in th current best optimisation
            # and those ops are in the set of peak ops identified above
            # then increase the split level and test if it reduces the peak
            # memory
            for split_op_idx in current_option['op_splits'].keys():
                # split_op = current_option['graph'].ops[split_op_idx]
                n_splits = current_option['op_splits'][split_op_idx]
                test_graph = self.clone(self.output_graph)

                test_split_op = test_graph.ops[split_op_idx]
                self.split_ops(test_graph, test_split_op, n_splits + 1)

                # optimise new graph and check if peak memory has reduced
                # allocator = self.tensor_allocator(test_graph,
                #                                  params=self.allocator_params)
                # test_graph = allocator.translate(verbose=True)
                test_graph = self.allocate_buffers(test_graph)

                if self.save_graphs:
                    graph_viz = graph_2_svg.SVGWriter(test_graph)
                    graph_viz.write("test_graph_%02d.svg" % test_graph_num)
                    mem_viz = mem_2_svg.SVGMemoryWriter(test_graph)
                    mem_viz.write("test_mem_%02d.svg" % test_graph_num)

                print("This is test graph [%d]" % test_graph_num)
                test_graph_num += 1

                if test_graph.get_peak_memory() < current_option['peak_mem']:
                    new_option = current_option.copy()
                    new_option['op_splits'] = new_option['op_splits'].copy()
                    new_option['desc'] = "opt"
                    new_option['graph'] = test_graph
                    new_option['peak_mem'] = test_graph.get_peak_memory()
                    new_option['mem_saved'] = (
                        self.split_options[0]['peak_mem'] -
                        new_option['peak_mem']
                    )
                    old_recomputations = self.get_recomputations(
                      test_split_op, n_splits)
                    new_recomputations = self.get_recomputations(
                      test_split_op, n_splits + 1)
                    new_option['recomputations'] += (new_recomputations -
                                                     old_recomputations)
                    new_option['recheck'] = test_graph.elements_computed()
                    new_option['op_splits'][split_op_idx] = n_splits + 1
                    self.split_options.append(new_option)
                    optimisation_found = True
                    print("### Increased existing split op. peak mem now %d" %
                          test_graph.get_peak_memory())
                    break

            # if no improved peak memory was found using existing split ops
            # then try splitting other ops in the set of peak memory ops
            if not optimisation_found:
                for peak_op in peak_ops:
                    if self.split_possible(peak_op):
                        print("Found possible split at operation %d [%s]" %
                              (current_option['graph'].get_opr_idx(peak_op),
                               peak_op.type))

                        test_graph = self.clone(self.output_graph)
                        # since the current graph may have been transformed
                        # from the original input graph by now, we need to
                        # lookup the peak ops index using it's unique label
                        split_op_idx = \
                            self.output_graph.get_opr_idx_from_label(
                              peak_op.label)

                        test_split_op = test_graph.ops[split_op_idx]
                        self.split_ops(test_graph, test_split_op, 2)
                        print("Split operation [%s] into 2 branches." %
                              peak_op.type)
                        # optimise new graph and check if peak
                        # memory has reduced
                        # allocator = self.tensor_allocator(
                        #  test_graph,
                        #  params=self.allocator_params
                        # )
                        # test_graph = allocator.translate(verbose=True)
                        test_graph = self.allocate_buffers(test_graph)
                        new_peak_mem = test_graph.get_peak_memory()

                        if self.save_graphs:
                            graph_viz = graph_2_svg.SVGWriter(test_graph)
                            graph_viz.write("test_graph_%02d.svg" %
                                            test_graph_num)
                            mem_viz = mem_2_svg.SVGMemoryWriter(test_graph)
                            mem_viz.write("test_mem_%02d.svg" % test_graph_num)

                        print("This is test graph [%d]" % test_graph_num)
                        test_graph_num += 1

                        if new_peak_mem < current_option[
                          'peak_mem']:
                          new_option = current_option.copy()
                          new_option['op_splits'] = \
                              new_option['op_splits'].copy()
                          new_option['desc'] = "opt"
                          new_option['graph'] = test_graph
                          new_option['peak_mem'] = new_peak_mem
                          new_option['mem_saved'] = (
                                  self.split_options[0]['peak_mem'] -
                                  new_option['peak_mem']
                          )
                          new_option['recomputations'] += \
                            self.get_recomputations(test_split_op, 2)
                          new_option['recheck'] = test_graph.elements_computed()
                          new_option['op_splits'][split_op_idx] = 2
                          self.split_options.append(new_option)
                          optimisation_found = True
                          print("@@@ Found a new split op and "
                                "reduced memory to %d" % new_peak_mem)
                          break

            if not optimisation_found:
                break

        print("Found %d possible optimisaed graphs with split operations" %
              (len(self.split_options)-1))
        print("op_level, desc 1, desc 2, peak mem, saving, saving %, re-computations, "
              "total re-computations, re-computations %")
        for i, option in enumerate(self.split_options):
            split_desc = ""
            for split_op_idx in option['op_splits'].keys():
                split_op = self.output_graph.ops[split_op_idx]
                split_desc += ("(split %s / %d)" %
                      (split_op.type,
                       option['op_splits'][split_op_idx]))
            print("%d, %s, %s, %d, %d, %f, %d, %d, %f" %
                  (i,
                   split_desc,
                   option['desc'],
                   option['peak_mem'],
                   option['mem_saved'],
                   (option['mem_saved'] /
                    float(self.split_options[0]['peak_mem'])),
                   option['recomputations'],
                   option['recheck'],
                   (option['recomputations'] /
                    self.split_options[0]['recheck'])
                   ))

        self.summary = "Op splitting algorithm Version 0.1"

        # set the output graph to the most optimisation option found.
        self.output_graph = self.split_options[-1]['graph']

    def translate(self, verbose=False):
        if verbose:
          print(self.summary)
        return self.output_graph