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

    This module defined the graph object used to contain the mutable tensor
    graphs used by TFMin to store and manipulate machine learning models.
"""
import copy
from enum import Enum
import numpy as np
import operator
import tf_min.types as types


class TenType(Enum):
    INPUT = 1
    INTERMEDIATE = 2
    CONSTANT = 3
    OUTPUT = 4


class TenMetaType(Enum):
    SINGLE = 1
    SUB = 2
    SUPER = 3


class TensorShape:
    """
    Class which stores and manipulates the shape and data layout of a tensor
    """

    def __init__(self, initial_shape):

        self.shape = initial_shape
        self.rank = len(initial_shape)

        # set default dimension ordering
        # this list maps input semantic dimesions into a list of dimensions
        # in order from the most major (significant) to minor (least
        # significant)
        self.dim_order = list(range(len(initial_shape)))

        # set default dimension extra steps, this array defines the
        # additional offset step for each dimension in units of the step of
        # the next higher dimension (1 in the case of the highest dimensions)
        self.dim_extra_steps = [0] * len(initial_shape)

        # initial base offset, the offset from the start of the buffer to the
        # first element. Usually zero unless this is a split sub-tensor.
        self.base_offset = 0

    def copy(self):
        new_copy = TensorShape(self.shape.copy())
        new_copy.dim_order = self.dim_order.copy()
        new_copy.dim_extra_steps = self.dim_extra_steps
        new_copy.base_offset = self.base_offset
        return new_copy

    def get_shape(self, batch_size=None):
        shape = []
        for dim in self.shape:
            if dim == -1 and batch_size is not None:
                dim = batch_size
            shape.append(dim)
        return shape

    def last_dim(self):
        return self.shape[self.dim_order[-1]]

    def get_element_count(self, batch_size=1):
        shape = self.get_shape(batch_size)
        return np.prod(shape)

    def convert_semantic_to_significance(self, semantic):
        """
        Convert a list of dimensions or indices in semantic order into a list
        in signifance order from major to minor.
        :param semantic: List of dimension sizes, or indices
        :return: List of dimension sizes or indices in order from major
                 to minor
        """
        assert len(semantic) == len(self.dim_order), \
            "Error the number of semantic dimensions must match the number " \
            "of dimenions order indices"
        sig_order = [None] * len(semantic)
        for idx, order in enumerate(self.dim_order):
            sig_order[idx] = semantic[order]
        return sig_order

    def convert_significance_to_semantic(self, sig_order):
      """
      Convert a list of dimensions or indices in significance order into a list
      in semantic order.
      :param sig_order: List of dimension sizes, or indices
      :return: List of dimension sizes or indices in semantic order
      """
      assert len(sig_order) == len(self.dim_order), \
        "Error the number of significance dimensions must match the number " \
        "of dimenions order indices"
      sem_order = [None] * len(sig_order)
      for idx, order in enumerate(self.dim_order):
        sem_order[order] = sig_order[idx]
      return sem_order

    def get_layout_addressing_coeffs(self, batch_size=1):
        """
        Method to compute and return the index coefficients needed to index
        into this shape and layout of data.
        :param batch_size: optional batch size to replace unknown dimension
                           with.
        :return: tuple of (coeff_0, ..., coeff_n, base_offset)
        """
        # re-order dims and extra steps from semantic to major-minor
        sig_dims = self.convert_semantic_to_significance(self.shape)
        sig_extra_steps = self.convert_semantic_to_significance(
          self.dim_extra_steps
        )

        # print("get coeffs: sig_dims = %s" % sig_dims)

        # add batch to at most one -1 dimension size
        updated_count = 0
        for idx, dim in enumerate(sig_dims):
            if dim == -1:
                sig_dims[idx] = batch_size
                updated_count += 1

        assert updated_count <= 1, "Error cannot have more than one batch dim"

        # compute dimension coefficients from minor to major
        sig_coeffs = [None] * self.rank
        last_step = 1
        for idx, dim in reversed(list(enumerate(sig_dims))):
            new_step = (sig_extra_steps[idx] + 1) * last_step
            sig_coeffs[idx] = new_step
            last_step = new_step * dim

        # convert coefficients from significance order back to semantic order
        coeffs = self.convert_significance_to_semantic(sig_coeffs)
        return coeffs + [self.base_offset]

    def old_get_layout_addressing_coeffs(self, batch_size=1):
        # TODO currently doesn't compute dim_extra_steps
        coeffs = []
        for idx, order in enumerate(self.dim_order):
          coeff = 1
          for order in self.dim_order[idx+1:]:
            dim = self.shape[order]
            if dim == -1:
              dim = batch_size
            coeff *= dim
          coeffs.append(coeff)

        return coeffs + [self.base_offset]

    def test_offset(self, i0, i1, i2, i3, batch_size=1):
        dims_data = self.get_shape(batch_size=batch_size)
        offset = ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3
        return offset

    def test_addressing_coeffs(self):
        print("Starting addressing coeffs test.")
        batch_size = 1
        coeffs = self.get_layout_addressing_coeffs(batch_size=batch_size)
        dim_sizes = self.get_shape(batch_size=batch_size)
        element_count = np.prod(dim_sizes)

        print("Shape %s element count [%d]" % (dim_sizes, element_count))

        for i0 in range(dim_sizes[0]):
          for i1 in range(dim_sizes[1]):
            for i2 in range(dim_sizes[2]):
              for i3 in range(dim_sizes[3]):
                test_offset = self.test_offset(i0, i1, i2, i3, batch_size)
                coeff_offset = i0 * coeffs[0] + i1 * coeffs[1] + i2 * coeffs[2] + i3 * coeffs[3] + coeffs[4]

                if test_offset != coeff_offset:
                  print("Error at index [%d][%d][%d][%d]" % (i0, i1, i2, i3))
                  print("Coeff offset doesn't match TFLite offset fn!")
                  return

                if test_offset < 0 or test_offset >= element_count:
                  print("Error at index [%d][%d][%d][%d]" % (i0, i1, i2, i3))
                  print("TFLite offset out of range!")
                  return

                if coeff_offset < 0 or coeff_offset >= element_count:
                  print("Error at index [%d][%d][%d][%d]" % (i0, i1, i2, i3))
                  print("coeff offset out of range!")
                  return

        print("Coeff adressing test passed.")


    def __getitem__(self, key):
        assert (isinstance(key, int) or isinstance(key, slice)), \
          "Error TensorShape __getitem__ key must be an integer or slice"
        assert isinstance(key, slice) or (key >= 0 and key < len(self.shape)), \
          "Error TensorShape __getitem__ key [%d] out of range" % key
        return self.shape[key]

    def __setitem__(self, key, value):
        assert isinstance(key, int), "Error TensorShape __setitem__ key " \
                                     "must be an integer"
        assert key >= 0 and key < len(self.shape), "Error TensorShape " \
                                                   "__setitem__ key out of " \
                                                   "range"
        self.shape[key] = value

    def __eq__(self, other):
        assert isinstance(other, TensorShape), \
          "Error: Comparing a tf_min.TensorShape to another type which isn't " \
          "a tf_min.TensorShape."
        return self.shape == other.shape

    def __str__(self):
        return str(self.shape)

    def len(self):
        return len(self.shape)


class Tensor:

    def __init__(self, source_tensor=None):

        self.creating_op = None
        self.dependent_ops = []
        self.value = None
        self.memory_offset = None
        self.buffer_size = None
        self.data_ptr_str = None
        self.label = "not set!"
        self.d_type = None
        self.highlight_color = None
        self.creation_idx = None
        self.last_use_idx = None
        self.data_layout = None
        self.shape = None
        self.type = None

        # sub/super tensor properties
        self.meta_type = TenMetaType.SINGLE
        self.super_tensor = None
        self.sub_tensors = []

        # safe overlap properties
        self.safe_overlap_preceding_tensor = None
        self.safe_overlap_bytes = None

        if isinstance(source_tensor, Tensor):
            self.creating_op = source_tensor.creating_op
            self.dependent_ops = source_tensor.dependent_ops.copy()
            if isinstance(source_tensor.value, np.ndarray):
                self.value = source_tensor.value.copy()
            else:
                self.value = copy.copy(source_tensor.value)
            self.memory_offset = source_tensor.memory_offset
            self.buffer_size = source_tensor.buffer_size
            self.data_ptr_str = source_tensor.data_ptr_str
            self.highlight_color = source_tensor.highlight_color
            self.creation_idx = source_tensor.creation_idx
            self.last_use_idx = source_tensor.last_use_idx
            self.meta_type = source_tensor.meta_type
            self.super_tensor = source_tensor.super_tensor
            self.sub_tensors = source_tensor.sub_tensors.copy()
            self.label = source_tensor.label
            self.d_type = source_tensor.d_type
            self.shape = source_tensor.shape.copy()
            self.type = source_tensor.type
            self.safe_overlap_preceding_tensor = (
              source_tensor.safe_overlap_preceding_tensor
            )
            self.safe_overlap_bytes = source_tensor.safe_overlap_bytes

    def get_tensor_shape(self, batch_size=None):
        assert self.shape is not None, "Error Tensor.shape is None"
        return self.shape.get_shape(batch_size=batch_size)

    def allocated(self):
        return self.memory_offset is not None

    def get_buffer_size(self, batch_size=1):
      """
      Method to return the size of the buffer needed to hold this tensor.
      Returns None if this is a SUB tensor whose storage is mapped onto
      another tensors memory.
      :return:
      """
      assert isinstance(self.d_type, types.TenDType), \
          "Error cannot call get_buffer_size on a tensor without a valid dtype"
      assert self.meta_type is not TenMetaType.SUB, \
          "Error cannot call get buffer size on a sub-tensor"
      element_count = np.prod(self.get_tensor_shape(batch_size))
      buffer_size = (types.get_dtype_size(self.d_type) *
                     element_count)
      return buffer_size

    def scope_overlaps(self, tensor_b):

        if self.creation_idx > tensor_b.last_use_idx:
            return False
        if tensor_b.creation_idx > self.last_use_idx:
            return False
        return True


class Operation:

    def __init__(self, source_op=None):

        self.type = None
        self.highlight_color = None
        self.inputs = []
        self.outputs = []
        self.params = {}
        self.label = None
        self.sequence_index = None

        if isinstance(source_op, Operation):
            self.type = source_op.type
            self.label = source_op.label
            self.highlight_color = source_op.highlight_color
            self.inputs = source_op.inputs.copy()
            self.outputs = source_op.outputs.copy()
            self.params = source_op.params.copy()
            self.sequence_index = source_op.sequence_index

    def get_outputs_by_type(self, type):
        outputs = []
        for output in self.outputs:
            if output.type == type:
                outputs.append(output)
        return outputs


class Graph:

    def __init__(self):
        self.ops = []
        self.tensors = []
        self.op_sequence = None

    def print_operation_counts(self):
        """
        Prints the occurences of each type of operation used in this graph.
        :return: None
        """
        counts = {}
        for op in self.ops:
            if op.type in counts:
                counts[op.type] += 1
            else:
                counts[op.type] = 1
        counts = sorted(counts.items(),
                        key=operator.itemgetter(1),
                        reverse=True)

        print("Graph contains %d different types of operation:" % len(counts))
        for count in counts:
            print("[%25s] %d occurences" % (count[0], count[1]))

    def print_safe_overlaps(self):
        """

        :return:
        """
        print("Identified safe buffer overlaps within this graph")
        for tensor in self.tensors:
            if tensor.safe_overlap_preceding_tensor is not None:
                if tensor == tensor.safe_overlap_preceding_tensor:
                    print("Error: tensor cannot overlap with itself!")
                print("The end of tensor [%s] can overlap with tensor "
                      "[%s] by %d bytes." %
                      (tensor.label,
                       tensor.safe_overlap_preceding_tensor.label,
                       tensor.safe_overlap_bytes))

    def get_tensors_by_type(self, type):
        inputs = []
        for tensor in self.tensors:
            if tensor.type == type:
                inputs.append(tensor)
        return inputs

    def count_tensors_by_type(self, ten_types, meta_type=None):
        count = 0
        if not isinstance(ten_types, list):
          ten_types = [ten_types]
        if meta_type is not None and not isinstance(meta_type, list):
            meta_type = [meta_type]

        for tensor in self.tensors:
            type_in = False
            for ten_type in ten_types:
                if tensor.type is ten_type:
                    type_in = True
                    break
            if type_in:
                if meta_type is None:
                    count += 1
                else:
                    for meta_t in meta_type:
                        if tensor.meta_type is meta_t:
                            count += 1
                            break
        return count

    def get_inputs(self):
        return self.get_tensors_by_type(TenType.INPUT)

    def get_outputs(self):
        return self.get_tensors_by_type(TenType.OUTPUT)

    def get_constants(self):
        return self.get_tensors_by_type(TenType.CONSTANT)

    def get_tensor_idx(self, tensor):
        for i, g_tensor in enumerate(self.tensors):
            if g_tensor == tensor:
                return i
        return None

    def get_tensor_idx_from_label(self, label):
        for i, g_tensor in enumerate(self.tensors):
            if g_tensor.label == label:
                return i
        return None

    def get_opr_idx(self, opr):
        for i, g_opr in enumerate(self.ops):
            if g_opr == opr:
                return i
        return None

    def get_opr_idx_from_label(self, label):
        for i, g_opr in enumerate(self.ops):
            if g_opr.label == label:
                return i
        return None

    def get_peak_memory(self, silent=True):
        peak_memory = 0
        for tensor in self.tensors:
            if tensor.type != TenType.INTERMEDIATE:
                continue
            if tensor.meta_type == TenMetaType.SUB:
                continue
            if tensor.memory_offset is not None and \
                    tensor.buffer_size is not None:
                peak_memory = max(peak_memory,
                                  tensor.memory_offset + tensor.buffer_size)
            else:
                if not silent:
                  print("Error, cannot get peak_memory of a graph which "
                        "contains intermediate tensors that have not been "
                        "pre-allocated.")
                  print("Tensor [%s] not allocated" % tensor.label)
                return None
        return peak_memory

    def get_weight_stats(self):
        """
        Method to return the number of constant weights in this model and
        the number of bytes they take to store
        :return: Tuple of weights count, weights bytes
        """
        weights_total = 0
        weights_bytes_total = 0
        for weight in self.get_constants():
            weights_total += weight.shape.get_element_count()
            weights_bytes_total += weight.get_buffer_size(batch_size=1)
        return weights_total, weights_bytes_total

    def get_nth_operation_of_type(self, nth, op_type):
        """
        Method to return the nth operation of the given type in the sequence
        of this graph. Useful for manual manipulation of graphs.
        :param nth: Int, the instance of operation to return
        :param op_type: String, the type of operation to search for.
        :return: TFMin.Operation or None if no operation was found.
        """
        assert self.sequenced(), "Error: Cannot get nth operation of type on " \
                                 "a graph which isn't sequenced."

        n_found = 0
        for operation in self.op_sequence:
            if operation.type == op_type:
                if n_found == nth:
                    return operation
                else:
                    n_found += 1
        return None

    def sequenced(self):
        """
        Method to check if this graph has been sequenced correctly
        :Return: True if this graph has a valid sequence, false otherwise.
        """
        # TODO implement this
        return True

    def sequence_ops(self):

        # clear existing sequence markers
        for opr in self.ops:
            opr.sequence_index = None

        # sequence using eager execution from the input tensors
        next_index = 0
        opr_added = True
        while opr_added:
            opr_added = False
            # find an operation which has not been sequenced but where all
            # input tensors are.
            for opr in self.ops:
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

        self.update_sequence_references()

        print("Sequenced %d operations using eager execution." % next_index)

    def update_sequence_references(self):
        self.op_sequence = [None] * len(self.ops)
        for opr in self.ops:
            if isinstance(opr.sequence_index, int):
                if 0 <= opr.sequence_index < len(self.ops):
                    self.op_sequence[opr.sequence_index] = opr
                else:
                    print("Error: sequence index [%d] of op [%s] "
                          "is out of range!" %
                          (opr.sequence_index,
                           opr.type))
                    break
            else:
                print("Error: sequence index is not an integer!")
                break
        if None in self.op_sequence:
            print("Error: Incomplete operation sequence in graph!")
            print("Operations should be sequenced between 0 and %d" %
                  (len(self.ops) - 1))
            for seq_idx, opr in enumerate(self.op_sequence):
                if opr is None:
                    print("No operation with sequence index [%d]" % seq_idx)
            for i, opr in enumerate(self.ops):
                print("[%d] %s has seq_idx %s" %
                      (i,
                       opr.type,
                       opr.sequence_index))

    def find_peak_ops_and_tensors(self, highlight=None):
        """
        Finds the set of operations whose outputs define the peak memory
        requirement of this model.
        This method is DMO safe, because it takes into account potential
        tensor overlaps while detecting adacent tensors.
        :param highlight: color with which to highlight peak operations and
                          tensors with if given.
        :return: tuple containing the set of peak operations and list of
                 peak tensors.
        """
        peak_mem = self.get_peak_memory()

        # find set of tensors which are at the top end of peak memory
        peak_tensors = []
        for tensor in self.tensors:
            if tensor.allocated():
                if tensor.memory_offset + tensor.buffer_size == peak_mem:
                    peak_tensors.append(tensor)

        # add tensors who's allocated buffers are left-adjacent
        # to these peak_ops
        while True:
          tensor_added = False
          for tensor in self.tensors:
            if tensor.allocated() and tensor not in peak_tensors:
              for peak_tensor in peak_tensors:
                if not tensor.scope_overlaps(peak_tensor):
                  continue
                tensor_end_offset = (tensor.memory_offset +
                                     tensor.buffer_size)
                if tensor.safe_overlap_preceding_tensor == peak_tensor:
                    tensor_end_offset -= tensor.safe_overlap_bytes
                if tensor_end_offset == peak_tensor.memory_offset:
                  peak_tensors.append(tensor)
                  tensor_added = True
                  break
          if not tensor_added:
            break

        # create the set of operations which generate these peak tensors
        peak_ops = set([])
        for tensor in peak_tensors:
            if tensor.creating_op is not None:
                peak_ops.add(tensor.creating_op)
            if tensor.meta_type is TenMetaType.SUPER:
                for sub_tensor in tensor.sub_tensors:
                    if sub_tensor.creating_op is not None:
                        peak_ops.add(sub_tensor.creating_op)

        # highlight tensors and operations if requested
        if highlight is not None:
            for opr in peak_ops:
                opr.highlight_color = highlight
            for tensor in peak_tensors:
                tensor.highlight_color = highlight

        return peak_ops, peak_tensors

    def clear_highlights(self):
        """
        Removes highlights from all tensors and operations
        :return: None
        """
        for tensor in self.tensors:
            tensor.highlight_color = None
        for opr in self.ops:
            opr.highlight_color = None

    def remove(self, items_to_remove):
        """
        Function to remove a mixed list of tensors and operations from the graph
        :param items_to_remove: list of tensor and operation objects
        :return: number of items successfully removed
        """
        items_removed = 0
        for item in items_to_remove:
            if isinstance(item, Tensor):
                tensor = item
                tensor_idx = self.get_tensor_idx(tensor)
                if tensor_idx is not None:
                    self.tensors.remove(self.tensors[tensor_idx])
                    items_removed += 1
            elif isinstance(item, Operation):
                opr = item
                opr_idx = self.get_opr_idx(opr)
                if opr_idx is not None:
                    self.ops.remove(self.ops[opr_idx])
                    items_removed += 1
        return items_removed

    def elements_computed(self):
        """
        This computes the sum of all elements in the outputs of each operation
        and output tensors
        :return: total number of elements computed by graph
        """
        element_count = 0
        for opr in self.ops:
            for output in opr.outputs:
               element_count += output.shape.get_element_count()
        return element_count

    @staticmethod
    def mark_not_orphaned(item):
        """

        :param item:
        :return:
        """
        item.orphaned = False

        if isinstance(item, Tensor):
          if item.creating_op is not None:
            Graph.mark_not_orphaned(item.creating_op)
        elif isinstance(item, Operation):
          for input in item.inputs:
            Graph.mark_not_orphaned(input)


    def find_orphans(self, print_debug=False):
        """
        Method which looks for unconnected tensor and operations and returns
        a tuple of containing lists of each type.
        :param print: Boolean, print useful debugging information
        :return: Tuple, ([List of tensor orphans], [List of op orphans])
        """

        # mark all tensors and ops as orphans
        for tensor in self.tensors:
          tensor.orphaned = True
        for opr in self.ops:
          opr.orphaned = True

        # recurssively mark all tensors and ops connected to an output as
        # not orphans
        for output in self.get_outputs():
          self.mark_not_orphaned(output)

        # collect orphaned tensors and operations
        orphaned_tensors = []
        orphaned_ops = []
        for tensor in self.tensors:
          if tensor.orphaned:
            orphaned_tensors.append(tensor)
        for opr in self.ops:
          if opr.orphaned:
            orphaned_ops.append(opr)

        # remove orphaned attrbutes
        for tensor in self.tensors:
          del tensor.orphaned
        for opr in self.ops:
          del opr.orphaned

        if print_debug:
          print("Found [%d] orphans [%d] tensors and [%d] operations" %
                ((len(orphaned_ops) + len(orphaned_tensors)),
                 len(orphaned_tensors),
                 len(orphaned_ops)))

        return (orphaned_tensors, orphaned_ops)