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


class GraphTranslator2:
    """
    GraphTranslator functionoid base class, defines the mechanism for
    loading settings and serializing and deserializing to and from XML
    """

    def __init__(self, source, defaults=None):
        """
        Constructor which configures this translator, either deserializing
        from XML or adding settings from a dictionary.
        :param source:
        """
        # If a dictionary was given then copy it to settings and merge with
        # and undefined default settings
        self.settings = {}
        self.translator_type = 'BaseTranslator'
        self.description = 'Base Graph Translator object, should not be' \
                           'instantiated or used!'
        if isinstance(source, dictionary):
            self.settings = source
            for key, value in defaults.items():
                if key not in self.settings:
                    self.settings[key] = value
        # If an XML node was passed then load settings from its attributes
        elif isinstance(source, ET.Element):
            for child in source:
                self.settings[child.tag] = child.attrib
        else:
            # unsupported source type, fail assertion
            assert False, "Unsupported source type of [%s] for " \
                          "GraphTranslator.__init__" % type(source)

    def to_xml(self):
        """
        Function to create an XML node representing this translator and
        its current settings.
        :return:
        """
        xml_node = ET.Element(self.translator_type)
        for setting, value in self.settings:
            xml_node.set(setting, value)
        return xml_node

    def __call__(self, input_graph):
        """
        Overload making this object callable, base class implementation
        takes an input Graph clones it and returns it unchanged.
        :param input_graph:
        :return:
        """
        output_graph = input_graph.clone()
        self.summary = "Base Graph Translator Object called. " \
                       "Returning unchanged graph."
        return output_graph


class RemoveIdentityOps2(GraphTranslator2):

    def __init__(self, source):
        super().__init__(source)
        self.translator_type = 'RemoveIdentityOps'
        self.description = "Translator which removes all pointless identity " \
                           "operations from a graph, merging surrounding " \
                           "tensors"

    def __call__(self, input):

      # clone input graph
      output_graph = input.clone()

      # mark ops and tensors to remove and update arcs to join gap
      ops_to_remove = []
      tensors_to_remove = []
      for opr in output_graph.ops:
        if opr.type == "Identity":
          input_tensor = opr.inputs[0]
          output_tensor = opr.outputs[0]
          input_tensor.dependent_ops = output_tensor.dependent_ops
          for dep_op in input_tensor.dependent_ops:
            for i, input in enumerate(dep_op.inputs):
              if input == output_tensor:
                dep_op.inputs[i] = input_tensor

          ops_to_remove.append(opr)
          tensors_to_remove.append(output_tensor)

      output_graph.remove(ops_to_remove + tensors_to_remove)

      self.summary = ("Removed %d identity operations and %d "
                      "associated tenors" % (len(ops_to_remove),
                                             len(tensors_to_remove)))
      return output_graph


class GraphTranslator:

    def __init__(self, graph):

        self.input_graph = graph
        self.label = "Base Graph Translator"
        self.description = "Base graph translator class, simply creates an" \
                           "returns a identical graph copy"
        self.output_graph = self.clone(graph)
        self.summary = "Copy"

    def translate(self):
        return self.output_graph

    @staticmethod
    def clone(graph):
        cloned_graph = Graph()

        # copy tensors and ops
        for tensor in graph.tensors:
            cloned_graph.tensors.append(Tensor(tensor))
        for opr in graph.ops:
            cloned_graph.ops.append(Operation(opr))

        # update internal references within new tensors
        for idx, tensor in enumerate(cloned_graph.tensors):

            # update creating_op reference
            if tensor.creating_op is not None:
                creating_op_idx = (
                  graph.get_opr_idx(graph.tensors[idx].creating_op)
                )
                tensor.creating_op = cloned_graph.ops[creating_op_idx]

            # update output references
            for i, output in enumerate(graph.tensors[idx].dependent_ops):
                output_idx = graph.get_opr_idx(output)
                tensor.dependent_ops[i] = cloned_graph.ops[output_idx]

            # update super tensor reference
            if tensor.super_tensor is not None:
                super_tensor_idx = (
                  graph.get_tensor_idx(graph.tensors[idx].super_tensor)
                )
                tensor.super_tensor = cloned_graph.tensors[super_tensor_idx]

            # update sub_tensor references
            if tensor.meta_type == TenMetaType.SUPER:
                for i, sub_tensor in enumerate(graph.tensors[idx].sub_tensors):
                    sub_tensor_idx = graph.get_tensor_idx(sub_tensor)
                    tensor.sub_tensors[i] = cloned_graph.tensors[sub_tensor_idx]

            # update safe overlap preceding tensor reference
            if tensor.safe_overlap_preceding_tensor is not None:
                preceding_tensor_idx = (graph.get_tensor_idx(
                  graph.tensors[idx].safe_overlap_preceding_tensor
                ))
                tensor.safe_overlap_preceding_tensor = \
                    cloned_graph.tensors[preceding_tensor_idx]

            """if len(graph.tensors[idx].sub_tensors) > 0:
                print("Cloning a tensor with %d sub tensors of type [%s] "
                      "(first sub tensor is [%s] idx [%s])" %
                      (len(graph.tensors[idx].sub_tensors),
                       tensor.meta_type,
                       graph.tensors[idx].sub_tensors[0],
                       graph.get_tensor_idx(
                        graph.tensors[idx].sub_tensors[0])))"""

        # update internal references within new operations
        for idx, opr in enumerate(cloned_graph.ops):
            for i, input in enumerate(graph.ops[idx].inputs):
                input_idx = graph.get_tensor_idx(input)
                opr.inputs[i] = cloned_graph.tensors[input_idx]
            for i, output in enumerate(graph.ops[idx].outputs):
                output_idx = graph.get_tensor_idx(output)
                opr.outputs[i] = cloned_graph.tensors[output_idx]

        # update sequence references if they exist
        if graph.op_sequence is not None:
            cloned_graph.update_sequence_references()

        # cloned_graph = copy.deepcopy(graph)

        return cloned_graph


class RemoveIdentityOps(GraphTranslator):

    def __init__(self, graph):
        super().__init__(graph)
        self.label = "Identity Operation Remover"
        self.description = "Translator which removes all pointless identity " \
                           "operations from a graph, merging surrounding " \
                           "tensors"

        self.remove_identity_ops()

    def remove_identity_ops(self):

        # mark ops and tensors to remove and update arcs to join gap
        ops_to_remove = []
        tensors_to_remove = []
        for opr in self.output_graph.ops:
            if opr.type == "Identity":
                input_tensor = opr.inputs[0]
                output_tensor = opr.outputs[0]
                input_tensor.dependent_ops = output_tensor.dependent_ops
                for dep_op in input_tensor.dependent_ops:
                    for i, input in enumerate(dep_op.inputs):
                        if input == output_tensor:
                            dep_op.inputs[i] = input_tensor

                ops_to_remove.append(opr)
                tensors_to_remove.append(output_tensor)

        for opr in ops_to_remove:
            opr_idx = self.output_graph.get_opr_idx(opr)
            self.output_graph.ops.remove(self.output_graph.ops[opr_idx])

        for tensor in tensors_to_remove:
            tensor_idx = self.output_graph.get_tensor_idx(tensor)
            self.output_graph.tensors.remove(
              self.output_graph.tensors[tensor_idx]
            )

        self.summary = ("Removed %d identity operations and associated tenors" %
                        len(ops_to_remove))

    def translate(self, verbose=False):
        if verbose:
          print(self.summary)
        return self.output_graph


class SequenceOps(GraphTranslator):

    def __init__(self, graph):
        super().__init__(graph)
        self.label = "Sequence Operations"
        self.description = "Sequences the operations of this graph using " \
                           "either eager or lazy ordering."

        self.sequence_ops()

    def sequence_ops(self):

        for opr in self.output_graph.ops:
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

        self.summary = ("Sequenced %d operations using eager execution." %
                        next_index)

    def translate(self, verbose=False):
        if verbose:
          print(self.summary)
        return self.output_graph


class FuseConvOps(GraphTranslator):

    def __init__(self, graph):
        super().__init__(graph)
        self.label = "Convolution Operation Fuser"
        self.description = "Translator which combines 2D conv and depthwise" \
                           "conv operations with any following bias and " \
                           "activation functions. As it done in TOCO."

        self.fuse_conv_ops()

    def fuse_conv_ops(self):

        conv_ops = ['Conv2D', 'DepthwiseConv2dNative']
        act_ops = ['Relu', 'Relu6', 'LeakyRelu']

        # search for the pattern of conv->bias->act operations within the graph
        count = 0
        ops_to_remove = []
        tensors_to_remove = []
        for opr in self.output_graph.ops:
            # ignore all patterns that do not match the fused conv operation
            if opr.type not in conv_ops:
                continue
            if len(opr.outputs[0].dependent_ops) != 1:
                continue
            bias_op = opr.outputs[0].dependent_ops[0]
            if bias_op.type != 'BiasAdd':
                continue
            # bias_tensor = bias_op.inputs[1]

            act_op = None
            if len(bias_op.outputs[0].dependent_ops) == 1 and \
                    bias_op.outputs[0].dependent_ops[0].type in act_ops:
                act_op = bias_op.outputs[0].dependent_ops[0]

            # opr.highlight_color = (50, 100, 50)
            # bias_op.highlight_color = (100, 50, 50)
            # bias_tensor.highlight_color = (100, 50, 50)
            # if act_op is not None:
            #     act_op.highlight_color = (50, 50, 100)
            count += 1

            # move bias tensor onto conv operation
            opr.inputs.append(bias_op.inputs[1])
            opr.inputs[2].dependent_ops = [opr]
            bias_op.inputs.remove(bias_op.inputs[1])

            # if no activation op was found just merge the conv and bias
            if act_op is None:
                output_tensor = bias_op.outputs[0]
                opr.outputs = [output_tensor]
                output_tensor.creating_op = opr
                ops_to_remove.append(bias_op)
                tensors_to_remove.append(bias_op.inputs[0])
            else:  # if bias and activation op was found merge them all
                output_tensor = act_op.outputs[0]
                opr.params['fused_activation_fn'] = act_op.type
                opr.outputs = [output_tensor]
                output_tensor.creating_op = opr
                ops_to_remove.extend([bias_op,
                                      act_op])
                tensors_to_remove.extend([bias_op.inputs[0],
                                          act_op.inputs[0]])

        for opr in ops_to_remove:
            opr_idx = self.output_graph.get_opr_idx(opr)
            self.output_graph.ops.remove(self.output_graph.ops[opr_idx])

        for tensor in tensors_to_remove:
            tensor_idx = self.output_graph.get_tensor_idx(tensor)
            self.output_graph.tensors.remove(
              self.output_graph.tensors[tensor_idx]
            )

        self.summary = ("Created %d fused convolution operations. Upd" %
                        count)

    def translate(self, verbose=False):
        if verbose:
          print(self.summary)
        return self.output_graph


"""def get_dtype_size(d_type):
    sizes = {'Float32': 4,
             'Float16': 2,
             'Int32': 4,
             'Uint8': 1,
             'Int64': 8,
             'Int16': 2,
             'Int8': 1}
    return sizes[d_type]"""


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
            self.value = source_tensor.value
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
        shape = []
        for dim in self.shape:
            if dim == -1 and batch_size is not None:
                dim = batch_size
            shape.append(dim)
        return shape

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
                return None
        return peak_memory

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
               element_count += np.prod(output.shape)
        return element_count
