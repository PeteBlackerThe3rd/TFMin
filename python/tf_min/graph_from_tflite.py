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

    This module contains the TFLite importer which generates a Graph
    object from a TFLine flatbuffer file.
"""
import copy
import numpy as np
import tf_min.graph as tg

# import flatbuffers and tflite schema interface
import flatbuffers as fb
import tflite.Model as TFLModel
import tflite.Tensor as TFLTensor
import tflite.Operator as TFLOperator
import tflite.BuiltinOptions as BuiltinOptions
import tflite.Conv2DOptions as Conv2DOptions
import tflite.Pool2DOptions as Pool2DOptions
import tflite.DepthwiseConv2DOptions as DepthwiseConv2DOptions
import tflite.BuiltinOperator as BuiltinOperator
import tflite.Metadata as Metadata
import tflite.Buffer as Buffer


TFLITE_OP_TRANSLATIONS = {'CONV_2D': 'Conv2D',
                          'DEPTHWISE_CONV_2D': 'DepthwiseConv2D',
                          'AVERAGE_POOL_2D': 'AvgPool',
                          'SOFTMAX': 'Softmax',
                          'RESHAPE': 'Reshape',
                          'MAX_POOL_2D': 'MaxPool',
                          'ADD': 'Add',
                          'SUB': 'Sub'}

TFLITE_PARAM_TRANSLATIONS = {'Conv2D': {'strideW': 'stride_width',
                                        'strideH': 'stride_height',
                                        'fusedActivationFunction': 'fused_activation_fn',
                                        'dilationWFactor': 'dilation_width_factor',
                                        'dilationHFactor': 'dilation_height_factor'},
                             'DepthwiseConv2D': {'strideW': 'stride_width',
                                                 'strideH': 'stride_height',
                                                 'fusedActivationFunction': 'fused_activation_fn',
                                                 'dilationWFactor': 'dilation_width_factor',
                                                 'dilationHFactor': 'dilation_height_factor'},
                             'MaxPool': {'strideW': 'stride_width',
                                         'strideH': 'stride_height',
                                         'filterWidth': 'filter_width',
                                         'filterHeight': 'filter_height'},
                             'AvgPool': {'strideW': 'stride_width',
                                         'strideH': 'stride_height',
                                         'filterWidth': 'filter_width',
                                         'filterHeight': 'filter_height'}
                             }

def tflite_act_transformer(tfl_act):
    acts = {0: None,
            1: 'Relu',
            2: 'ReluN1To1',
            3: 'Relu6',
            4: 'TanH',
            5: 'SignBit'}
    return acts[tfl_act]

def tflite_padding_transformer(tfl_pad):
    paddings = {0: 'SAME',
                1: 'VALID'}
    return paddings[tfl_pad]

TFLITE_PARAM_VALUE_TRANSFORMERS = {'Conv2D': {'fused_activation_fn': tflite_act_transformer,
                                              'padding': tflite_padding_transformer},
                                   'DepthwiseConv2D': {'fused_activation_fn': tflite_act_transformer,
                                                       'padding': tflite_padding_transformer}
                                   }


def tfl_type_to_tfmin(tfl_type):
    d_types = {0: 'Float32',
               1: 'Float16',
               2: 'Int32',
               3: 'Uint8',
               4: 'Int64',
               7: 'Int16',
               9: 'Int8'}
    if tfl_type not in d_types:
        return "Unknown_Type"
    return d_types[tfl_type]


def tflite_to_tensor(tflite_tensor):
    new_tensor = tg.Tensor()
    new_tensor.label = tflite_tensor.name.decode('utf-8')
    new_tensor.d_type = tfl_type_to_tfmin(tflite_tensor.type)
    new_tensor.shape = tflite_tensor.shape
    new_tensor.type = tg.TenType.INTERMEDIATE
    return new_tensor


def tflite_to_operation(tflite_opr, model):
    new_opr = tg.Operation()
    opr_code = model.operatorCodes[tflite_opr.opcodeIndex]

    custom_op = opr_code.customCode is not None

    if custom_op:
        new_opr.type = opr_code.customCode
    else:
        new_opr.type = \
            list(BuiltinOperator.BuiltinOperator.__dict__.keys())[
                list(
                    BuiltinOperator.BuiltinOperator.__dict__.values()).index(
                    opr_code.builtinCode)]

    # translate operation type from tflite to TFMin
    if new_opr.type in TFLITE_OP_TRANSLATIONS:
        new_opr.type = TFLITE_OP_TRANSLATIONS[new_opr.type]

    # print("Added operation [%s]" % new_opr.type)

    # if custom_op:
    #    print("importing a custom op. opr_code.customCode is [%s]" % opr_code.customCode)
    # else:
    #    print("importing a builting op. opr_code.customCode is [%s]" % opr_code.customCode)

    # print("Found op [%s]" % self.type)
    if not custom_op:
        if tflite_opr.builtinOptions is None:
            print("Warning non-custom operation doesn't have a builtin "
                  "Options attribute. Op is [%s]" % new_opr.type)
        else:
            for param, value in tflite_opr.builtinOptions.__dict__.items():
                # print("Found param [%s] with value [%s]" % (param, value))
                # translate parameter name from tflite to TFMin
                if new_opr.type in TFLITE_PARAM_TRANSLATIONS:
                    if param in TFLITE_PARAM_TRANSLATIONS[new_opr.type]:
                        param = TFLITE_PARAM_TRANSLATIONS[new_opr.type][param]
                # transform parameter value from tflite to TFMin
                if new_opr.type in TFLITE_PARAM_VALUE_TRANSFORMERS:
                    if param in TFLITE_PARAM_VALUE_TRANSFORMERS[new_opr.type]:
                        value = \
                          TFLITE_PARAM_VALUE_TRANSFORMERS[new_opr.type][param](value)
                new_opr.params[param] = value
                # print(" - Param [%s] %s" % (param, value))

    return new_opr

def graph_from_tflite(flatbuffer, sub_graph_idx=0):
    """
    method to populate this graph from the given tflite flatbuffer object
    :param flatbuffer: bytearray containing raw flatbuffer data
    :param sub_graph_idx: index of subgraph within flatbuffer to extract
    :return: populated TFMin Graph object
    """
    new_graph = tg.Graph()

    # extract model and sub-graph from flatbuffer
    n = fb.encode.Get(fb.packer.uoffset, flatbuffer, 0)
    model = TFLModel.ModelT.InitFromBuf(flatbuffer, n)
    graph = model.subgraphs[sub_graph_idx]

    # Add tensors from tflite model
    for idx, tensor in enumerate(graph.tensors):
        new_tensor = tflite_to_tensor(tensor)
        if idx in graph.inputs:
            new_tensor.type = tg.TenType.INPUT
        elif idx in graph.outputs:
            new_tensor.type = tg.TenType.OUTPUT
        new_graph.tensors.append(new_tensor)

    # Add operations and link to tensors
    for idx, opr in enumerate(graph.operators):
        new_op = tflite_to_operation(opr, model=model)
        for input_idx in opr.inputs:
            new_op.inputs.append(new_graph.tensors[input_idx])
            new_graph.tensors[input_idx].dependent_ops.append(new_op)
        for output_idx in opr.outputs:
            new_op.outputs.append(new_graph.tensors[output_idx])
            new_graph.tensors[output_idx].creating_op = new_op
        # tflite doesn't store operation labels so to give them a
        # sensible label here we use the label of their first output
        # tensor with a suffix
        new_op.label = new_op.outputs[0].label + "_opr"
        new_graph.ops.append(new_op)
        #print("Added %s operation with %d inputs and %d outputs." %
        #      (new_op.type,
        #       len(new_op.inputs),
        #       len(new_op.outputs)))

    # Mark constant tensors & get values
    for tensor in new_graph.tensors:
        if tensor.creating_op is None and tensor.type != tg.TenType.INPUT:
            tensor.type = tg.TenType.CONSTANT
            # TODO Get value from buffer

    return new_graph
