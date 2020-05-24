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

    Core Exporter object which is used by end user code to generate C++
    implementations of TensorFlow models.
"""
import os
import tensorflow as tf
import numpy as np
import struct
import copy
import textwrap
import tf_min.cpp_code_gen as code_gen
import tf_min.tf_utils as tf_utils
import tf_min.cpp_gen.cpp_gen as cpp_gen

# use the dynamic op_kernel_loader to import op_kernels from all requires paths
from tf_min.op_kernels import base_op
import tf_min.op_kernels.import_op_kernels as op_kernel_loader
op_kernel_loader.import_op_kernels(verbose=False)

from tf_min.mem_opt import *
from tf_min.mem_opt.base_optimiser import Operation as MemOptOperation
from tf_min.mem_opt.base_optimiser import Buffer as MemOptBuffer
from tf_min.mem_opt.heap_memory_allocator import HeapAllocator as HeapAllocator


def generate_boilerplate_model_class(class_name, memory_size):

    class_obj = cpp_gen.ClassDef(class_name, super_classes=['TFMin::ModelBase'])

    constructor = cpp_gen.ClassMethod(class_name, constructor=True, inline=False)
    constructor.comment = cpp_gen.Comment("Constructor\nAllocates memory for intermediate calculations\n"
                                          "and loads/sets up model weights.", style="/**")
    constructor.code_block.add_statement(cpp_gen.Statement("memoryBlock = new char[%d]" % memory_size))
    constructor.initialiser_list += ["TFMin::ModelBase()"]
    class_obj.add(constructor)

    destructor = cpp_gen.ClassMethod(class_name, destructor=True, inline=True)
    destructor.comment = cpp_gen.Comment("Destructor")
    destructor.code_block.add_statement(cpp_gen.Statement("delete[] memoryBlock"))
    class_obj.add(destructor)

    memory_block = cpp_gen.ClassProperty('memoryBlock', type=cpp_gen.TypeDefinition('char', ptr_levels=1))
    memory_block.comment = cpp_gen.Comment("Buffer used to hold intermediate tensor values during\n"
                                           "calculation of the model.")
    memory_block.access_modifier = 'private'
    class_obj.add(memory_block)

    evaluator_template = cpp_gen.TemplateDefinition()
    evaluator_template.add_element('Device', type='typename')

    explc_inst_default = cpp_gen.TemplateInstance()
    explc_inst_default.add_element(cpp_gen.TypeDefinition('Eigen::DefaultDevice'))

    eval = cpp_gen.ClassMethod('eval',
                               type=cpp_gen.TypeDefinition('void'),
                               template=evaluator_template,
                               explicit_instantiations=[explc_inst_default])
    eval.parameter_list.add(cpp_gen.Parameter('d', cpp_gen.TypeDefinition('Device', ref=True, const=True)))
    eval.comment = cpp_gen.Comment("Model evaluation method\n", style="/**")
    class_obj.add(eval)

    validate = cpp_gen.ClassMethod('validate',
                                   type=cpp_gen.TypeDefinition('bool'),
                                   template=evaluator_template,
                                   explicit_instantiations=[explc_inst_default])
    validate.parameter_list.add(cpp_gen.Parameter('d', cpp_gen.TypeDefinition('Device', ref=True, const=True)))
    validate.comment = cpp_gen.Comment("Model validation method\nexecutes a pre-defined calculation"
                                       "and returns true if correctly executed.\n", style="/**")

    timing = cpp_gen.ClassMethod('timing',
                                 type=cpp_gen.TypeDefinition('TimingResult', namespace='TFMin'),
                                 template=evaluator_template,
                                 explicit_instantiations=[explc_inst_default])
    timing.parameter_list.add(cpp_gen.Parameter('d', cpp_gen.TypeDefinition('Device', ref=True, const=True)))
    timing.comment = cpp_gen.Comment("Model timing method\nfunctionally the same as eval, but returns additional "
                                     "timing information.\n", style="/**")

    return [class_obj, constructor, eval, validate, timing]


class TensorMemoryBlock:

    def __init__(self, tensor, start, end, size):
        self.tensor = tensor
        self.start = start
        self.end = end
        self.size = size

        self.mem_offset = None


class Exporter(object):

    def __init__(self, sess, outputs, default_feed_dict={}):
        self.sess = sess
        tf_utils.active_session = sess
        self.output_tensor_names = outputs
        self.output_tensors = []
        self.default_feed_dict = default_feed_dict
        self.lib_namespace = "TFMin::"
        self.list_training_tensors = []
        self.list_input_placeholders = []
        self.list_operations = []
        self.list_verification_tensors = []
        self.memory_map = []
        self.memory_map_size = 0
        self.use_memory_map = False
        self.allocated_memory_areas = []
        self.print_ops_printed = []
        self.data_layout = 'ColMajor'
        self.validation_type = 'None'
        self.export_memory_trace = False

        # set default memory optimisation algorithm
        self.memory_opt_algorithm = HeapAllocator

        # add output tensors
        # (if any strings were given lookup tensors with that name)
        for out in outputs:
            if isinstance(out, tf.Tensor):
                self.output_tensors.append(out)
            else:
                try:
                    self.output_tensors.append(
                      self.sess.graph.get_tensor_by_name(out))
                except KeyError:
                    print("Error: No tensor named \"%s\" found in graph!" % out)
                    for opr in self.sess.graph.get_operations():
                        for ten in opr.outputs:
                            print("Did you mean : %s" % ten.name)

    def print_graph(self):

        tf_utils.print_ops_printed = []
        print("[%d] output tensors found okay." % len(self.output_tensors))
        for out_t in self.output_tensors:
            print("  <\"%s\" with size %s>" % (out_t.name, str(out_t.shape)))
            tf_utils.show_parent_of_tensor(out_t, "  ")

        print("-------------------------------------")

    def check_operations_supported(self, always_print=False):

        unsupported_list = {}
        production_list = {}
        dev_list = {}
        testing_list = {}
        op_kernels = op_kernel_loader.get_op_kernels()

        for op in self.list_operations:

            supported = False
            for k in op_kernels:
                if k.matches(op):
                    supported = True
                    if k.status() == "Development":
                        if op.type in dev_list:
                            dev_list[op.type] += 1
                        else:
                            dev_list[op.type] = 1
                    elif k.status() == "Testing":
                        if op.type in testing_list:
                            testing_list[op.type] += 1
                        else:
                            testing_list[op.type] = 1
                    else:
                        if op.type in production_list:
                            production_list[op.type] += 1
                        else:
                            production_list[op.type] = 1
                    break

            if not supported:
                if op.type in unsupported_list.keys():
                    unsupported_list[op.type] += 1
                else:
                    unsupported_list[op.type] = 1

        if len(unsupported_list) > 0:
            print("Error : This model uses %d types of operation that "
                  "are not currently supported by TFMin:" %
                  len(unsupported_list))
            for op_name, count in unsupported_list.items():
                print("%s (%d occurences)" % (op_name, count))
            # for op_name in unsupported_list:
            #    print(op_name)
            print("---------------------------------")
            return False

        if len(dev_list) > 0 or len(testing_list) > 0 or always_print:
            if len(dev_list) > 0 or len(testing_list) > 0:
                print("Warning : This model will use development "
                      "or testing OpKernels:")
            for t, c in testing_list.items():
                print("(  \033[91mTesting\033[39m  ) %s [%d occurences]" %
                      (t, c))
            for d, c in dev_list.items():
                print("(\033[93mDevelopment\033[39m) %s [%d occurences]" %
                      (d, c))
            for p, c in production_list.items():
                print("(\033[92mProduction\033[39m ) %s [%d occurences]" %
                      (p, c))

        return True

    def add_operations_to_method(self, method, type='eval'):

        if type == 'validate':
            base_op.BaseOpKernel.evaluate_all = True

        if type == 'timing':
            comment = cpp_gen.Comment("Timing working variables")
            method.code_block.add_statement(cpp_gen.Statement(
              self.lib_namespace + "TimingResult result", comment=comment))
            method.code_block.add_statement(cpp_gen.Statement("float start"))

        # Add operations, including validation and timing as required
        for idx, op in enumerate(self.list_operations):

            operation_code = cpp_gen.CodeBlock()

            if type == 'timing':
                operation_code.add_statement(
                  cpp_gen.Statement("start = getTime()"))

                print_if_statement = cpp_gen.IfStatement("print")
                print_if_statement.if_code.add_statement(cpp_gen.Statement(
                    "std::cout << \"Starting %s [%s]\" << std::endl" %
                    (op.name, op.type)))
                operation_code.add_statement(print_if_statement)

            if type == 'validate':
                operation_code.add_statement(
                    cpp_gen.Statement("std::cout << \"About to perform "
                                      "%s operation [%s]\\n\"" %
                                      (op.type, op.name))
                )

            if self.export_memory_trace:

                first_parameter = self.list_input_placeholders[0]
                source_identifier = code_gen.c_safe_identifier(
                  first_parameter.outputs[0].name)
                operation_code.add_statement(
                  cpp_gen.Statement("*(traceEvents[%d].addr) = "
                                    "*(int*)%s.data();" %
                                    (idx, source_identifier)))
                #identifier = code_gen.c_safe_identifier(op.name) + '_TraceEvent'
                #first_parameter = self.list_input_placeholders[0]
                #source_identifier = code_gen.c_safe_identifier(first_parameter.outputs[0].name)

                #operation_code.add_statement(
                #  cpp_gen.Statement("*(%s) = *(int*)%s.data()" %
                #                    (identifier,
                #                     source_identifier))
                #)

            # Find op_kernel for this operation type and generate code
            k = op_kernel_loader.find_op_kernel(op)
            if k is not None:
              op_code = k.generate(op)

              if isinstance(op_code, cpp_gen.CodeBlock):
                operation_code.add_block(op_code)
              else:
                op_statements = op_code.split(";")

                for s in op_statements:
                  if s.strip() != "":
                    operation_code.add_statement(
                      cpp_gen.Statement(s.strip()))

            if self.export_memory_trace:

                # identifier = code_gen.c_safe_identifier(op.name)+'_TraceEvent'

                operation_code.add_statement(
                  cpp_gen.Statement("std::cout << *(traceEvents[%d].addr)"
                                    " << std::endl" % idx))

            if type == 'timing':
                operation_code.add_statement(
                  cpp_gen.Statement("result.push_back(TFMin::OperationTime("
                                    "\"%s\", getTime()-start))" % op.name))

                print_if_statement = cpp_gen.IfStatement("print")
                print_if_statement.if_code.add_statement(cpp_gen.Statement(
                    "std::cout << \"Completed %s [%s]"
                    "operation\" << std::endl" % (op.name, op.type)))
                operation_code.add_statement(print_if_statement)

            if type == 'validate':
                for out in op.outputs:
                    identifier = code_gen.c_safe_identifier(out.name)
                    val_if = cpp_gen.IfStatement("!tensorsApproximatelyEqual("
                                                 "%s, %s_val, true)" %
                                                 (identifier, identifier))
                    val_if.if_code.add_statement(cpp_gen.Statement(
                        "std::cout << \"Validation failed at "
                        "operation [%s]\\n\"" % identifier))
                    val_if.if_code.add_statement(
                      cpp_gen.Statement("return false"))
                    operation_code.add_statement(val_if)

            op_comment = cpp_gen.Comment("Generated %s [%s] operation." %
                                         (op.name, op.type), style='//')
            operation_code.statements[0].comment = op_comment
            method.code_block.add_block(operation_code)

        if type == 'timing':

            if_print = cpp_gen.IfStatement("print")
            if_print.if_code.add_statement(
              cpp_gen.Statement("printTiming(result)"))
            method.code_block.add_statement(if_print)

            method.code_block.add_statement(cpp_gen.Statement("return result"))

        if type == 'validate':
            method.code_block.add_statement(cpp_gen.Statement("return true"))
            base_op.BaseOpKernel.evaluate_all = False

    def write_data_header(self,
                          file_name,
                          class_name,
                          validation_type='Full',
                          validation_inputs=None):

        # write model training data file
        with open(file_name, "w") as data_file:

            # write file header
            data_file.write("#ifndef __%s_WEIGHTS_H__\n" % class_name.upper())
            data_file.write("#define __%s_WEIGHTS_H__\n" % class_name.upper())
            data_file.write("//" + "-" * 80 + "\n")
            data_file.write("// Training data literal declarations.\n")
            data_file.write("// Generated by TFMin, do not edit.\n")
            data_file.write("//" + "-" * 80 + "\n")

            data_order = 'F'
            if self.data_layout == 'RowMajor':
                data_order = 'C'

            export_data = True  # Debugging aid to generate data header
            #                     without MBs of text so it is unloadable!

            data_file.write("namespace %sWeights\n{\n\n" % class_name)

            # evaluate and write model weights
            for tensor in self.list_training_tensors:
                # write flat version
                [var_value] = self.sess.run([tensor], {})

                identifier = code_gen.c_safe_identifier(tensor.name) + "Flat"
                flat_tensor_values = var_value.reshape(var_value.size,
                                                       order=data_order)

                tf_utils.write_numpy_array_c(data_file,
                                             "    " + identifier,
                                             flat_tensor_values,
                                             export_data)

            # if required add verification data
            if validation_type == "Full":

                # self.list_verification_tensors = self.output_tensors
                for op in self.list_operations:
                    for tensor in op.outputs:
                        self.list_verification_tensors += [tensor]

                for tensor in self.list_verification_tensors:

                    [verification_value] = self.sess.run([tensor],
                                                         validation_inputs)
                    identifier = (code_gen.c_safe_identifier(tensor.name) +
                                  "VerificationData")
                    flat_tensor_values = verification_value.reshape(
                      np.prod(verification_value.shape), order=data_order)
                    tf_utils.write_numpy_array_c(data_file,
                                                 "    " + identifier,
                                                 flat_tensor_values,
                                                 export_data)

            data_file.write("}\n\n")

            data_file.write("#endif  // __%s_WEIGHTS_H__\n" %
                            class_name.upper())

    @staticmethod
    def ensure_path_exists(path):

        directory = os.path.dirname(path)
        if directory != "":
            if not os.path.exists(directory):
                os.makedirs(directory)

    def add_weights_to_class(self, class_obj, constructor):

        # Add stored tensors to properties and constructor initialiser list
        for t in self.list_training_tensors:

            type = code_gen.get_c_dtype(t.dtype.base_dtype)
            rank = max(1, len(tf_utils.np_tensor_shape(t)))

            inner_template = cpp_gen.TemplateInstance()
            inner_template.add_element(cpp_gen.TypeDefinition(type))
            inner_template.add_element(str(rank))
            inner_template.add_element("Eigen::"+self.data_layout)
            template = cpp_gen.TemplateInstance()
            template.add_element(cpp_gen.TypeDefinition('Tensor', namespace='Eigen', template=inner_template))
            tensor_type = cpp_gen.TypeDefinition('TensorMap', namespace='Eigen', template=template)
            tensor_map_property = cpp_gen.ClassProperty(code_gen.c_safe_identifier(t.name),
                                                        tensor_type)
            tensor_map_property.access_modifier = "private"
            class_obj.add(tensor_map_property)

            # For now just use literal values, TODO add option to load weights from file as well
            literal_name = class_obj.identifier + "Weights::" + \
                           code_gen.c_safe_identifier(t.name) + "Flat"
            if type == "float" or type == "double" or type == "long double":
                literal_name += "Hex"
            shape = code_gen.ndarray_1d_to_literal(tf_utils.np_tensor_shape(t),
                                                   open='', close='')
            # convert rank zero tensor to rank 1 for eigen
            if shape == '  ':
                shape = ' 1 '

            constructor.initialiser_list += ["%s((%s*)%s,%s)" %
                                             (code_gen.c_safe_identifier(t.name),
                                              type,
                                              literal_name,
                                              shape)]

    def add_verification_to_class(self, class_obj, constructor):
        if self.validation_type == 'Full':
            for op in self.list_operations:
                for out in op.outputs:

                    identifier = code_gen.c_safe_identifier(out.name)
                    shape = tf_utils.np_tensor_shape(out)
                    if len(shape) == 0:
                        shape = [1]
                    type = code_gen.get_c_dtype(out.dtype)

                    inner_template = cpp_gen.TemplateInstance()
                    inner_template.add_element(cpp_gen.TypeDefinition(type))
                    inner_template.add_element(str(len(shape)))
                    inner_template.add_element("Eigen::"+self.data_layout)
                    template = cpp_gen.TemplateInstance()
                    template.add_element(cpp_gen.TypeDefinition('Tensor', namespace='Eigen', template=inner_template))
                    tensor_type = cpp_gen.TypeDefinition('TensorMap', namespace='Eigen', template=template)
                    tensor_map_property = cpp_gen.ClassProperty(identifier+"_val", tensor_type)
                    tensor_map_property.access_modifier = "private"
                    class_obj.add(tensor_map_property)
                    lit_suffix = ""
                    if type == "float" or type == "double" or type == "long double":
                        lit_suffix = "Hex"

                    literal_identifier = (class_obj.identifier + "Weights::" +
                                          identifier + "VerificationData" +
                                          lit_suffix)

                    constructor.initialiser_list += ["%s((%s*)%s,%s)" %
                                                     (identifier + "_val",
                                                      type,
                                                      literal_identifier,
                                                      code_gen.ndarray_1d_to_literal(shape, open='', close=''))]

    def add_parameters_to_methods(self,
                                  eval_method,
                                  validate_method,
                                  timing_method,
                                  class_name):
        parameter_comment = "Input tensors\n"
        for i, input_placeholder in enumerate(self.list_input_placeholders):
            type = code_gen.get_c_dtype(input_placeholder.outputs[0].dtype.base_dtype)
            identifier = code_gen.c_safe_identifier(input_placeholder.outputs[0].name)
            shape = tf_utils.np_tensor_shape(input_placeholder.outputs[0])
            if len(shape) == 0:
                shape = [1]

            parameter_comment += "[%s] %s %s\n" % (type,
                                                   identifier,
                                                   str(input_placeholder.outputs[0].shape[1:]))

            eval_method.parameter_list.add(cpp_gen.Parameter(identifier+"Param",
                                                             cpp_gen.TypeDefinition(type, ptr_levels=1)))
            timing_method.parameter_list.add(cpp_gen.Parameter(identifier+"Param",
                                                               cpp_gen.TypeDefinition(type, ptr_levels=1)))

            param_tensor_map = "Eigen::TensorMap<Eigen::Tensor" \
                               "<%s, %d, %s>> %s(%s,%s)" % \
                               (type,
                                len(shape),
                                "Eigen::"+self.data_layout,
                                identifier,
                                identifier+"Param",
                                code_gen.ndarray_1d_to_literal(shape,
                                                               open='',
                                                               close=''))

            val_data_identifier = (class_name + "Weights::" +
                                   identifier + "VerificationDataHex")

            val_tensor_map = ("Eigen::TensorMap<Eigen::Tensor"
                              "<%s, %d, %s>> %s((%s*)%s,%s)" %
                              (type,
                               len(shape),
                               "Eigen::"+self.data_layout,
                               identifier,
                               type,
                               val_data_identifier,
                               code_gen.ndarray_1d_to_literal(shape,
                                                              open='',
                                                              close='')))

            comment = None
            if i == 0:
                comment = cpp_gen.Comment("Creating TensorMaps of inputs")

            eval_method.code_block.add_statement(cpp_gen.Statement(param_tensor_map, comment))
            timing_method.code_block.add_statement(cpp_gen.Statement(param_tensor_map, comment))

            validate_method.code_block.add_statement(cpp_gen.Statement(val_tensor_map, comment))

        parameter_comment += "Output tensors\n"
        for out in self.output_tensors:
            type = code_gen.get_c_dtype(out.dtype)
            identifier = code_gen.c_safe_identifier(out.name)
            shape = tf_utils.np_tensor_shape(out)

            parameter_comment += "[%s] %s [%s]\n" % \
                                 (type,
                                  identifier,
                                  code_gen.ndarray_1d_to_literal(shape, open='', close=''))

            eval_method.parameter_list.add(cpp_gen.Parameter(identifier+"Param",
                                                             cpp_gen.TypeDefinition(type, ptr_levels=1)))
            timing_method.parameter_list.add(cpp_gen.Parameter(identifier+"Param",
                                                               cpp_gen.TypeDefinition(type, ptr_levels=1)))

            # create buffers to hold final output tensors in the validate method which doesn't actually
            # return anything to the calling process
            dummy_param = "%s %s[%d]" % (type, identifier+"Param", np.prod(shape))
            dummy_param_comment = cpp_gen.Comment("Dummy parameter for output")
            validate_method.code_block.add_statement(cpp_gen.Statement(dummy_param, dummy_param_comment))

            # Tag this tensor as an output so that operation kernels will
            # map the output to the given function parameter instead of a block in the memory map.
            # out.tfmin_is_output = True
            if out.op.type == 'Identity':
                out = out.op.inputs[0]
            out.tfmin_output_identifier = identifier+"Param"

        timing_method.parameter_list.add(cpp_gen.Parameter('print', cpp_gen.TypeDefinition('bool'), default='true'))
        eval_method.comment.text += parameter_comment
        timing_method.comment.text += parameter_comment

    def convert_val_inputs(self, validation_inputs):
        """ Converts any string keys into their corresponding placeholder references
        :param validation_inputs: dictionary of string/ref keys and numpy array values
        :return: dictionary of ref keys and numpy array values
        """
        converted_inputs = {}

        if type(validation_inputs) is dict:
            for key in validation_inputs:
                if type(key) is not str:
                    converted_inputs[key] = validation_inputs[key]
                else:
                    placeholder = self.sess.graph.get_tensor_by_name(key + ":0")
                    converted_inputs[placeholder] = validation_inputs[key]

        return converted_inputs

    def list_memory_optimisers(self):
      """
      Method to print a list of available memory optimisation algorithms.
      :return:
      """

      # list optimisation algorithms available
      optimisers = base_optimiser.BaseMemoryOptimiser.__subclasses__()

      print("  Optimisers available:")
      print("-------------------------")
      for optimiser in optimisers:
        opt = optimiser(op_list,
                        buffer_list,
                        alignment=4)
        print("[%s] - %s" %
              (opt.name(),
               opt.description()))
      print("-------------------------")

    def optimise_memory(self,
                        print_type=False,
                        export_filename=""):
      """
      Method which uses the current memory optimiser defined in
      self.memory_opt_algorithm to optimse operation order and buffer
      placement, to reduce memory requirements of inference.

      :param print_type: if True the current optimiser name and description
                         will be printed
      :param export_filename: name of file to dump debugging information to
      :return: None
      """

      # add intermediate buffers by looping through each operation used, and
      # adding each output tensor which is also used as an input to another
      # used operation
      buffer_list = []
      for op in self.list_operations:
        for output in op.outputs:
          output_is_used = False
          for inner_op in self.list_operations:
            if output in inner_op.inputs:
              output_is_used = True
              break

          if output_is_used:
            buffer_list += [MemOptBuffer(output)]

      # add operations in the order originally given
      op_list = []
      for op in self.list_operations:
        op_list += [MemOptOperation(op,
                                    buffer_list=buffer_list)]

      # optimise memory using the given optimiser
      optimiser = self.memory_opt_algorithm(op_list,
                                            buffer_list,
                                            alignment=4)

      if print_type:
        print("Optimising memory with %s:\n%s" %
              (optimiser.name(),
               optimiser.description()))

      # execute optimisation algorithm
      [self.memory_map_size,
       new_operation_order,
       allocated_buffers] = optimiser.optimise_memory()

      # safe structure of allocates memory areas
      if self.export_memory_trace:
        with open(export_filename, "w") as layout_csv:
          layout_csv.write("Creation #, Creation name, Final use #, Final use name, offset (bytes), size (bytes)\n")
          for buffer in allocated_buffers:

            #print("Creation is a [%s] with value [%s]" %
            #      (str(type(buffer.creation)), buffer.creation))
            memory_area = {'start_op':new_operation_order[buffer.creation].name,
                           'end_op':new_operation_order[buffer.final_use].name,
                           'offset':buffer.offset,
                           'size':buffer.size}
            self.allocated_memory_areas += [memory_area]

            layout_csv.write("%d, %s, %d, %s, %d, %d\n" %
                             (buffer.creation,
                              new_operation_order[buffer.creation].name,
                              buffer.final_use,
                              new_operation_order[buffer.final_use].name,
                              buffer.offset,
                              buffer.size))
          layout_csv.close()

      # update list_operations with new order
      self.list_operations = []
      for reordered_op in new_operation_order:
        self.list_operations += [reordered_op._underlying_op]

      if print_type:
        print("Optimised memory use. %d bytes required." % self.memory_map_size)

    def add_memory_trace(self, model_class, constructor):
        """
        add_memory_trace method, adds the properties, calls and template
        instantiations required to run this model attached to the memory
        tracer utility and analyse it's memory use pattern.
        :param model_class:
        :param constructor:
        :return:
        """

        # make the memory block pointer public
        model_class.element_by_identifier('memoryBlock').\
          access_modifier = "public"

        # add safe write space incase the calling process doesn't initialize
        # the event location pointers
        safe_write_space = cpp_gen.ClassProperty('safeWriteSpace', type=cpp_gen.TypeDefinition('int'))
        safe_write_space.comment = cpp_gen.Comment("Default location for event trace writes.")
        safe_write_space.access_modifier = 'private'
        model_class.add(safe_write_space)

        # add vector of trace pointers and operation names to class
        trace_events = cpp_gen.ClassProperty('traceEvents',
                                               type=cpp_gen.TypeDefinition('TFMin::MemoryTraceEvents'))
        model_class.add(trace_events)

        # add vector of memory areas to class
        memory_areas = cpp_gen.ClassProperty('memoryAreas',
                                             type=cpp_gen.TypeDefinition('TFMin::MemoryTraceAreas'))
        model_class.add(memory_areas)

        # populate events in class constructor
        for op in self.list_operations:
          constructor.code_block.add_statement(
            cpp_gen.Statement("traceEvents.push_back(TFMin::MemoryTraceEvent"
                              "(\"%s\", &safeWriteSpace))" % op.name))

        # populate memore areas in class constructor
        for area in self.allocated_memory_areas:
          constructor.code_block.add_statement(
            cpp_gen.Statement("memoryAreas.push_back(TFMin::MemoryTraceArea"
                              "(%d, %d, \"%s\", \"%s\"))" %
                              (area['offset'],
                               area['size'],
                               area['start_op'],
                               area['end_op'])))

        # add memory map size property
        memory_map_size = cpp_gen.ClassProperty('memoryMapSize',
                                                type=cpp_gen.TypeDefinition('unsigned long'))
        model_class.add(memory_map_size)
        constructor.code_block.add_statement(
          cpp_gen.Statement("memoryMapSize = %d" % self.memory_map_size))

        # add event trace pointers to each operation
        for op in self.list_operations:
          identifier = code_gen.c_safe_identifier(op.name) + '_TraceEvent'
          trace_pointer = cpp_gen.ClassProperty(identifier, type=cpp_gen.TypeDefinition('int', volatile=True, ptr_levels=1))
          trace_pointer.access_modifier = 'public'
          model_class.add(trace_pointer)

          constructor.code_block.add_statement(
            cpp_gen.Statement("%s = &safeWriteSpace" % identifier))

        # add 'Eigen::MemPreallocDevice' explicit instantiation to all
        # evaluation
        # explc_inst_pre_device = cpp_gen.TemplateInstance()
        # explc_inst_pre_thread_device = cpp_gen.TemplateInstance()
        # explc_inst_pre_device.add_element(cpp_gen.TypeDefinition('Eigen::MemPreallocDevice'))
        # explc_inst_pre_thread_device.add_element(cpp_gen.TypeDefinition('Eigen::ThreadPoolDevice'))

        # additional_explc_instationations = [explc_inst_pre_device,
        #                                     explc_inst_pre_thread_device]

        """eval_method = model_class.element_by_identifier("eval")
        eval_method.explicit_instantiations += additional_explc_instationations

        timing_method = model_class.element_by_identifier("timing")
        if timing_method is not None:
          timing_method.explicit_instantiations += additional_explc_instationations

        validate_method = model_class.element_by_identifier("validate")
        if validate_method is not None:
          validate_method.explicit_instantiations += additional_explc_instationations"""

    def tag_ops_to_resolve(self):

      # get list of available op_kernels
      op_kernels = base_op.BaseOpKernel.__subclasses__()

      for op in self.list_operations:
        concrete_needed = False
        for out_op in tf_utils.get_output_ops(op, self.list_operations):
          op_kernel = None
          for k in op_kernels:
            if k.matches(out_op):
              op_kernel = k

          assert op_kernel is not None
          if op_kernel.requires_concrete_inputs():
            # print("Tagging operation %s:%s as concrete "
            #       "because of output op %s:%s" %
            #      (op.type,
            #       op.name,
            #       out_op.type,
            #       out_op.name))
            concrete_needed = True

        op.tfmin_concrete_needed = concrete_needed

    def generate(self,
                 base_file_name,
                 class_name,
                 validation_type='None',
                 validation_inputs=None,
                 timing=False,
                 layout='ColMajor'):

        # python hack for mutable default parameters
        if validation_inputs is None:
            validation_inputs = []

        # convert any validation inputs defined by strings to their
        # corresponding placeholder references
        validation_inputs = self.convert_val_inputs(validation_inputs)

        self.data_layout = layout
        base_op.BaseOpKernel.data_layout = "Eigen::" + layout
        self.validation_type = validation_type

        # self.analyse_graph(self.output_tensors)

        # extract relavent lists from graph
        [self.list_input_placeholders,
         self.list_verification_tensors,
         self.list_training_tensors,
         self.list_operations] = tf_utils.build_graph_lists(self.output_tensors)
        print("Analysed flow-graph")

        Exporter.ensure_path_exists(base_file_name)
        if not self.check_operations_supported(always_print=True):
            return False

        self.optimise_memory(print_type=True,
                             export_filename=base_file_name + "_mem_layout.csv")
        self.use_memory_map = True
        base_op.BaseOpKernel.use_memory_map = self.use_memory_map
        print("Optimised memory map.")

        # Now the operations are in their final order. Tag operations which
        # need to be resolved to concrete tensors because of the following
        # operations
        self.tag_ops_to_resolve()

        # Define C++ class of model
        [model_class,
         constructor,
         eval_method,
         validate_method,
         timing_method] = generate_boilerplate_model_class(class_name,
                                                           self.memory_map_size)

        # Add stored tensors to class properties and
        # constructor initialiser list
        self.add_weights_to_class(model_class, constructor)

        # if a memory trace is requested then added the required volatile vars
        if self.export_memory_trace:
            self.add_memory_trace(model_class, constructor)

        # Add validation tensor maps to class and constructor
        # initialiser list if needed
        self.add_verification_to_class(model_class, constructor)

        # Add parameters and comments to eval and timing methods
        self.add_parameters_to_methods(eval_method,
                                       validate_method,
                                       timing_method,
                                       class_name)

        # Add operations to required class methods
        self.add_operations_to_method(eval_method, 'eval')
        if validation_type != 'None':
            self.add_operations_to_method(validate_method, 'validate')
            model_class.add(validate_method)
        if timing:
            self.add_operations_to_method(timing_method, 'timing')
            model_class.add(timing_method)

        # Write C++ class source files
        (path, new_base_file_name) = os.path.split(base_file_name)
        source = cpp_gen.SourcePair(new_base_file_name)

        source.dependencies += [cpp_gen.Dependency('tf_min.h')]
        # if self.export_memory_trace:
        #   source.dependencies +=\
        #       [cpp_gen.Dependency('TensorDeviceMemPrealloc.h')]
        model_data_header = cpp_gen.Dependency(new_base_file_name+'_data.h',
                                               type='local')
        source.definition_dependencies += [model_data_header]
        source.classes += [model_class]

        source.comment_h = cpp_gen.Comment("%s model class definition\n"
                                           "Automatically generated by TFMin"
                                           ", do not edit." % class_name,
                                           style='//--')

        source.comment_cpp = cpp_gen.Comment("%s model class declaration\n"
                                             "Automatically generated by TFMin"
                                             ", do not edit." % class_name,
                                             style='//--')

        source.write(path)

        print("Wrote C++ code.")

        self.write_data_header(base_file_name + "_data.h",
                               class_name=class_name,
                               validation_type=validation_type,
                               validation_inputs=validation_inputs)
        print("Generated data header.")

        return True
