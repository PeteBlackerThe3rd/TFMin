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

    This module contains the verify object which can generate, built, and
    execute graphs within a special test harness allowing the input and output
    tensors to be fed in via stdin and read out via stdout. The object uses
    this test harness to compare the output of the built model to some
    expected output.

    This object is used within the unit test framework of this project to
    verify that the op_kernels used to generate c code for each operation
    are correctly matching the original Tensorflow operations.
"""
import scipy.io
import numpy as np
import struct
import argparse
import sys
import os
import shutil
import fcntl
import subprocess as sp
import select

from tf_min import graph as tfm_g
from tf_min import types
# from tf_min import cpp_code_gen as c_gen
from tf_min import graph_c_gen
import tf_min.v2_kernels.base_op_kernel as base


class VerificationResult:
    """

    """

    def __init__(self, success):
        """
        Create a VerificationResult object
        :param success: if this object is representing a successful result
        """
        self.success = success
        self.output_failures = []
        self.output_failure_descriptions = []
        self.build_failure = False
        self.tensor_count_missmatch = False
        self.binary_timeout = False
        self.type_error = False
        self.type_error_msg = ""


class GraphVerifyOutput:
    """
    Object which generates unit_test versions of a graph, builds them and
    compares their output to a set of expected outputs.
    """
    MAKEFILE_TEMPLATE = """
# Verify output test makefile
# automatically generated by TFMin. Do not edit.

compiler = <compiler>
c_flags = -O3 -LM

test : test.c <base_name>model.c <base_name>weights.c
\t$(compiler) $(c_flags) test.c <base_name>model.c <base_name>weights.c -o test
"""

    SOURCE_TEMPLATE = """
#include <stdio.h>
#include <stdlib.h>
#include "<basename>model.h"

int main(int argc, char **argv[]) {

    // allocate tensor arena
    void *tensorArena = malloc(TENSOR_ARENA_SIZE);
    if (tensorArena == NULL) {
        fprintf(stderr, "Failed to allocate tensor arena of %d bytes\\n",""" + \
                      """ TENSOR_ARENA_SIZE);
        return 1;
    }
    
    // input and output tensor info
    int inputCount = <input_count>;
    int outputCount = <output_count>;
    unsigned int inputSizes[] = {<input_sizes>};
    unsigned int outputSizes[] = {<output_sizes>};
    
    // allocate input tensor buffers
    void *inputBuffers[<input_count>];
    for (int i = 0; i < inputCount; ++i) {
        inputBuffers[i] = malloc(inputSizes[i]);
        if (inputBuffers[i] == NULL) {
            fprintf(stderr, "Failed to allocate input buffer %d\\n", i);
            return 1; 
        }
    }
    
    // allocate output tensor buffers
    void *outputBuffers[<output_count>];
    for (int i = 0; i < outputCount; ++i) {
        outputBuffers[i] = malloc(outputSizes[i]);
        if (outputBuffers[i] == NULL) {
            fprintf(stderr, "Failed to allocate output buffer %d\\n", i);
            return 1; 
        }
    }
    
    // read input buffers from stdin
    freopen(NULL, "rb", stdin);
    for (int i = 0; i < <input_count>; ++i) {
        unsigned int count;
        count = fread(inputBuffers[i], sizeof(char), inputSizes[i], stdin);
        if (count != inputSizes[i]) {
            fprintf(stderr, "Failed to read the correct tensor buffer """ + \
                      """size from stdin.\\n");
            return 1;
        }
    }
    
    // execute inference model on test
    model(tensorArena, <parameters>);
    
    // write output buffers to stdout
    freopen(NULL, "wb", stdout);
    for (int i = 0; i < <output_count>; ++i) {
        unsigned int count;
        count = fwrite(outputBuffers[i], sizeof(char), outputSizes[i], stdout);
        if (count != outputSizes[i]) {
            fprintf(stderr, "Failed to write the correct tensor buffer """ + \
                      """size to stdout.\\n");
            return 1;
        }
    }
    
    // free memory and exit successfully
    free(tensorArena);
    for (int i = 0; i < inputCount; ++i)
        free(inputBuffers[i]);
    for (int i = 0; i < outputCount; ++i)
        free(outputBuffers[i]);
    exit(0);
}
    """

    def __init__(self, graph, verbose=False, tmp_dir=None, compiler="gcc"):
        """
        Creates a verification object. This initialisation generates the c
        project including the generated version of the graph, and compiles it
        ready for verification runs.
        Any errors in these steps are flagged
        :param graph: tf_min.Graph object, the inference model to verify
        """
        self.graph = graph
        self.ready_to_execute = True
        self.base_name = "test_"
        self.compiler = compiler
        self.tmp_dir = os.path.join('.', 'test_verify_output')
        if tmp_dir is not None:
            self.tmp_dir = tmp_dir

        # generate the c test implementation of this graph
        c_generator = graph_c_gen.CodeGenerator(graph=graph,
                                                base_name=self.base_name,
                                                prefix="",
                                                path=tmp_dir,
                                                clang_format='Google',
                                                byte_order="@", batch_size=1
                                                )
        if c_generator(silent=True):
            if verbose:
                print("Generated c code for model on test okay.")
        else:
            print("Error: Failed to generate c code for model on test.")
            self.ready_to_execute = False
            return

        # generate makefile and test harness entry point c source
        if self.generate_test_harness():
            if verbose:
                print("Generated test harness files okay.")
        else:
            print("Error: Failed to generate test harness files.")
            self.ready_to_execute = False
            return

        # attempt to build the test harness c project
        if self.build_test_harness():
            if verbose:
                print("Built test harness successfully.")
        else:
            print("Error: Failed to build test harness.")
            self.ready_to_execute = False
            return

    def __del__(self):
        """
        Destructor, detects if the tmp_dir exists and deletes it if it does
        :return: None
        """
        #if os.path.exists(self.tmp_dir):
        #    shutil.rmtree(self.tmp_dir)

    def generate_test_harness(self):
        """
        Method which generates the makefile and test harness c source
        files for this test
        :return: True on success, False on failure
        """

        # generate makefile
        try:
            makefile = open(os.path.join(self.tmp_dir, 'Makefile'), 'w')

            makefile.write(base.BaseOpKernel.process_template(
              GraphVerifyOutput.MAKEFILE_TEMPLATE,
              {"<compiler>": self.compiler,
               "<base_name>": self.base_name}
            ))
            makefile.close()
        except IOError:
            print("Error: Failed to create Makefile.")
            return False

        # generate test harness source
        try:
            source = open(os.path.join(self.tmp_dir, "test.c"), "w")

            input_sizes = []
            for input in self.graph.get_inputs():
                input_sizes.append(str(input.get_buffer_size()))
            output_sizes = []
            for output in self.graph.get_outputs():
                output_sizes.append(str(output.get_buffer_size()))

            parameters = []
            for idx, input in enumerate(self.graph.get_inputs()):
                parameters.append("(%s*)inputBuffers[%d]" % (
                  types.get_dtype_c_type(input.d_type),
                  idx
                ))
            for idx, output in enumerate(self.graph.get_outputs()):
                parameters.append("(%s*)outputBuffers[%d]" % (
                  types.get_dtype_c_type(output.d_type),
                  idx
                ))

            source.write(base.BaseOpKernel.process_template(
              GraphVerifyOutput.SOURCE_TEMPLATE,
              {"<basename>": self.base_name,
               "<input_count>": len(input_sizes),
               "<output_count>": len(output_sizes),
               "<input_sizes>": ', '.join(input_sizes),
               "<output_sizes>": ', '.join(output_sizes),
               "<parameters>": ', '.join(parameters)}
            ))
            source.close()
        except IOError:
            print("Error: Failed to create test.c source file.")
            return False

        return True

    def build_test_harness(self):
        """
        Method to build the generated C inference model and test harness entry
        point
        :return: True if successful, False otherwise
        """
        try:
            sp.run(['make', '-C', self.tmp_dir],
                   check=True,
                   stdout=sp.PIPE, stderr=sp.PIPE)
        except sp.CalledProcessError as e:
          print("Error: Failed to build test harness.\n")
          print("Return code %d" % e.returncode)
          print("Error:\n%s" % e.output.decode('utf-8'))
          return False

        return True

    def verify_model(self, input_tensors,
                     expected_output_tensors,
                     timeout=30.0,
                     tollerance=None):
        """
        Method to verify the output of the inference model on test.
        This function accepts a list of input tensor values as numpy.ndarrays
        and a list of expected output tensor values as numpy.ndarrays.
        The build C model is executed and input tensor values piped in via
        stdin then the generated output tests read back from stdout.

        The generated output values are then compared with the expected output
        values, and an object with pass/fail status and additional information
        is returned.
        :param input_tensors:
        :param expected_output_tensors:
        :param timeout: maximum timeout to wait for test binary to generate
                        output tensors.
        :return: tfmin.VerificationResult object
        """

        if self.ready_to_execute is not True:
            print("Error: Cannot execute model, it has not been "
                  "successfully built.")
            result = VerificationResult(False)
            result.build_failure = True
            return result

        # check the number of input and outputs tensors matches the graph
        if (len(input_tensors) != len(self.graph.get_inputs()) or
                len(expected_output_tensors) != len(self.graph.get_outputs())):
            print("Error: number of input and output tensors doesn't "
                  "match graph.")
            result = VerificationResult(False)
            result.tensor_count_missmatch = True
            return result

        # check the data type of input and expected output ndarrays match
        # the data types of the graph
        for idx, input in enumerate(input_tensors):
            if not isinstance(input, np.ndarray):
                print("Error: input tensor is not a numpy.ndarray.")
            given_type = types.np_to_tfmin(input.dtype)
            expected_type = self.graph.get_inputs()[idx].d_type
            if given_type != expected_type:
                msg = ("Error: Input tensor type missmatch, expected %s "
                       "but was given %s!" % (expected_type, given_type))
                print(msg)
                result = VerificationResult(False)
                result.type_error = True
                result.type_error_msg = msg
                return result
        for idx, output in enumerate(expected_output_tensors):
            if not isinstance(output, np.ndarray):
                print("Error: input tensor is not a numpy.ndarray.")
            given_type = types.np_to_tfmin(output.dtype)
            expected_type = self.graph.get_outputs()[idx].d_type
            if given_type != expected_type:
                msg = ("Error: Output tensor type missmatch, expected %s "
                       "but was given %s!" % (expected_type, given_type))
                print(msg)
                result = VerificationResult(False)
                result.type_error = True
                result.type_error_msg = msg
                return result

        # start c test binary process
        test_process = sp.Popen(os.path.join(self.tmp_dir, "test"),
                                shell=False,
                                stdin=sp.PIPE,
                                stdout=sp.PIPE,
                                universal_newlines=False)
        fd = test_process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        # pipe input tensor values to the c test binary
        for input in input_tensors:

            # TODO need to make this respect the data layout of the tensor
            flat_values = input.flatten()

            d_type = types.np_to_tfmin(input.dtype)
            struct_d_type = types.get_dtype_struct_type(d_type)
            buffer = struct.pack(struct_d_type * flat_values.size,
                                 *flat_values)
            test_process.stdin.write(buffer)
            test_process.stdin.flush()

        # wait for the given timeout for data to be available
        readable, _, _ = select.select([test_process.stdout], [], [], timeout)
        if len(readable) == 0:
            print("Error: timed out waiting for output tensor values from "
                  "binary on test.")
            result = VerificationResult(False)
            result.binary_timeout = True
            return result

        # read output tensor values from the c test binary
        generated_output_tensors = []
        for output in self.graph.get_outputs():
            buffer = test_process.stdout.read(output.get_buffer_size())
            if len(buffer) != output.get_buffer_size():
                print("Error: Failed to read correct buffer size for output.")
            struct_d_type = types.get_dtype_struct_type(output.d_type)
            element_count = output.shape.get_element_count()
            flat_values = struct.unpack(struct_d_type * element_count,
                                        buffer)
            generated_tensor = np.array(flat_values)
            generated_tensor = \
                generated_tensor.reshape(output.shape.get_shape(1))
            generated_output_tensors.append(generated_tensor)
            # TODO create numpy.ndarray with correct layout from this data

        print("Completed communication with c test binary")
        if test_process.poll() is not None:
            test_process.terminate()

        correct = True

        for idx, expected_output in enumerate(expected_output_tensors):

            # use the given tollerance and if none was given then use
            # default for this type
            this_tollerance = tollerance
            if this_tollerance is None:
                d_type = self.graph.get_outputs()[idx].d_type
                if d_type == types.TenDType.FLOAT64:
                    this_tollerance = 1e-12
                elif d_type == types.TenDType.FLOAT32:
                    this_tollerance = 1e-6
                else:
                    this_tollerance = 0

            errors = np.absolute(expected_output -
                                 generated_output_tensors[idx])
            if np.amax(errors) > this_tollerance:
                correct = False
                print("Output %d [%s] output differs by %f" % (
                  idx,
                  self.graph.get_outputs()[idx].label,
                  np.amax(errors)
                ))
                print("output [%d]" % idx)
                print("Expected:\n%s" % expected_output)
                print("Generated:\n%s\n------" % generated_output_tensors[idx])

        if correct:
            print("Verification Passed.")

        result = VerificationResult(correct)
        return result
