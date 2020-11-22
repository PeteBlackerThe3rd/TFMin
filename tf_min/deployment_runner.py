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

    This module contains the DeploymentRunner object which can generate,
    build, and execute graphs within a special test harness allowing the
    input and output tensors to be fed in via stdin and read out via stdout.

    This object is used by the graph comparison and graph test objects,
    to collect the results of a model deployed in a particular way
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

# from tf_min import graph as tfm_g
from . import types
# from tf_min import types
# from tf_min import cpp_code_gen as c_gen
from . import graph_c_gen
from . import exceptions as exc
from .op_kernels.base_op_kernel import BaseOpKernel


class DeploymentRunner:
  """
  Object which deploys a test versions of a graph, builds it and
  provides functions to execute the deployed graph feeding data in and
  recieving results
  """
  MAKEFILE_TEMPLATE = """
# Verify output test makefile
# automatically generated by TFMin. Do not edit.

compiler = <compiler>
c_flags = <flags> -LM

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

  GENERATOR = graph_c_gen.CodeGenerator

  def __init__(self, graph, verbose=False, tmp_dir=None,
               compiler="gcc",
               flags=['-O3']):
    """
    Creates a verification object. This initialisation generates the c
    project including the generated version of the graph, and compiles it
    ready for verification runs.
    Any errors in these steps are flagged
    :param graph: tf_min.Graph object, the inference model to verify
    :param verbose: Boolean, true prints out debugging information
    :param tmp_dir: String or None, if given overrides the default
                    build directory name
    :param compiler: String, compiler command to use
    :param flags: List of strings, additional flags to pass to the compiler
    """
    self.graph = graph
    self.ready_to_execute = False
    self.base_name = "test_"
    self.compiler = compiler
    self.tmp_dir = os.path.join('.', 'test_verify_output')
    if tmp_dir is not None:
      self.tmp_dir = tmp_dir

    # generate the c test implementation of this graph
    c_generator = self.GENERATOR(graph=graph,
                                 base_name=self.base_name,
                                 prefix="",
                                 path=self.tmp_dir,
                                 clang_format='Google',
                                 byte_order="@", batch_size=1
                                 )
    c_generator(silent=True)
    if verbose:
      print("Generated c code for model on test okay.")

    # generate makefile and test harness entry point c source
    self.generate_test_harness(flags)
    if verbose:
      print("Generated test harness files okay.")

    # attempt to build the test harness c project
    self.build_test_harness()
    if verbose:
      print("Built test harness successfully.")

  def __del__(self):
    """
    Destructor, detects if the tmp_dir exists and deletes it if it does
    :return: None
    """
    # if os.path.exists(self.tmp_dir):
    #     shutil.rmtree(self.tmp_dir)

  def generate_test_harness(self, flags):
    """
    Method which generates the makefile and test harness c source
    files for this test
    :param flags: List of string, additional compiler flags
    :return: True on success, False on failure
    """

    # generate makefile
    try:
      makefile = open(os.path.join(self.tmp_dir, 'Makefile'), 'w')
      makefile.write(BaseOpKernel.process_template(
        self.MAKEFILE_TEMPLATE,
        {"<compiler>": self.compiler,
         "<base_name>": self.base_name,
         "<flags>": " ".join(flags)}
      ))
      makefile.close()
    except IOError as e:
      raise exc.TFMinDeploymentFailed("Error: Failed to create "
                                      "Makefile.\n%s" % str(e))

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

      source.write(BaseOpKernel.process_template(
        self.SOURCE_TEMPLATE,
        {"<basename>": self.base_name,
         "<input_count>": len(input_sizes),
         "<output_count>": len(output_sizes),
         "<input_sizes>": ', '.join(input_sizes),
         "<output_sizes>": ', '.join(output_sizes),
         "<parameters>": ', '.join(parameters)}
      ))
      source.close()
    except IOError as e:
      raise exc.TFMinDeploymentFailed("Error: Failed to test.c source "
                                      "file.\n%s" % str(e))

  def build_test_harness(self):
    """
    Method to build the generated C inference model and test harness entry
    point
    """
    try:
      sp.run(['make', '-C', self.tmp_dir],
             check=True,
             stdout=sp.PIPE, stderr=sp.PIPE)
      self.ready_to_execute = True
    except sp.CalledProcessError as e:
      msg = ("Error: Failed to build test harness.\n"
             "Return code %d\nError: %s" % (e.returncode,
                                            e.output.decode('utf-8')))
      raise exc.TFMinDeploymentFailed(msg)

  def execute_model(self,
                    input_tensors,
                    timeout=30.0):
    """
    Method to execute the test deplotment of the model.
    This function accepts input tensor values as numpy.ndarrays
    which are piped to the model on test via stdin then the output
    tensors are read back from stdout.
    :param input_tensors: Dictionary of of input names and values
    :param timeout: maximum timeout to wait for test binary to generate
                    output tensors.
    :return: dictionary of output tensor names and values
    """

    if self.ready_to_execute is not True:
      raise exc.TFMinDeploymentExecutionFailed(
        "Error: Cannot execute model, it has not been successfully built."
      )

    # check the number of input tensors matches the graph
    if len(input_tensors) != len(self.graph.get_inputs()):
      raise exc.TFMinDeploymentExecutionFailed(
        "Error: number of provided input tensors (%d) doesn't match "
        "graph (%d)" % (len(input_tensors),
                        len(self.graph.get_inputs()))
      )

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
        raise exc.TFMinDeploymentExecutionFailed(msg)

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
      raise exc.TFMinDeploymentExecutionFailed(
        "Error: timed out waiting for output tensor values from binary "
        "on test."
      )

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
      generated_tensor = generated_tensor.astype(
        types.tfmin_to_np(output.d_type)
      )

      # TODO create numpy.ndarray with correct layout from this data

      generated_output_tensors.append(generated_tensor)

    # print("Completed communication with c test binary")
    # ensure test binary has terminated
    if test_process.poll() is not None:
      test_process.terminate()

    return generated_output_tensors