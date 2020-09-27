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

    This module contains the DeploymentTimer object, this is a specialisation
    of the DeploymentRunner object which generates sepcial timing
    implementations of models. These models are executed and per layer
    timing information is extracted and returned.
"""
import scipy.io
import numpy as np
import struct
import argparse
import sys
import os
import re
import shutil
import fcntl
import subprocess as sp
import select

from tf_min import graph as tfm_g
from tf_min import types
# from tf_min import cpp_code_gen as c_gen
# from tf_min import graph_c_gen
from tf_min import exceptions as exc
from .timer_code_gen import TimingCodeGenerator
from .deployment_runner import DeploymentRunner


class DeploymentTimer(DeploymentRunner):
  """
  Specialisation of the DeploymentRunner object which collects per layer
  timing information from the inference implementation.
  """

  def __init(self, graph, verbose=False, tmp_dir=None,
             compiler="gcc",
             flags=['-O3']):
    """

    :param graph:
    :param verbose:
    :param tmp_dir:
    :param compiler:
    :param flags:
    :return:
    """
    DeploymentRunner.__init__(self, graph, verbose, tmp_dir, compiler, flags)

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

    // execute inference model and print timing to stdout
    model(tensorArena, <parameters>);

    // free memory and exit successfully
    free(tensorArena);
    for (int i = 0; i < inputCount; ++i)
        free(inputBuffers[i]);
    for (int i = 0; i < outputCount; ++i)
        free(outputBuffers[i]);
    exit(0);
}
    """

  GENERATOR = TimingCodeGenerator

  def time_model(self,
                 iterations,
                 timeout=30.0):
    """
    Overidden method which executes a timing build this model and
    captures the results. returning a list of times in secords for each layer,
    this list matches the sequenced order of the input graph.
    :param iterations: Integer, the number of timing executions to run.
    :param timeout: maximum timeout to wait for test binary to generate
                    output tensors.
    :return: List of float, seconds per layers ordered in sequence.
    """

    if self.ready_to_execute is not True:
      raise exc.TFMinDeploymentExecutionFailed(
        "Error: Cannot execute timing model, it has not been "
        "successfully built."
      )

    layer_results = np.zeros((iterations, len(self.graph.op_sequence)))

    # run timing test N times
    for i in range(iterations):
      # start c test binary process

      test_process = sp.Popen(os.path.join(self.tmp_dir, "test"),
                              shell=False,
                              stdin=sp.PIPE,
                              stdout=sp.PIPE,
                              universal_newlines=True)

      # read timing results from test binary
      out, err = test_process.communicate()

      # extract results and find total runtime
      results = self.parse_timing_results(out)
      if len(results) != len(self.graph.op_sequence):
        print("---- ERROR ----------------------------")
        print("Timing model binary didn't produce the correct output")
        results = [0.0] * len(self.graph.op_sequence)
      layer_results[i, :] = results

    layer_durations = np.median(layer_results, 0)

    total_duration = layer_durations.sum()

    '''for idx, opr in enumerate(self.graph.op_sequence):
      print("Layer [%02d] %s took %f seconds" %
            (idx, opr.type, layer_durations[idx]))'''

    return layer_durations, total_duration

  @staticmethod
  def parse_timing_results(results):
    """
    Method which parses the text output of a timing implementation
    into a list of floating point durations in seconds
    :param results: String, input text
    :return: List of floats
    """
    result_pattern = r"(\d+),\s+(\d*\.\d+|\d+)"

    last_time = 0.0
    layer_durations = []

    for line in results.split("\n"):
      match = re.match(result_pattern, line)
      if match:
        index, time_s = match.groups()
        index = int(index)
        time_s = float(time_s)
        assert index == len(layer_durations), \
            "Error out of order timing result found!"
        duration = time_s - last_time
        last_time = time_s
        layer_durations.append(duration)

    return np.array(layer_durations)
