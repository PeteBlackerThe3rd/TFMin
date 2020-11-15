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

    This module contains the DeploymentTimerLeon object, this is a
    specialisation of the DeploymentTimer object which generates leon
    binaries and executes the models using the TSIM emulator to get accurate
    timing results on this target processor
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
from .deployment_timer import DeploymentTimer


class DeploymentTimerLeon(DeploymentTimer):
  """
  Specialisation of the DeploymentTimer object which collects per layer
  timing information from the inference implementation running on an
  emulated LEON processor.
  """

  def __init__(self, graph, verbose=False, tmp_dir=None,
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
    # override compiler and add extra flags
    compiler = "sparc-gaisler-elf-gcc"
    flags.append('-mcpu=leon3')
    DeploymentTimer.__init__(self, graph, verbose, tmp_dir, compiler, flags)

  def time_model(self,
                 iterations,
                 timeout=30.0):
    """
    Overidden method which executes a timing build of this model on the TSIM
    LEON emulator and captures the results. returning a list of times in
    secords for each layer, this list matches the sequenced order of the
    input graph.
    :param iterations: Ignored in this specialisation
    :param timeout: maximum timeout to wait for test binary to generate
                    output tensors.
    :return: List of float, seconds per layers ordered in sequence.
    """

    if self.ready_to_execute is not True:
      raise exc.TFMinDeploymentExecutionFailed(
        "Error: Cannot execute timing model, it has not been "
        "successfully built."
      )

    # loop running the model in TSim until all layers durations have been
    # captured. If a single run doesn't complete in the time limit
    # then rebuild the test without the first layers what have complete
    # and run it again.
    layer_durations = []

    while len(layer_durations) != len(self.graph.op_sequence):

      # start c test binary process
      test_binary_path = os.path.join(self.tmp_dir, "test")
      command = "tsim-leon3 %s -freq 200 -e run -e quit" % test_binary_path
      test_process = sp.Popen(command,
                              shell=True,
                              stdin=sp.PIPE,
                              stdout=sp.PIPE,
                              universal_newlines=True)

      # read timing results from test binary
      out, err = test_process.communicate()

      # print("--- tsim output ---\n %s\n-------" % out)

      # extract results and add new results from this iteration
      new_layer_durations = self.parse_timing_results(out)
      for idx, duration in enumerate(new_layer_durations):
        if idx >= len(layer_durations):
          layer_durations.append(duration)

      # if any layer results are missing replace timed layers with NOPs
      # rebuild the test binary and iterate again.
      if len(layer_durations) < len(self.graph.op_sequence):

        # print("Timed %d of %d operations, adding NOPs rebuilding "
        #       "test harness and running another time." %
        #       (len(layer_durations),
        #        len(self.graph.op_sequence)))

        for idx in range(len(layer_durations)):
          self.graph.op_sequence[idx].type = "NOP"

        c_generator = self.GENERATOR(graph=self.graph,
                                     base_name=self.base_name,
                                     prefix="",
                                     path=self.tmp_dir,
                                     clang_format='Google',
                                     byte_order="@", batch_size=1
                                     )
        c_generator(silent=True)
        self.build_test_harness()

    layer_durations = np.array(layer_durations)
    total_duration = layer_durations.sum()

    '''for idx, opr in enumerate(self.graph.op_sequence):
      print("Layer [%02d] %s took %f seconds" %
            (idx, opr.type, layer_durations[idx]))'''

    return layer_durations, total_duration