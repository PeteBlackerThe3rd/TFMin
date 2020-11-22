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

from .deployment_runner import DeploymentRunner
from . import types


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

  def passed(self):
    return self.success


class GraphVerifyOutput(DeploymentRunner):
    """
    Object which generates unit_test versions of a graph, builds them and
    compares their output to a set of expected outputs.
    """

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

        actual_outputs = self.execute_model(input_tensors,
                                            timeout=timeout)

        correct = True

        for idx, expected_output in enumerate(expected_output_tensors):

            # use the given tollerance and if none was given then use
            # default for this type
            this_tollerance = tollerance
            if this_tollerance is None:
                d_type = types.np_to_tfmin(actual_outputs[idx].dtype)
                if d_type == types.TenDType.FLOAT64:
                    this_tollerance = 1e-10
                elif d_type == types.TenDType.FLOAT32:
                    this_tollerance = 1e-5
                else:
                    this_tollerance = 0

            errors = np.absolute(expected_output - actual_outputs[idx])
            if np.amax(errors) > this_tollerance:
                correct = False
                print("Output %d [%s] output differs by %e (tollerance %e)" % (
                  idx,
                  self.graph.get_outputs()[idx].label,
                  np.amax(errors),
                  this_tollerance
                ))
                print("output [%d]" % idx)
                print("Expected:\n%s" % expected_output)
                print("Generated:\n%s\n------" % actual_outputs[idx])

        if correct:
            print("Verification Passed.")

        result = VerificationResult(correct)
        return result
