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
from .exceptions import TFMinVerificationExecutionFailed
from . import types


class VerificationResult:
  """

  """

  def __init__(self, success, failure_reason=""):
    """
    Create a VerificationResult object
    :param success: if this object is representing a successful result
    """
    self.success = success
    self.failure_reason = failure_reason

  def passed(self):
    return self.success

  def reason(self):
    return self.failure_reason

  def __bool__(self):
    return self.success


class GraphVerifyOutput(DeploymentRunner):
  """
  Object which generates unit_test versions of a graph, builds them and
  compares their output to a set of expected outputs.
  """

  def verify_model(self, input_tensors,
                   expected_output_tensors,
                   timeout=30.0,
                   tollerance=None,
                   verbose=False):
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
    :param tollerance: None or float, allowable error when comparing
                       expected and actual model output.
    :param verbose: Boolean, if true additional debugging info is printed
    :return: tf_min.VerificationResult object
    :raises: tf_min.TFMinVerificationExecutionFailed
             tf_min.TFMinDeploymentFailed
             tf_min.TFMinDeploymentExecutionFailed
             ValueError
    """

    # check the given number of input and output tensors matches the
    # graph.
    if len(input_tensors) != len(self.graph.get_inputs()):
      raise TFMinVerificationExecutionFailed(
          "Unexpected number of input tensors given. Expected %d, got %d" %
          (len(self.graph.get_inputs()), len(input_tensors))
      )
    if len(expected_output_tensors) != len(self.graph.get_outputs()):
      raise TFMinVerificationExecutionFailed(
          "Unexpected number of output tensors given. Expected %d, got %d" %
          (len(self.graph.get_outputs()), len(expected_output_tensors))
      )

    # check the data type of input and expected output ndarrays match
    # the data types of the graph
    for idx, input in enumerate(input_tensors):
      if not isinstance(input, np.ndarray):
        raise TFMinVerificationExecutionFailed(
          "Given input %d tensor is not a numpy.ndarray." % idx
        )
      given_type = types.np_to_tfmin(input.dtype)
      expected_type = self.graph.get_inputs()[idx].d_type
      if given_type != expected_type:
        raise TFMinVerificationExecutionFailed(
          "Given input %d tensor type missmatch, expected %s "
          "but was given %s!" % (idx, expected_type, given_type)
        )
      given_shape = list(input.shape)
      expected_shape = \
          self.graph.get_inputs()[idx].shape.get_shape(batch_size=1)
      if given_shape != expected_shape:
        raise TFMinVerificationExecutionFailed(
          "Given input %d tensor shape missmatch, expected %s but was "
          "given %s!" % (idx, str(expected_shape), str(given_shape))
        )

    for idx, output in enumerate(expected_output_tensors):
      if not isinstance(output, np.ndarray):
        raise TFMinVerificationExecutionFailed(
          "Given output %d tensor is not a numpy.ndarray." % idx
        )
      given_type = types.np_to_tfmin(output.dtype)
      expected_type = self.graph.get_outputs()[idx].d_type
      if given_type != expected_type:
        raise TFMinVerificationExecutionFailed(
          "Given output %d tensor type missmatch, expected %s "
          "but was given %s!" % (idx, expected_type, given_type)
        )
      given_shape = list(output.shape)
      expected_shape = \
          self.graph.get_outputs()[idx].shape.get_shape(batch_size=1)
      if given_shape != expected_shape:
        raise TFMinVerificationExecutionFailed(
          "Given output %d tensor shape missmatch, expected %s but was "
          "given %s!" % (idx, str(expected_shape), str(given_shape))
        )

    # Build and execute the model with the provided input tensors
    actual_outputs = self.execute_model(input_tensors,
                                        timeout=timeout)

    # Check if model output was within 'tollerance' of the given expected
    # output.
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
        reason_msg = "Output %d [%s] output differs by %e (tollerance %e)\n" % (
            idx,
            self.graph.get_outputs()[idx].label,
            np.amax(errors),
            this_tollerance
          )
        reason_msg += "output [%d]\n" % idx
        reason_msg += "Expected:\n%s\n" % expected_output
        reason_msg += "Generated:\n%s\n------\n" % actual_outputs[idx]

        if verbose:
          print(reason_msg)

        return VerificationResult(False, reason_msg)

    if verbose:
      print("Verification Passed.")

    result = VerificationResult(True)
    return result
