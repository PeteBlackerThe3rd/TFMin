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

    This module contains the DeploymentCompare object, this object
    takes two TFMin Graph objects and deploys them both using the same
    build settings. A set of example input data is then fed into both
    deployed graphs and the outputs compared to each other.
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
from tf_min import cpp_code_gen as c_gen
from tf_min import graph_c_gen
from tf_min import exceptions as exc
from tf_min.deployment_runner import DeploymentRunner


class ComparisonResult:
  """
  Object to store the results of comparing two graphs
  """

  def __init__(self, results_a, results_b):
    """
    Create results object
    :param results_a: List of numpy.ndarrays
    :param results_b: List of numpy.ndarrays
    """
    self.results_a = results_a
    self.results_b = results_b

  def max_error(self):
    """
    Method to return the maximum error between matching elements of the
    results of each graph
    :return: Scalar Maximum error
    """


class DeploymentCompare:
  """
  Object which deploys a two graphs and feeds the same data into each
  comparing the final outputs.
  """

  def __init__(self, graph_a, graph_b,
               compiler="gcc", flags=['-O3']):
    """
    Create a DeploymentCompare object which will compare the two graphs
    :param graph_a: First graph to build
    :param graph_b: Second graph to build
    :param compiler: String, the compiler command to use
    :param flags: List of string, additional compiler flags to use
    """
    self.runner_a = DeploymentRunner(graph_a, tmp_dir="tmp_comp_a",
                                     compiler=compiler, flags=flags)
    self.runner_b = DeploymentRunner(graph_b, tmp_dir="tmp_comp_b",
                                     compiler=compiler, flags=flags)

  def compare(self, inputs):
    """
    Method to run bot graphs with the given input and produce a
    ComparisonResult object
    :param inputs: List of nump.ndarray input tensors
    :return: ComparisonResult object
    """
    results_a = self.runner_a.execute_model(inputs)
    results_b = self.runner_b.execute_model(inputs)

    return ComparisonResult(results_a, results_b)
