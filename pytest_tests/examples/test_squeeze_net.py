# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

script_path = os.getcwd()

# Add examples folder to python path just for these test scripts
sys.path.append(script_path + '/examples')

import squeeze_net.squeezenet_tf as example


def test_squeeze_net():
  """
  This verifies that the squeeze net example executes and
  successfully exports the TFMin generated code without failing.
  :return: True if successful False if not.
  """

  # Setup command line flags for this test
  class Options:
    def __init__(self):
      self.input = "examples/squeeze_net/poodle.jpeg"

  options = Options()

  print("About to export squeeze net model.")
  test_result = example.export_sqz_net(options)

  assert test_result

  return test_result


def main():

  if test_squeeze_net():
    exit(0)
  else:
    exit(1)


if __name__ == '__main__':
  main()