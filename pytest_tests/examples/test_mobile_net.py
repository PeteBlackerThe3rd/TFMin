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

# ugly hack to get the module imports used in the MobileNet example to
# work with pytest!
sys.path.append(script_path + '/examples/mobile_net')

# print("sys.path is:")
# for path in sys.path:
#  print("Path [ %s]" % path)

import mobile_net.mobile_net as example


def test_mobile_net():
  """
  This verifies that the mobile net example executes and
  successfully exports the TFMin generated code without failing.
  :return: True if successful False if not.
  """

  print("About to export mobile net model.")
  test_result = example.build_mobile_net_model('examples/mobile_net/config/test.json')
  print("----")

  assert test_result

  return test_result

def main():

  if example.build_mobile_net_model('examples/mobile_net/config/test.json'):
    exit(0)
  else:
    exit(1)

if __name__ == '__main__':
  main()