
# ==============================================================================
"""A
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import copy
import numpy as np
import math as m
import tensorflow as tf
from tf_min import exporter as tfm_ex


ranges = {'input_shape': [[None, 50, 50, 1], [None, 50, 50, 3]],
          'filter_shape': [[1, 3, 8], [3, 1, 8], [3, 3, 8], [2, 2, 1]],
          'strides': [[1, 1], [2, 1], [1, 2], [2, 2], [3, 3]],
          'padding': ['VALID', 'SAME'],
          'dtype': [tf.float32],
          'layout': ['RowMajor']}


def get_test_list(test_ranges):

  key = list(test_ranges.keys())[0]
  test_range = test_ranges[key]
  remaining_ranges = copy.copy(test_ranges)
  del remaining_ranges[key]

  # base case where we have reached the last range in the dictionary
  if len(remaining_ranges) == 0:
    test_list = []
    for value in test_range:
      test_list += [{key: value}]
    return test_list
  else:  # std case where we multiply the test lists of existing ranges
    inner_test_list = get_test_list(remaining_ranges)
    test_list = []
    for value in test_range:
      additional_test_list = copy.deepcopy(inner_test_list)
      for additional_test_item in additional_test_list:
        additional_test_item[key] = value
      test_list += additional_test_list
    return test_list


def create_and_export_model(test_setup):

  # --- reset and setup new graph. (Safe to run after a prevision session)
  tf.reset_default_graph()
  sess = tf.Session()

  # --- Create minimal test graph
  # input_shape = [None, input_width, input_height, input_channels]
  input = tf.placeholder(test_setup['dtype'],
                         test_setup['input_shape'],
                         name='input')

  # filter_shape = [filter_width, filter_height, input_channels, filter_count]
  filter_shape = test_setup['filter_shape'][0:2] +\
                 [test_setup['input_shape'][3]] +\
                 [test_setup['filter_shape'][2]]
  initial = tf.truncated_normal(filter_shape, stddev=1.0)
  filter = tf.Variable(initial_value=initial,
                       dtype=test_setup['dtype'],
                       name='filter')

  output = tf.nn.convolution(
    input=input,
    filter=filter,
    strides=test_setup['strides'],
    padding=test_setup['padding'],
    name="convolution")

  tf.global_variables_initializer().run(session=sess)

  # --- Export test model with validation via TFMin
  validation_input = np.random.rand(1,
                                    test_setup['input_shape'][1],
                                    test_setup['input_shape'][2],
                                    test_setup['input_shape'][3])
  c_exporter = tfm_ex.Exporter(sess, [output])
  path_of_example_script = os.path.dirname(os.path.realpath(__file__))

  # generate the following c++ code encapsulating this inference model
  res = c_exporter.generate(path_of_example_script +
                            "/tmp/test_source_conv2d",
                            "Conv2dTest",
                            layout=test_setup['layout'],
                            validation_inputs={"input": validation_input},
                            validation_type='Full',
                            timing=False)

  assert res and "Model failed to export without errors."
  print("------ Test Export Passed ------")


def build_model(test_setup):

  create_and_export_model(test_setup)

  script_path = os.path.dirname(os.path.realpath(__file__))
  make_cmd = "make -C " + str(script_path)

  exit_code = subprocess.call([make_cmd], shell=True)
  assert exit_code == 0 and "Model failed to build without errors."
  print("------ Test Build Passed ------")


def test_conv2d():

  test_list = get_test_list(ranges)
  print("Found a test list of %d tests" % len(test_list))
  script_path = os.path.dirname(os.path.realpath(__file__))
  for idx, test in enumerate(test_list):
    print("------ Starting test (%d/%d) [%s] -------" %
          (idx+1,
           len(test_list),
           str(test)))

    build_model(test)
    exit_code = subprocess.call([script_path + "/tmp/test_binary"])

    assert exit_code == 0 and "Model failed to validate without errors."
    print("------ Test Validation Passed -------\n")

  print("All tests passed.")


def main():
  test_conv2d()


if __name__ == '__main__':
  main()
