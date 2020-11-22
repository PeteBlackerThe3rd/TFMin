#!/usr/bin/env python -W ignore::DeprecationWarning
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

    A simple example of TFMin C++ code generation using a MNIST classifier
    made up of dense layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="unclosed.*<ssl.SSLSocket.*>")
warnings.simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf_min
from examples.shared import layers as layer_hlprs


def model(input_tensor, dtype=tf.float32, quant_settings=None):

    layer1 = layer_hlprs.DenseLayer(
        input_tensor,
        output_dim=300,
        layer_name="Layer1",
        act=tf.nn.relu,
        dtype=dtype,
        quant_settings=layer_hlprs.BaseLayer.get_quant(quant_settings, 0)
    )

    layer2 = layer_hlprs.DenseLayer(
        layer1.output,
        output_dim=10,
        layer_name="Layer2",
        act=tf.identity,
        dtype=dtype,
        quant_settings=layer_hlprs.BaseLayer.get_quant(quant_settings, 1)
    )

    return [layer1, layer2]


def train(args):

  #
  # Download the MNIST dataset used for training and evaluation.
  #
  print("Getting training data")
  mnist = input_data.read_data_sets(args.data_dir, fake_data=False)
  print("Done")

  #
  # Create simple two layer fully connected network.
  # Setup cross entropy for training and classification accuracy.
  #
  tf.reset_default_graph()
  sess = tf.Session()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)
  layers = model(x)
  y = layers[-1].output
  print("Built model.")

  training = layer_hlprs.TrainingCrossEntropy(y,
                                              y_,
                                              tf.train.AdamOptimizer,
                                              args.learning_rate)
  accuracy = layer_hlprs.ClassificationAccuracy(y, y_)
  tf.global_variables_initializer().run(session=sess)
  print("Added training operations to model.")

  #
  # Train the model using the downloaded MNIST dataset, and display the
  # accuracy which is achieved.
  #
  def feed_dict(train_d):
    """ Make feed_dict: maps data onto input tensors """
    if train_d:
      xs, ys = mnist.train.next_batch(100, False)
    else:
      xs, ys = mnist.test.images.astype(np.float32),\
               mnist.test.labels.astype(np.uint8)

    return {x: xs, y_: ys}

  layer_hlprs.train_model(sess,
                          training.train_step,
                          feed_dict,
                          accuracy.accuracy,
                          args.log_dir,
                          quality_label="Accuracy",
                          count=args.max_steps,
                          training_label="Training MNIST Classifier")

  #
  # Everything up to this point has been a conventional Tensorflow work flow.
  # Here is where we introduce the TFMin framework and use it to generate
  # an ANSI C implementation of this model. This C code will then be built
  # and executed and verification data passed into it to confirm that it
  # produces the same results as the original TF model.
  #

  # Command generates a tf_min.Graph object which describes the model
  # stored in the current TF session.
  graph = tf_min.graph_from_tf_sess(sess, outputs=[layers[-1].output])

  # Before this code can be generated for this graph the sequence of its
  # operations must be defined, and the location of any intermediate
  # tensor buffers pre-allocated.
  # A pre-defined pipeline of graph translators is invoked here to perform
  # both these tasks with a single command.
  # The GreedyHeap pipeline includes the SequenceOps GraphTranslator and the
  # HeapSmartOrder buffer pre-allocator.
  tf_min.Pipeline(builtin="GreedyHeap")(graph)

  # Our graph is now ready to export to C code, but before we do lets
  # export it's topology as a SVG diagram so we can confirm it has been
  # imported correctly.
  svg_writer = tf_min.SVGWriter(graph)
  svg_writer.write("mnist_dense_graph.svg")
  print("Done.")

  # verify the output of this graph matches tensorflow when it is
  # exported, built, and executed
  print("Testing verify output test harness")
  verifier = tf_min.GraphVerifyOutput(graph=graph,
                                      verbose=True)

  [expected_output] = sess.run(
    [layers[-1].output],
    feed_dict={x: mnist.test.images[:1]}
  )

  result = verifier.verify_model(input_tensors=[mnist.test.images[:1]],
                                 expected_output_tensors=[expected_output])

  if result.passed():
    print("Exported model passed verification.")
  else:
    print("Exported model failed verification.")

  sess.close()
  print("Complete")
  return result.success


def main(args):
  if tf.gfile.Exists(args.log_dir):
    tf.gfile.DeleteRecursively(args.log_dir)
  tf.gfile.MakeDirs(args.log_dir)
  train(args)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
    '--data_dir',
    type=str,
    default=os.path.join(
      os.getenv('TEST_TMPDIR', '/tmp'),
      'tensorflow/tf_min_mnist_dense_example/input_data'
    ),
    help='Directory for storing input data')
  parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(
      os.getenv('TEST_TMPDIR', '/tmp'),
      'tensorflow/tf_min_mnist_dense_example/logs/mnist_with_summaries'
    ),
    help='Summaries log directory')
  cmd_line_args, _ = parser.parse_known_args()

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    main(cmd_line_args)
