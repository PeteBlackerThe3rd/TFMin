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

    A simple example of TFMin Ansi C code generation using an MNIST classifier
    made up of two dense layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

import tf_min

from examples.shared import layers as layer_hlprs


def model(input_tensor):
  """
  Function used to create our MNIST classification model
  :param input_tensor: tf.Placeholder input object.
  :return: List of layer objects.
  """

  layer1 = layer_hlprs.DenseLayer(
      input_tensor,
      output_dim=300,
      layer_name="Layer1",
      act=tf.nn.relu,
      dtype=tf.float32
  )

  layer2 = layer_hlprs.DenseLayer(
      layer1.output,
      output_dim=10,
      layer_name="Layer2",
      act=tf.identity,
      dtype=tf.float32
  )

  return [layer1, layer2]


def main(args):
  #
  # Download the MNIST dataset used for training and evaluation.
  #
  print("Getting training data.")
  mnist = input_data.read_data_sets(args.data_dir, fake_data=False)
  print("Done. \n\n")

  #
  # Create simple two layer fully connected network.
  # Setup cross entropy for training and classification accuracy.
  #
  print("Building MNIST Classification Model.")
  tf.compat.v1.reset_default_graph()
  sess = tf.compat.v1.Session()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.compat.v1.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.compat.v1.placeholder(tf.int64, [None], name='y-input')

  layers = model(x)
  y = layers[-1].output

  training = layer_hlprs.TrainingCrossEntropy(y,
                                              y_,
                                              tf.compat.v1.train.AdamOptimizer,
                                              args.learning_rate)
  accuracy = layer_hlprs.ClassificationAccuracy(y, y_)
  tf.compat.v1.global_variables_initializer().run(session=sess)
  print("Added training operations to model.")
  print("Built model. \n\n")

  #
  # Train the model using the downloaded MNIST dataset, and display the
  # accuracy which is achieved.
  #
  def feed_dict(train_d):
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
                          quality_label="Accuracy",
                          count=args.max_steps,
                          training_label="Training MNIST Classifier")
  print("Training Complete. \n\n")

  #
  # Everything up to this point has been a conventional Tensorflow work flow.
  # Here is where we introduce the TFMin framework and use it to generate
  # an ANSI C implementation of this model. This C code will then be built
  # and executed and verification data passed into it to confirm that it
  # produces the same results as the original TF model.
  #

  # Command generates a tf_min.Graph object which describes the model
  # stored in the current TF session.
  print("Importing model from tensorflow session to TFMin Graph")
  graph = tf_min.graph_from_tf_sess(sess, outputs=[layers[-1].output])

  # Before this code can be generated for this graph the sequence of its
  # operations must be defined, and the location of any intermediate
  # tensor buffers pre-allocated.
  # A pre-defined pipeline of graph translators is invoked here to perform
  # both these tasks with a single command.
  # The GreedyHeap pipeline includes the SequenceOps GraphTranslator and the
  # HeapSmartOrder buffer pre-allocator.
  tf_min.Pipeline(builtin="GreedyHeap")(graph)
  print("Import Complete. \n\n")

  # Our graph is now ready to export to C code, but before we do lets
  # export it's topology as a SVG diagram so we can confirm it has been
  # imported correctly.
  svg_writer = tf_min.SVGWriter(graph)
  svg_writer.write("mnist_dense_graph.svg")

  # Use the GraphVerifyOutput class to verify the output of the c implementation
  # of this graph matches tensorflow when it is exported, built, and executed
  print("Testing generated C implementation produces results which "
        "Match those from Tensorflow.")
  verifier = tf_min.GraphVerifyOutput(graph=graph,
                                      verbose=False)

  # use our tensorflow session one last time to get the expected output
  # for the first input image.
  [expected_output] = sess.run(
    [layers[-1].output],
    feed_dict={x: mnist.test.images[:1]}
  )
  sess.close()

  # Run the verification, this call generates the C code and a main entrypoint
  # builds and executes it. The test input is then fed in and the output
  # tensor value is read out, and compared the expected result.
  # All temporary files created during this process are removed before
  # this function returns.
  result = verifier.verify_model(input_tensors=[mnist.test.images[:1]],
                                 expected_output_tensors=[expected_output],
                                 tollerance=1e-5)

  if result:
    print("Exported model passed verification. \n\n")
  else:
    print("Exported model failed verification.")
    print(result.reason())
    exit(-1)

  # Finally now we know that the ansi-c implementation of this model we
  # can generate it again. Here we specify the filename and identifier
  # prefixes we need along with the coding style and byte order of the
  # target platform.
  # In this case the code will be placed in the `ansi_c_model` directory
  # where there is already a main.c and Makefile setup to build this code
  # which this call generates.
  print("Generating final ansi-c version of this model.")
  c_generator = tf_min.CodeGenerator(graph=graph,
                                     base_name="mnist_",  # source file prefix
                                     prefix="mnist_",  # c identifier prefix
                                     path="ansi_c_model",  # output dir
                                     clang_format='Google',
                                     byte_order="@", batch_size=1
                                     )
  c_generator()
  print("Code generation complete.\n")
  return result.success


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
  cmd_line_args, _ = parser.parse_known_args()

  main(cmd_line_args)
