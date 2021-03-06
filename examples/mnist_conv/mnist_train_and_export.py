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
    made up of an initial convolution layer and two dense layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import math as m

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf_min import exporter as tfm_ex
import tf_min.layers as tfm_layers

FLAGS_g = None


def model(input_tensor, dtype=tf.float32, quant_settings=None):

    layer1 = tfm_layers.Conv2DLayer(
        input_tensor,
        filter_size=5,
        filter_count=16,
        layer_name='Conv_1',
        pooling=3,
        stride=2,
        dtype=dtype,
        quant_settings=tfm_layers.BaseLayer.get_quant(quant_settings, 0)
    )

    layer2 = tfm_layers.DenseLayer(
        layer1.output,
        output_dim=300,
        layer_name="Dense_1",
        act=tf.nn.relu,
        dtype=dtype,
        quant_settings=tfm_layers.BaseLayer.get_quant(quant_settings, 1)
    )

    layer3 = tfm_layers.DenseLayer(
        layer2.output,
        output_dim=10,
        layer_name="Dense_2",
        act=tf.identity,
        dtype=dtype,
        quant_settings=tfm_layers.BaseLayer.get_quant(quant_settings, 2)
    )

    return [layer1, layer2, layer3]


def train(flags):

    print("Getting training data")
    mnist = input_data.read_data_sets(flags.data_dir, fake_data=False)
    print("Done")

    # sess = tf.InteractiveSession()
    tf.reset_default_graph()
    sess = tf.Session()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

    """-------------------------------------------------------------------------
    Create three layer network, one conv, two fully connected. Use cross 
    entropy to train.
    -------------------------------------------------------------------------"""
    layers = model(image_shaped_input)
    y = layers[-1].output

    training = tfm_layers.TrainingCrossEntropy(y, y_, tf.train.AdamOptimizer,
                                               flags.learning_rate)
    accuracy = tfm_layers.ClassificationAccuracy(y, y_)
    tf.global_variables_initializer().run(session=sess)

    """-------------------------------------------------------------------------
    Train the model, and also write summaries.
    Every 10th step, measure test-set accuracy, and write test summaries
    All other steps, run train_step on training data, & add training summaries
    -------------------------------------------------------------------------"""

    def feed_dict(train_d):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train_d:
            xs, ys = mnist.train.next_batch(100, False)
        else:
            xs, ys = mnist.test.images.astype(np.float32),\
                     mnist.test.labels.astype(np.uint8)

        return {x: xs, y_: ys}

    tfm_layers.train_model(sess,
                           training.train_step,
                           feed_dict,
                           accuracy.accuracy,
                           flags.log_dir,
                           quality_label="Accuracy",
                           count=flags.max_steps,
                           training_label="Training MNIST Classifier")

    """-------------------------------------------------------------------------
    Export Inference model to c++ code using the TFMin library
    -------------------------------------------------------------------------"""
    print("Using TFMin library to export inference C++ implimentation of this "
          "TensorFlow graph.")
    c_exporter = tfm_ex.Exporter(sess, [layers[-1].output])

    # display the sub-set of the flow-graph being exported.
    c_exporter.print_graph()

    # extract the input tensor from the first row of training data to use
    # for validation
    validation_input = mnist.test.images[:1]

    # get the path of this script file. This is done so this works
    # correctly when executed stand alone and when executed as module during
    # testing.
    path_of_example_script = os.path.dirname(os.path.realpath(__file__))

    # generate the following c++ code encapsulating this inference model
    #
    # tfmin_generated/mnist_model.cpp
    # tfmin_generated/mnist_model.h
    # tfmin_generated/mnist_model_data.h
    #
    res = c_exporter.generate(path_of_example_script +
                              "/tfmin_generated/mnist_model",
                              "MNISTModel",
                              layout='RowMajor',
                              validation_inputs={"input/x-input":
                                                 validation_input},
                              validation_type='Full',
                              timing=True)

    sess.close()
    print("Complete")
    return res


def main(_):
    if tf.gfile.Exists(flags_g.log_dir):
        tf.gfile.DeleteRecursively(flags_g.log_dir)
    tf.gfile.MakeDirs(flags_g.log_dir)
    train(flags_g)


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
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/input_data'),
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/mnist_with_summaries'),
        help='Summaries log directory')
    flags_g, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
