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

    A simple example of TFMin C++ code generation using an recurrent MNIST
    classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf_min import exporter as tfm_ex
import tf_min.progress_bar as pb

flags_g = None


# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def train(flags):

    print("Getting training data")
    mnist = input_data.read_data_sets(flags.data_dir, fake_data=False)
    print("Done")

    tf.reset_default_graph()
    sess = tf.Session()

    model_batch_size = 1  # or None
    model_tensor_batch_size = 1  # or -1
    training_batch_size = 1  # same as above if defined or anything you
    # want if a variable batch is defined

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [model_batch_size, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [model_batch_size], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [model_tensor_batch_size, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    """
    ----------------------------------------------------------------------------
    Create a simple RNN which sequentially processes a row of image
    data at a time
    ----------------------------------------------------------------------------
    """

    input_img = tf.reshape(x, [28, 28], "img_input")

    rnn_input = []

    for s in range(28):
        rnn_input += [tf.slice(input_img,
                               [s, 0],
                               [1, 28],
                               "input_seq_%02d" % s)]

    rnn_cell = tf.nn.rnn_cell.LSTMCell(28)

    rnn_outputs, rnn_states = tf.nn.static_rnn(rnn_cell,
                                               rnn_input,
                                               dtype=rnn_input[0].dtype)

    rnn_output = rnn_outputs[-1]

    y = nn_layer(rnn_output, 28, 10, 'fconn', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                labels=y_, logits=y)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(flags.learning_rate).minimize(
            cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(flags.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(flags.log_dir + '/test')
    tf.global_variables_initializer().run(session=sess)

    """
    ----------------------------------------------------------------------------
    Train the model and write summaries.
    Every 10th step, measure test-set accuracy, and write test summaries
    All other steps, run train_step on training data, & add training summaries
    ----------------------------------------------------------------------------
    """
    def feed_dict(train_d):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train_d:
            xs, ys = mnist.train.next_batch(training_batch_size, False)
        else:
            xs, ys = mnist.test.images[0].reshape([1, 784]),\
                     mnist.test.labels[0].reshape([1])
        return {x: xs, y_: ys}

    acc = 1.0
    for i in range(flags.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            pb.update_progress_bar(i / float(flags.max_steps),
                                   pre_msg=' Training MNIST Classifier',
                                   post_msg='Accuracy is %s' % acc, size=40)
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    pb.update_progress_bar(1.0,
                           pre_msg=' Training MNIST Classifier',
                           post_msg='Accuracy is %s' % acc, size=40,
                           c_return='\n')
    train_writer.close()
    test_writer.close()

    """
    ----------------------------------------------------------------------------
    Export Inference model to c++ code using the TFMin library
    ----------------------------------------------------------------------------
    """
    print("Using TFMin library to export minimal C++"
          "implimentation of this TensorFlow graph.")
    c_exporter = tfm_ex.Exporter(sess, ['fconn/activation:0'])

    # display the sub-set of the flow-graph being exported.
    c_exporter.print_graph()

    # extract the input tensor from the first row of training data to use for
    # validation
    validation_input = mnist.test.images[:1]

    # get the path of this script file. This is done so this works
    # correctly when executed stand alone and when executed as module during
    # testing.
    path_of_example_script = os.path.dirname(os.path.realpath(__file__))

    # generate the following c++ code encapsulating this inference model
    #
    # tfmin_generated/rnn_model.cpp
    # tfmin_generated/rnn_model.h
    # tfmin_generated/rnn_model_data.h
    #
    res = c_exporter.generate(path_of_example_script +
                              "/tfmin_generated/rnn_model",
                              "RNNModel",
                              validation_inputs={"input/"
                                                 "x-input": validation_input},
                              validation_type='Full',
                              timing=True,
                              layout='RowMajor')
    print("TFMin export Complete")

    return res


def main(_):
    if tf.gfile.Exists(flags_g.log_dir):
        tf.gfile.DeleteRecursively(flags_g.log_dir)
    tf.gfile.MakeDirs(flags_g.log_dir)
    train(flags=flags_g)


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
                             'tensorflow/mnist/logs/tfmin_rnn_mnist_example'),
        help='Summaries log directory')
    flags_g, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
