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

    Module of objects which create different types of Neural Network layers as
    well as optimisizer, measurement and training components.

    These layers also support automatic quantization of weights and the
    generation of equivalent fixed point versions. Automating the process of
    converting a model trained with floating point calculations into a
    model using fixed point calculations.
"""
import sys
import numpy as np
import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from . import progress_bar as pb


class BaseLayer:

    def __init__(self):
        self.layer_type = "Base"

    def get_quantised_weights(self, input_radix_point, dtype, sess,
                              test_data_dict={}, fixed_radix=None):
        print("Error calling get_quantised_weights on BaseLayer "
              "'abstract' class!")
        return {}

    @staticmethod
    def weight_variable(shape=[], dtype=tf.float32, initial=None):
        """Create a weight variable with appropriate initialization."""
        if initial is None:
            initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial_value=initial, dtype=dtype)

    @staticmethod
    def bias_variable(shape=[], dtype=tf.float32, initial=None):
        """Create a bias variable with appropriate initialization."""
        if initial is None:
            initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value=initial, dtype=dtype)

    @staticmethod
    def variable_summaries(var):
        """Attach  summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def optimal_radix(min_value, max_value, dtype):

        if dtype == tf.int8:
            bits = 8
        elif dtype == tf.int16:
            bits = 16
        elif dtype == tf.int32:
            bits = 32
        else:
            print("Error unsupported type (%s) for Layer.optimal_radix" %
                  str(type(dtype)))
            return 0

        # Because the range of Two's compliment signed numbers is 1
        # smaller for negative values that it is for positive values we need
        # to calculate the maximum radix shift for positive and negative
        # values differently. Then we pick the smaller of the two, this way
        #  we garantee making the best use of the fixed point range
        negative_radix = math.log2(pow(2, bits-1)-1) - math.log2(abs(min_value))
        positive_radix = (bits-1) - math.log2(abs(max_value))

        return int(min(negative_radix, positive_radix))

    @staticmethod
    def get_quant(quant_settings, index):
        """
        Shorthand method to get a particular layers quantified
        settings from a list, or null if there are none (i.e. it's
        not a list at all)

        :param quant_settings: The list of settings or None
        :param index: Index of the layer requested
        :return: None if the input is None or the specified layer otherwise.
        """

        if type(quant_settings) is list and len(quant_settings) > index:
            return quant_settings[index]
        else:
            return None

    @staticmethod
    def get_quantised_layer_weights(layers,
                                    input_radix_point,
                                    dtype,
                                    sess,
                                    test_data_dict={},
                                    fixed_radix=None):

        quantised_layers = []
        current_radix = input_radix_point

        for i, l in enumerate(layers):

            print("layer type is [%s]", str(type(l)))

            print("Quantising %s layer %s, with with type %s" %
                  (l.layer_type,
                   l.layer_name,
                   str(dtype)))

            if not isinstance(l, BaseLayer):
                print("Error in 'get_quantised_layer_weights' layer %d is not "
                      "an object derived from BaseLayer!" % i)

            quantised_layers += [
              l.get_quantised_weights(current_radix,
                                      dtype,
                                      sess,
                                      test_data_dict=test_data_dict,
                                      fixed_radix=fixed_radix)
            ]
            current_radix = quantised_layers[-1]['RadixOutput']

        return quantised_layers


class DenseLayer(BaseLayer):

    def __init__(self,
                 input_tensor,
                 output_dim,
                 layer_name,
                 act=tf.nn.relu,
                 dtype=tf.float32,
                 quant_settings=None):

        super().__init__()

        self.layer_type = "Dense"
        self.layer_name = layer_name
        self.quant_settings = quant_settings
        self.output_radix = 0

        # if the input tensor isn't rank 1 then flatten it
        if input_tensor.shape.ndims > 1:

            size = 1
            for i in range(1, input_tensor.shape.ndims):
                size *= input_tensor.shape.dims[i].value

            batch_dim = input_tensor.shape.dims[0].value
            if batch_dim is None:
                batch_dim = -1

            input_tensor = tf.reshape(input_tensor, [batch_dim, size])

        input_size = input_tensor.shape.dims[1].value

        print("Generating a dense layer, input[%d] output[%d] type[%s]" %
              (input_size, output_dim, str(dtype)))

        if ((dtype == tf.int8 or dtype == tf.int16 or dtype == tf.int32) and
                quant_settings is not None):

            if dtype == tf.int8:
                calc_type = tf.int32
            elif dtype == tf.int16:
                calc_type = tf.int32
            else:
                calc_type = tf.int64

            self.output_radix = quant_settings['RadixOutput']

            post_mm_div = int(math.pow(2, quant_settings['PostMMShift']))

            print("setting up descretised layer [%s] type %s reduction %d" %
                  (layer_name,
                   str(calc_type),
                   quant_settings['PostMMShift']))

            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    self.weights = BaseLayer.weight_variable(
                      initial=quant_settings['Weights'], dtype=dtype
                    )
                with tf.name_scope('biases'):
                    self.biases = BaseLayer.bias_variable(
                      initial=quant_settings['Biases'], dtype=dtype
                    )
                with tf.name_scope('Wx_plus_b'):
                    input_calc_type = tf.cast(input_tensor, dtype=calc_type)
                    self.mat_mul = tf.matmul(tf.cast(input_calc_type,
                                                     dtype=calc_type),
                                             tf.cast(self.weights,
                                                     dtype=calc_type))
                    preactivate = tf.cast(tf.div(self.mat_mul, post_mm_div),
                                          dtype) + self.biases
                self.output = act(preactivate, name='activation')

        # if this is any other type build a regular float layer
        else:
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    self.weights = BaseLayer.weight_variable(
                      shape=[input_size, output_dim], dtype=dtype
                    )
                    BaseLayer.variable_summaries(self.weights)
                with tf.name_scope('biases'):
                    self.biases = BaseLayer.bias_variable(
                      shape=[output_dim], dtype=dtype
                    )
                    BaseLayer.variable_summaries(self.biases)
                with tf.name_scope('Wx_plus_b'):
                    self.mat_mul = tf.matmul(input_tensor, self.weights)
                    preactivate = self.mat_mul + self.biases
                    tf.summary.histogram('pre_activations', preactivate)
                self.output = act(preactivate, name='activation')
                tf.summary.histogram('activations', self.output)

    def get_quantised_weights(self, input_radix_point, dtype, sess,
                              test_data_dict={}, fixed_radix=None):

        quant_weights = {'LayerType': self.layer_type,
                         'RadixIn': input_radix_point}

        print("Quantising %s layer %s." % (self.layer_type, self.layer_name))

        # the radix point of the weights for matrix multiplication is
        # independant of the input radix point
        matmul_weights = np.array(sess.run([self.weights]))
        matmul_weights = matmul_weights.reshape(matmul_weights.shape[1:])

        if fixed_radix is not None:
            quant_weights['RadixWeights'] = fixed_radix
        else:
            quant_weights['RadixWeights'] = BaseLayer.optimal_radix(
              np.min(matmul_weights),
              np.max(matmul_weights),
              dtype
            )

        mm_min = np.min(matmul_weights)
        mm_max = np.max(matmul_weights)

        scaled_weights = matmul_weights * (2 ** quant_weights['RadixWeights'])
        quant_weights['Weights'] = scaled_weights.astype(dtype.as_numpy_dtype())

        print("Weights radix [%d], multiplier [%f]" %
              (quant_weights['RadixWeights'],
               math.pow(2, quant_weights['RadixWeights'])))

        print("Quantised dense layer weights original range [%f - %f] "
              "quant range [%d - %d] scaled [%f - %f]" %
              (mm_min, mm_max,
               np.min(quant_weights['Weights']),
               np.max(quant_weights['Weights']),
               np.min(scaled_weights),
               np.max(scaled_weights)))

        # calculate the radix point of the output, this needs to be
        # optimised for all the values of the biases, the output of
        # mat_mul, and the resulting add
        mat_mul_result, biases, add_result = sess.run([self.mat_mul,
                                                       self.biases,
                                                       self.output],
                                                      test_data_dict)

        if fixed_radix is not None:
            quant_weights['RadixOutput'] = fixed_radix
        else:
            mat_mul_min = np.min(np.array(mat_mul_result))
            biases_min = np.min(np.array(biases))
            add_min = np.min(np.array(add_result))
            mat_mul_max = np.max(np.array(mat_mul_result))
            biases_max = np.max(np.array(biases))
            add_max = np.max(np.array(add_result))
            output_min = min(mat_mul_min, biases_min, add_min)
            output_max = min(mat_mul_max, biases_max, add_max)
            quant_weights['RadixOutput'] = BaseLayer.optimal_radix(output_min,
                                                                   output_max,
                                                                   dtype)

        quant_weights['Biases'] = (
                np.array(biases) * (2**quant_weights['RadixOutput'])
        ).astype(dtype.as_numpy_dtype())

        quant_weights['PostMMShift'] = \
          (input_radix_point +
           quant_weights['RadixWeights']) - quant_weights['RadixOutput']

        return quant_weights


class Conv2DLayer(BaseLayer):

    def __init__(self,
                 input_tensor,
                 filter_size=3,
                 filter_count=16,
                 layer_name="",
                 stride=2,
                 padding='VALID',
                 act=tf.nn.leaky_relu,
                 pooling=2,
                 pool_stride=2,
                 dtype=tf.float32,
                 quant_settings=None):
            """Reusable code for making a convolutional neural net layer.
            It does a convolution, bias add, and then uses an activation function
            (ReLu defauly) nonlinearize followed by an optional max-pooling layer.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """

            super().__init__()
            self.layer_type = "Conv2D"
            self.layer_name = layer_name

            with tf.name_scope(layer_name):
                input_channels = int(input_tensor.shape[3])

                # setup and generate initial weights for the filters
                # create weights summaries
                with tf.name_scope('filter_weights'):
                    self.filters = BaseLayer.weight_variable([filter_size, filter_size, input_channels, filter_count])
                    BaseLayer.variable_summaries(self.filters)

                weights_count = filter_size * filter_size * input_channels * filter_count

                # cryptic code to generate a summary image of the filters
                """if input_channels == 1:
                    t = tf.reshape(filters, [filter_size, filter_size, filter_count])
                    filter_size += 2
                    t = tf.image.resize_image_with_crop_or_pad(t, filter_size, filter_size)

                    # if the filter count is a square number
                    filter_count_power = m.log(filter_count, 2)
                    if filter_count_power % 2 == 0:
                        f_rows = f_cols = int(m.pow(2, (filter_count_power / 2)))
                    else:
                        f_rows = int(m.pow(2, (filter_count_power + 1) / 2))
                        f_cols = int(m.pow(2, (filter_count_power - 1) / 2))

                    t = tf.reshape(t, (filter_size, filter_size, f_rows, f_cols))
                    t = tf.transpose(t, (2, 0, 3, 1))
                    filters_image = tf.reshape(t, (1, f_rows * filter_size, f_cols * filter_size, 1))
                    tf.summary.image('conv1_filters', filters_image, 1)"""

                strides = [1, stride, stride, 1]

                # create convotutional layer
                self.conv2d_result = tf.nn.convolution(
                    input=input_tensor,
                    filter=self.filters,
                    strides=strides[1:3],
                    padding=padding,
                    name="convolution")

                # create convotutional layer
                """self.conv2d_result = tf.nn.conv2Dd(
                    input=input_tensor,
                    filter=self.filters,
                    strides=strides,
                    padding=padding,
                    name="convolution")"""

                # add activation function
                activations = act(self.conv2d_result, name='activations')

                if pooling == 0:
                    print('produced convolutional layer [%s] with size : %s, weights %d' %
                          (layer_name, activations.shape, weights_count))
                    self.output = activations  # return activations
                else:
                    max_pooling = tf.nn.max_pool(value=activations,
                                                 ksize=[1, pooling, pooling, 1],
                                                 strides=[1, pool_stride, pool_stride, 1],
                                                 padding=padding,
                                                 name='pooling')

                    print('produced convolutional layer [%s] with size : %s, weights %d' %
                          (layer_name, max_pooling.shape, weights_count))
                    self.output = max_pooling  # return max_pooling

    def get_quantised_weights(self, input_radix_point, dtype, sess, test_data_dict={}, fixed_radix=None):

        quant_weights = {'LayerType': self.layer_type, 'RadixIn': input_radix_point}

        print("Quantising %s layer %s." % (self.layer_type, self.layer_name))

        # the radix point of the weights for matrix multiplication is independant of the input radix point
        filter_weights = np.array(sess.run([self.filters]))
        filter_weights = filter_weights.reshape(filter_weights.shape[1:])

        if fixed_radix is not None:
            quant_weights['RadixWeights'] = fixed_radix
        else:
            quant_weights['RadixWeights'] = BaseLayer.optimal_radix(np.min(filter_weights),
                                                                    np.max(filter_weights),
                                                                    dtype)

        mm_min = np.min(filter_weights)
        mm_max = np.max(filter_weights)

        scaled_weights = filter_weights * math.pow(2, quant_weights['RadixWeights'])
        quant_weights['Weights'] = scaled_weights.astype(dtype.as_numpy_dtype())

        print("Weights radix [%d], multiplier [%f]" %
              (quant_weights['RadixWeights'],
               math.pow(2, quant_weights['RadixWeights'])))

        print("Quantised %s layer weights original range [%f - %f] "
              "quant range [%d - %d] scaled [%f - %f]" %
              (self.layer_type,
               mm_min, mm_max,
               np.min(quant_weights['Weights']),
               np.max(quant_weights['Weights']),
               np.min(scaled_weights), np.max(scaled_weights)))

        # calculate the radix point of the output, this needs to be
        # optimised for the output of the covolution operation
        conv2d_result = sess.run([self.conv2d_result], test_data_dict)

        if fixed_radix is not None:
            quant_weights['RadixOutput'] = fixed_radix
        else:
            output_min = np.min(np.array(conv2d_result))
            output_max= np.max(np.array(conv2d_result))
            quant_weights['RadixOutput'] = BaseLayer.optimal_radix(output_min,
                                                                   output_max,
                                                                   dtype)

        quant_weights['PostMMShift'] =\
            (input_radix_point + quant_weights['RadixWeights']) - quant_weights['RadixOutput']

        return quant_weights

class TrainingCrossEntropy:

    def __init__(self,
                 model_lables,
                 training_labels,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.001):

        with tf.name_scope('training'):

            self.loss = tf.losses.sparse_softmax_cross_entropy(
              labels=training_labels, logits=model_lables
            )
            tf.summary.scalar('cross_entropy_loss', self.loss)

            self.train_step = optimizer(learning_rate).minimize(self.loss)


class ClassificationAccuracy:

    def __init__(self, model_output, training):

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(model_output, 1),
                                          training, name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                   tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', self.accuracy)


def train_model(sess,
                train_step,
                feed_fn,
                quality_metric,
                log_dir,
                quality_label="Quality Metric",
                count=1000,
                training_label="Training Model"):

    # setup training summaries and writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    quality = 0.0
    for i in range(count):

        # Train the model with a batch and update the training summary writer
        summary, _ = sess.run([merged, train_step], feed_dict=feed_fn(True))
        train_writer.add_summary(summary, i)

        # every 10th step calculate the accurary and update
        # summary and progress bar
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, quality = sess.run([merged, quality_metric],
                                        feed_dict=feed_fn(False))
            test_writer.add_summary(summary, i)
            pb.update_progress_bar(i / float(count),
                                   pre_msg=" " + training_label,
                                   post_msg='%s is %s' % (quality_label,
                                                          quality),
                                   size=40)

    # Add final progress bar update at 100% and close summary writers
    pb.update_progress_bar(1.0,
                           pre_msg=" " + training_label,
                           post_msg='%s is %s' % (quality_label, quality),
                           size=40,
                           c_return='\n')
    train_writer.close()
    test_writer.close()
