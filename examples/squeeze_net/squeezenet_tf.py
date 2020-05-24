# Copyright (c) 2017 Andrey Voroshilov

import os
import tensorflow as tf
import numpy as np
import scipy.io
import time

from PIL import Image
from argparse import ArgumentParser
from tf_min import exporter as tfm_ex


def imread_resize(path):
    img_orig = scipy.misc.imread(path)
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)

    if len(img.shape) == 2:                     # grayscale
        img = np.dstack((img, img, img))

    if len(img.shape) == 3 and img.shape[2] == 4:   # RGB - Alpha
        img = img[:, :, 0:3]

    return img, img_orig.shape


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def get_dtype_np():
    return np.float32


def get_dtype_tf():
    return tf.float32

# SqueezeNet v1.1 (signature pool 1/3/5)
########################################


def load_net(data_path):
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist."
                     "(Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)

    # Converting to needed type
    conv_time = time.time()
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append(kernels.astype(get_dtype_np()))
            weights[name].append(bias.astype(get_dtype_np()))
    print("Converted network data(%s): %fs" %
          (get_dtype_np(), time.time() - conv_time))

    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    return weights, mean_pixel


def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel


def unprocess(image, mean_pixel):
    swap_img = np.array(image + mean_pixel)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out


def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return weights, biases


def fire_cluster(net, x, preloaded, cluster_name):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x,
                    weights, biases, padding='VALID')
    x = _act_layer(net, layer_name + '_actv', x)

    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_l = _conv_layer(net, layer_name + '_conv', x,
                      weights, biases, padding='VALID')
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_r = _conv_layer(net, layer_name + '_conv', x,
                      weights, biases, padding='SAME')
    x_r = _act_layer(net, layer_name + '_actv', x_r)

    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x

    return x


def net_preloaded(preloaded, input_image, pooling, needs_classifier=False):
    net = {}
    cr_time = time.time()

    x = tf.cast(input_image, get_dtype_tf())

    # Feature extractor
    #####################

    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases,
                    padding='VALID', stride=(2, 2))
    x = _act_layer(net, layer_name + '_actv', x)
    x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3),
                    stride=(2, 2), padding='VALID')

    # fire2 + fire3 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire2')
    # fire2_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire3')
    x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3),
                    stride=(2, 2), padding='VALID')

    # fire4 + fire5 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire4')
    # fire4_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire5')
    x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3),
                    stride=(2, 2), padding='VALID')

    # remainder (no pooling)
    x = fire_cluster(net, x, preloaded, cluster_name='fire6')
    # fire6_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire7')
    x = fire_cluster(net, x, preloaded, cluster_name='fire8')
    x = fire_cluster(net, x, preloaded, cluster_name='fire9')

    # Classifier
    #####################
    if needs_classifier:
        # Dropout [use value of 50% when training]
        # x = tf.nn.dropout(x, keep_prob)

        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases)
        x = _act_layer(net, layer_name + '_actv', x)

        # Global Average Pooling
        x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1),
                           strides=(1, 1, 1, 1), padding='VALID')
        net['classifier_pool'] = x

        x = tf.reshape(x, (1, 1000))

        x = tf.nn.softmax(x)
        net['classifier_actv'] = x

    print("Network instance created: %fs" % (time.time() - cr_time))

    return net


def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
    conv = tf.nn.conv2d(input, tf.constant(weights),
                        strides=(1, stride[0], stride[1], 1),
                        padding=padding)
    x = tf.nn.bias_add(conv, bias)
    net[name] = x
    return x


def _act_layer(net, name, input):
    x = tf.nn.relu(input)
    net[name] = x
    return x


def _pool_layer(net, name, input,
                pooling, size=(2, 2),
                stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1),
                           strides=(1, stride[0], stride[1], 1),
                           padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1),
                           strides=(1, stride[0], stride[1], 1),
                           padding=padding)
    net[name] = x
    return x


def build_parser():
    ps = ArgumentParser()
    ps.add_argument('--in',
                    dest='input',
                    help='input file',
                    metavar='INPUT',
                    required=True)
    return ps


def export_sqz_net(options):

    # Loading image
    img_content, orig_shape = imread_resize(options.input)
    img_content_shape = (tf.Dimension(None),) + img_content.shape

    # get the path of this script file. This is done so this works
    # correctly when executed stand alone and when executed as module during
    # testing.
    path_of_example_script = os.path.dirname(os.path.realpath(__file__))

    # Loading ImageNet classes info
    try:
      classes_filename = path_of_example_script + '/synset_words.txt'
      classes_file = open(classes_filename, 'r')
    except IOError:
      print("Error couldn't open classes file [%s]" % classes_filename)
      return False
    else:
      with classes_file:
        classes = classes_file.read().splitlines()

        # Loading network
        try:
          weights_filename = path_of_example_script + '/sqz_full.mat'
          data, sqz_mean = load_net(weights_filename)
        except IOError:
          print("Error couldn't open weights file [%s]" % classes_filename)
          return False

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'

        g = tf.Graph()

        # 1st pass - simple classification
        with g.as_default(), tf.Session(config=config) as sess:
            # Building network
            image = tf.placeholder(dtype=get_dtype_tf(),
                                   shape=img_content_shape,
                                   name="image_placeholder")
            keep_prob = tf.placeholder(get_dtype_tf())
            sqznet = net_preloaded(data, image, 'max', True)

            input_image = preprocess(img_content, sqz_mean)

            # Classifying
            sqznet_results = sqznet['classifier_actv'].eval(
              feed_dict={image: [input_image], keep_prob: 1.})[0]

            # Outputting result
            sqz_class = np.argmax(sqznet_results)
            print("\nclass: [%d] '%s' with %5.2f%% confidence" %
                  (sqz_class,
                   classes[sqz_class],
                   sqznet_results[sqz_class] * 100))

            # Exporting network to C++
            print("Using TFMin library to export minimal"
                  "C++ implimentation of SqueezeNet.")
            c_exporter = tfm_ex.Exporter(sess, ['Softmax:0'])

            c_exporter.print_graph()

            # first test sample as validation data
            validation_input = [input_image]
            # validation_output = [sqznet_results]

            res = c_exporter.generate(path_of_example_script +
                                      "/tfmin_generated/squeeze_net",
                                      "SqueezeNet",
                                      validation_inputs={image: validation_input},
                                      # ,sqznet['classifier_actv']:validation_output},
                                      validation_type='Full',
                                      timing=True,
                                      layout='RowMajor')

            return res


def main():

  parser = build_parser()
  options = parser.parse_args()

  if export_sqz_net(options):
    exit(0)
  else:
    exit(1)


if __name__ == '__main__':
    main()
