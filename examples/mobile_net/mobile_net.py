import tensorflow as tf
# import scipy.misc as misc
import scipy.io
import scipy.misc
import numpy as np
import argparse
import sys
import os

from utils import parse_args, create_experiment_dirs, calculate_flops
from model import MobileNet
from train import Train
from data_loader import DataLoader
from summarizer import Summarizer

from tf_min import exporter as tfm_ex


def imread_resize(path, size=(227, 227)):
    img_orig = scipy.misc.imread(path)
    img = scipy.misc.imresize(img_orig, size).astype(np.float)

    # grayscale
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))

    # RGB / RGBA
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, 0:3]

    return img


def build_mobile_net_model(config_filename):
    # Parse the JSON arguments
    try:
        config_args = parse_args(config_filename)
    except argparse.ArgumentError as exc:
        print("Add a config file using \'--config file_name.json\'\n%s" % exc)
        return False

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = \
        create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    data = DataLoader(config_args.batch_size, config_args.shuffle)
    print("Loading Data...")
    (config_args.num_channels,
     config_args.train_data_size,
     config_args.test_data_size) = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    model = MobileNet(config_args)
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    trainer = Train(sess, model, data, summarizer)

    """if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()"""

    if config_args.to_test:
        print("Final test!")
        trainer.test('val')
        print("Testing Finished\n\n")

    # write graph to tensorboard summary
    # merged = tf.summary.merge_all()
    graph_writer = tf.summary.FileWriter('/tmp/tensorflow/mobile_net/graph',
                                         sess.graph)
    tf.global_variables_initializer().run(session=sess)
    graph_writer.close()

    # Exporting network to C++
    print("Using TFMin library to export C++ implimentation of MobileNet.")
    c_exporter = tfm_ex.Exporter(
        sess,
        ['output/ArgMax:0'],
        default_feed_dict={trainer.model.is_training: False}
    )

    # c_exporter.print_graph()

    print("input tensor shape is [%s]" % str(model.X.shape))

    path_of_script = os.path.dirname(os.path.realpath(__file__))
    test_img = imread_resize(path_of_script + "/data/test_images/3.jpg",
                             size=(224, 224))
    test_img = test_img.reshape((1, 224, 224, 3))

    # first test sample as validation data
    # validation_input = [input_image]
    # validation_output = [sqznet_results]

    res = c_exporter.generate(path_of_script + "/tfmin_generated/mobile_net",
                              "MobileNet",
                              validation_inputs={model.X: test_img},
                              validation_type='None',
                              timing=True,
                              layout='RowMajor')

    sess.close()
    print("Complete")
    return res


def main():

    # create argument parser
    parser = argparse.ArgumentParser(
        description="MobileNet TensorFlow Implementation")
    parser.add_argument('--config', default="config/test.json", type=str,
                        help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    if not build_mobile_net_model(args.config):
        exit(1)
    else:
        exit(0)


if __name__ == '__main__':
    main()
