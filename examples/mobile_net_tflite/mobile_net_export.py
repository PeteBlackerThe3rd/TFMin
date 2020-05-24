import tensorflow as tf
import scipy.io
import numpy as np
import argparse
import sys
import os

from tf_min import exporter as tfm_ex
from tf_min import graph as tfm_g
from tf_min import graph_from_tflite as g_gen
from tf_min import graph_2_svg as tfm_svg
import tf_min.graph_mem_2_svg as tfm_mem_svg
import tf_min.mem_opt.graph_heap_opt as tfm_heap_opt
import tf_min.mem_opt.graph_peak_reorder_opt as tfm_peak_reorder_opt
import tf_min.mem_opt.heap_smart_order as tfm_heap_smart_order
import tf_min.mem_opt.op_split as tfm_op_split


def load_and_export_model(args):

    # read tflite flatbuffer
    try:
        fb_file = open(args.flatbuffer, 'rb')
        flatbuffer = bytearray(fb_file.read())
    except IOError as err:
        print("Error failed to read flatbuffer file \"%s\" [%s]" %
              (args.flatbuffer,
               str(err)))
        return

    # create TFMin graph from tflite model
    graph = g_gen.graph_from_tflite(flatbuffer)

    #svg_writer = tfm_svg.SVGWriter(graph)
    #svg_writer.write("original_graph.svg")

    sequencer = tfm_g.SequenceOps(graph)
    graph_seq = sequencer.translate(verbose=True)

    mem_opt = tfm_heap_smart_order.HeapSmartOrder(graph_seq, {'order': 'forwards'})
    graph_alloc = mem_opt.translate(verbose=True)
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph_alloc)
    mem_svg_writer.write("heap_smart_order_memory.svg")

    # mem_opt = tfm_heap_opt.HeapAllocateGraph(graph_seq, {'order': 'backwards'})
    """mem_opt = tfm_peak_reorder_opt.PeakReorderAllocateGraph(
      graph_seq,
      {'initial-order': 'backwards'}
    )
    graph = mem_opt.translate(verbose=True)
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph)
    mem_svg_writer.write("peak_reorder_memory.svg")"""

    """mem_opt = tfm_heap_opt.HeapAllocateGraph(graph_seq, {'order': 'backwards'})
    graph = mem_opt.translate(verbose=True)
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph)
    mem_svg_writer.write("heap_memory.svg")"""

    heap_allocator_params = {'allocator-params': {'order': 'forwards'},
                             'output-debug': True}
    peak_reorder_params = {'allocator': tfm_peak_reorder_opt.PeakReorderAllocateGraph,
                           'allocator-params': {'initial-order': 'backwards'},
                           'output-debug': True}
    heap_smart_order_params = {'allocator': tfm_heap_smart_order.HeapSmartOrder,
                               'allocator-params': {'order': 'forwards'},
                               'output-debug': True}

    # test op_splitting
    op_splitter = tfm_op_split.OprSplit(graph_seq,
                                        params=heap_smart_order_params)
    graph_new = op_splitter.translate(verbose=True)
    #svg_writer = tfm_svg.SVGWriter(graph_new)
    #svg_writer.write("graph_split.svg")

    mem_opt = tfm_heap_opt.HeapAllocateGraph(graph_new, {'order': 'backwards'})
    graph = mem_opt.translate(verbose=True)
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph)
    mem_svg_writer.write("graph_split_memory.svg")

    # Exporting network to C++
    """print("Using TFMin library to export minimal C++ implimentation of MobileNet.")
    c_exporter = tfm_ex.Exporter(sess,
                                 ['output/ArgMax:0'],
                                 default_feed_dict={trainer.model.is_training: False})

    #c_exporter.print_graph()

    print("input tensor shape is [%s]" % str(model.X.shape))

    path_of_script = os.path.dirname(os.path.realpath(__file__))
    test_img = imread_resize(path_of_script + "/data/test_images/3.jpg",
                             size=(224, 224))
    test_img = test_img.reshape((1,224,224,3))

    # first test sample as validation data
    #validation_input = [input_image]
    #validation_output = [sqznet_results]

    res = c_exporter.generate(path_of_script + "/tfmin_generated/mobile_net",
                              "MobileNet",
                              validation_inputs={model.X: test_img},
                              validation_type='None',
                              timing=True,
                              layout='RowMajor')

    sess.close()
    print("Complete")
    return res"""


def main():

  # create argument parser
  parser = argparse.ArgumentParser(
    description="MobileNet TensorFlow Implementation"
  )
  parser.add_argument('--flatbuffer',
                      default="models/mobilenet_v2_0.35_224.tflite",
                      type=str,
                      help='flatbuffer file containing tflite model.')

  args = parser.parse_args()
  load_and_export_model(args)


if __name__ == '__main__':
    main()
