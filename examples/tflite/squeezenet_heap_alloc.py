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
from tf_min import graph_add_safe_overlaps as aso
import tf_min.graph_mem_2_svg as tfm_mem_svg
import tf_min.mem_opt.graph_heap_opt as tfm_heap_opt
import tf_min.mem_opt.graph_peak_reorder_opt as tfm_peak_reorder_opt
import tf_min.mem_opt.graph_seq_allocator as tfm_seq_opt
import tf_min.mem_opt.heap_smart_order as tfm_heap_smart_order
import tf_min.mem_opt.op_split as tfm_op_split


def load_and_sequence_graph(buffer, dmo=False):

    graph = g_gen.graph_from_tflite(buffer)
    sequencer = tfm_g.SequenceOps(graph)
    graph_seq = sequencer.translate()

    if dmo:
        aso.add_safe_overlaps_to_graph(graph_seq)

    return graph_seq


def load_and_export_model():

    attempt_seq = False
    flatbuffer_filename = "models/squeezenet.tflite"

    # read tflite flatbuffer
    try:
        fb_file = open(flatbuffer_filename, 'rb')
        flatbuffer = bytearray(fb_file.read())
    except IOError as err:
        print("Error failed to read flatbuffer file \"%s\" [%s]" %
              (flatbuffer_filename,
               str(err)))
        return

    # generate SVG of flow graph if requested
    print("Saving svg of models flow-graph.")
    graph = load_and_sequence_graph(flatbuffer)
    svg_writer = tfm_svg.SVGWriter(graph)
    svg_writer.write("original_input_graph.svg")
    print("Done.")

    # sequencer = tfm_g.SequenceOps(graph)
    # graph_seq = sequencer.translate(verbose=True)
    # graph_seq = graph

    if attempt_seq:
      print("\n==== Allocating memory using sequential allocation. ====")
      seq_opt = tfm_seq_opt.SeqAllocateGraph(
        load_and_sequence_graph(flatbuffer)
      )
      if seq_opt.graph_allocated:
        seq_opt_graph = seq_opt.translate()
        seq_opt_result = seq_opt_graph.get_peak_memory()
        mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(seq_opt_graph)
        mem_svg_writer.write("sequential_memory.svg")
        print("Sequential graph allocated okay.")
      else:
        print("Skipped, graph is not sequential.")

    print("\n==== Allocating memory using heap ordering ====")
    mem_opt = tfm_heap_opt.HeapAllocateGraph(
      load_and_sequence_graph(flatbuffer),
      {'order': 'forwards'}
    )
    graph_heap_alloc = mem_opt.translate(verbose=True)
    result_heap_order = graph_heap_alloc.get_peak_memory()
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph_heap_alloc)
    mem_svg_writer.write("heap_order_memory.svg")
    print("-"*80)

    print("\n==== Allocating memory using heap smart ordering. ====")
    mem_opt = tfm_heap_smart_order.HeapSmartOrder(
      load_and_sequence_graph(flatbuffer),
      {'order': 'forwards'}
    )
    graph_alloc = mem_opt.translate(verbose=True)
    result_smart_order = graph_alloc.get_peak_memory()
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph_alloc)
    mem_svg_writer.write("heap_smart_order_memory.svg")
    print("-"*80)

    return

    print("\n==== Allocating memory using peak re-ordering ====")

    peak_reorder_opt = tfm_peak_reorder_opt.PeakReorderAllocateGraph(
      load_and_sequence_graph(flatbuffer),
      {'initial-order': 'backwards'})
    peak_re_allocated = peak_reorder_opt.translate(verbose=True)
    result_peak_reorder = peak_re_allocated.get_peak_memory();
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(peak_re_allocated)
    mem_svg_writer.write("peak_reorder_memory.svg")
    print("-"*80)

    if attempt_seq:
      print("\n==== Allocating memory using sequential allocation and DMO. ====")
      seq_opt_dmo = tfm_seq_opt.SeqAllocateGraph(load_and_sequence_graph(flatbuffer, dmo=True))
      if seq_opt_dmo.graph_allocated:
        seq_opt_graph_dmo = seq_opt_dmo.translate()
        seq_opt_dmo_result = seq_opt_graph.get_peak_memory()
        mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(seq_opt_graph_dmo)
        mem_svg_writer.write("DMO_sequential_memory.svg")
        print("Sequential graph allocated okay.")
      else:
        print("Skipped, graph is not sequential.")
      print("-"*80)

    print("\n==== Allocating memory using heap smart ordering and DMO. ====")
    # graph_alloc.print_operation_counts()
    # aso.add_safe_overlaps_to_graph(graph_alloc)

    # graph_alloc.print_safe_overlaps()

    heap_mem_opt = tfm_heap_smart_order.HeapSmartOrder(load_and_sequence_graph(flatbuffer, dmo=True),
                                                  {'order': 'backwards'})
    dmo_allocated = heap_mem_opt.translate(verbose=True)
    result_smart_order_dmo = dmo_allocated.get_peak_memory()
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(dmo_allocated)
    mem_svg_writer.write("DMO_smart_order_memory.svg")
    print("-"*80)

    print("\n==== Allocating memory using peak reordering and DMO. ====")
    peak_reorder_opt = tfm_peak_reorder_opt.PeakReorderAllocateGraph(load_and_sequence_graph(flatbuffer, dmo=True),
                                                                     {'initial-order': 'backwards'})
    peak_re_allocated = peak_reorder_opt.translate(verbose=True)
    result_peak_reorder_dmo = peak_re_allocated.get_peak_memory()
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(peak_re_allocated)
    mem_svg_writer.write("DMO_peak_reorder_memory.svg")
    print("-"*80)

    if attempt_seq:
      print("\n==== Allocating memory using op_splitting and sequential allocation. ====")

      sequential_params = {
        'allocator': tfm_seq_opt.SeqAllocateGraph,
        'allocator-params': {'order': 'forwards'},
        'output-debug': True,
        'save-graphs': False,
        'attempt-seq': False}

      # test op_splitting
      op_splitter_seq = tfm_op_split.OprSplit(load_and_sequence_graph(flatbuffer),
                                              params=sequential_params)
      graph_new = op_splitter_seq.translate(verbose=True)

      mem_opt = tfm_seq_opt.SeqAllocateGraph(graph_new, {'order': 'forwards'})
      graph = mem_opt.translate(verbose=True)
      mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph)
      mem_svg_writer.write("graph_split_memory_sequential.svg")
      if args.print_graph:
        svg_writer = tfm_svg.SVGWriter(graph)
        svg_writer.write("op_split_seq_graph.svg")
      print("-"*80)

    print("\n==== Allocating memory using op-splitting and heap smart order ====")
    heap_smart_order_params = {
      'allocator': tfm_heap_smart_order.HeapSmartOrder,
      'allocator-params': {'order': 'forwards'},
      'output-debug': True,
      'save-graphs': False,
      'attempt-seq': False}

    # test op_splitting
    op_splitter_hso = tfm_op_split.OprSplit(load_and_sequence_graph(flatbuffer),
                                        params=heap_smart_order_params)
    op_splitter_hso.translate(verbose=True)
    graph_new = op_splitter_hso.split_options[-1]['graph']
    mem_opt = tfm_heap_smart_order.HeapSmartOrder(graph_new, {'order': 'forwards'})
    graph = mem_opt.translate(verbose=True)
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph)
    mem_svg_writer.write("op_split_memory_smart_reorder.svg")
    print("-"*80)


    print(
      "\n==== Allocating memory using op-splitting and peak reordering ====")
    peak_reorder_params = {
      'allocator': tfm_peak_reorder_opt.PeakReorderAllocateGraph,
      'allocator-params': {'initial-order': 'backwards'},
      'output-debug': True,
      'save-graphs': False,
      'attempt-seq': False
    }
    # sequencer = tfm_g.SequenceOps(graph)
    # graph_seq = sequencer.translate(verbose=True)

    op_splitter_pr = tfm_op_split.OprSplit(load_and_sequence_graph(flatbuffer),
                                        params=peak_reorder_params)
    graph_new = op_splitter_pr.translate(verbose=True)

    mem_opt = tfm_heap_smart_order.HeapSmartOrder(graph_new,
                                                  {'order': 'forwards'})
    graph = mem_opt.translate(verbose=True)
    mem_svg_writer = tfm_mem_svg.SVGMemoryWriter(graph)
    mem_svg_writer.write("graph_split_memory_smart_reorder.svg")
    print("-" * 80)

    print("\n\n --- Results Summary ---\n")
    print("Modle file : %s" % args.flatbuffer)
    if attempt_seq and seq_opt.graph_allocated:
      print("sequentual allocator : %10d (%7d KB)" %
            (seq_opt_result, round(seq_opt_result / 1024.0)))
    else:
      print("sequentual allocator : n/a")
    if attempt_seq and seq_opt_dmo.graph_allocated:
      print("seq alloctor (DMO)   : %10d (%7d KB)" %
            (seq_opt_dmo_result, round(seq_opt_dmo_result / 1024.0)))
    else:
      print("seq alloctor (DMO)   : n/a")
    print("heap smart order       : %10d (%7d KB)" %
          (result_smart_order, round(result_smart_order / 1024.0)))
    print("heap smart order (DMO) : %10d (%7d KB)" %
          (result_smart_order_dmo, round(result_smart_order_dmo / 1024.0)))
    print("peak re-order          : %10d (%7d KB)" %
          (result_peak_reorder, round(result_peak_reorder / 1024.0)))
    print("peak re-order (DMO)    : %10d (%7d KB)" %
          (result_peak_reorder_dmo, round(result_peak_reorder_dmo / 1024.0)))

    print("-- op-split results --")
    if attempt_seq and seq_opt.graph_allocated:
      best_seq = op_splitter_seq.split_options[-1]
      print("seq allocator          : %10d (%7d KB) [Recomp %d %5.2f%%]" %
            (best_seq['peak_mem'],
             best_seq['peak_mem'] / 1024,
             best_seq['recomputations'],
             (best_seq['recomputations'] /
              op_splitter_seq.split_options[0]['recheck'])
             ))
    else:
      print("seq allocator          : n/a")
    best_hso = op_splitter_hso.split_options[-1]
    print("heap smart order       : %10d (%7d KB) [Recomp %d %5.2f%%]" %
          (best_hso['peak_mem'],
           best_hso['peak_mem'] / 1024,
           best_hso['recomputations'],
           (best_hso['recomputations'] /
            op_splitter_hso.split_options[0]['recheck'])
           ))
    best_pr = op_splitter_pr.split_options[-1]
    print("peak re-order          : %10d (%7d KB) [Recomp %d %5.2f%%]" %
          (best_pr['peak_mem'],
           best_pr['peak_mem'] / 1024,
           best_pr['recomputations'],
           (best_pr['recomputations'] /
            op_splitter_pr.split_options[0]['recheck'])
           ))


def main():

  load_and_export_model()


if __name__ == '__main__':
    main()
