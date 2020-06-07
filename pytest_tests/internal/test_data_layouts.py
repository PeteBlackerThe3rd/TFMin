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

    This test creates a set of very simple tensorflow graphs and compares
    the output produced when run within tensorflow and the output produced
    when the models are converted to c code and executed.

    This test verifies that the data layouts are being respected correctly and
    that simple binary elementwise operators and transpose are being generated
    in c correctly
"""
import scipy.io
import numpy as np
import struct as st
import argparse
import sys
import os
import fcntl
import subprocess as sp
import tensorflow as tf

from tf_min import graph as tfm_g
from tf_min import graph_from_tf as tfm_tf
from tf_min import graph_2_svg as tfm_svg
import tf_min.mem_opt.graph_heap_opt as tfm_heap_opt
from tf_min.graph_verify import GraphVerifyOutput


def main():
    """

    """

    tf.reset_default_graph()
    sess = tf.Session()

    # make super simple graph which multiplies it input vector by a 5x5
    # identity matrix
    input = tf.placeholder(tf.float32, [1, 5], name='input')
    constant = tf.constant(np.identity(5), dtype=tf.float32)
    output = tf.matmul(input, constant, name="output")
    tf.global_variables_initializer().run(session=sess)

    # Generate expected output from this graph
    test_input = np.array([[1, 2, 3, 4, 5]]).astype(np.float32)
    [expected_output] = sess.run([output], feed_dict={input: test_input})

    # convert this graph into a tfmin graph
    graph = tfm_tf.graph_from_tf_sess(sess, outputs=[output])

    const = graph.get_constants()[0]
    print("[%s] value type is %s" % (const.label, type(const.value)))

    # prepare this graph for export by sequencing the operation and allocating
    # any intermediate buffers
    sequencer = tfm_g.SequenceOps(graph)
    graph = sequencer.translate()

    const = graph.get_constants()[0]
    print("[%s] value type is %s" % (const.label, type(const.value)))

    mem_opt = tfm_heap_opt.HeapAllocateGraph(graph,
                                             {'order': 'forwards'})
    graph_alloc = mem_opt.translate(verbose=True)

    const = graph_alloc.get_constants()[0]
    print("[%s] value type is %s" % (const.label, type(const.value)))

    svg_writer = tfm_svg.SVGWriter(graph_alloc)
    svg_writer.write("original_input_graph.svg")
    print("Done.")

    # verify the output of this graph matches tensorflow when it is
    # exported, built, and executed
    print("Testing verify output test harness")
    verifier = GraphVerifyOutput(graph=graph_alloc,
                                 verbose=True,
                                 tmp_dir="verify_tmp")

    result = verifier.verify_model(input_tensors=[test_input],
                                   expected_output_tensors=[expected_output])

    print("verification result:\n%s" % result)



if __name__ == "__main__":
    main()
