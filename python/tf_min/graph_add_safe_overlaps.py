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

    ...
"""
import tf_min.graph as tm

# use the dynamic op_kernel_loader to import op_kernels from all requires paths
from tf_min.op_kernels import base_op
from tf_min.v2_kernels.base_op_kernel import BaseOpKernel
from tf_min.v2_kernels import *
# import tf_min.op_kernels.import_op_kernels as op_kernel_loader
# op_kernel_loader.import_op_kernels(verbose=False)


def add_safe_overlaps_to_graph(graph, verbose=False):
  """
  Function searches for op_kernels for each operation in the graph, and
  uses them to compute the safe overlap between input & output tensors of
  each operation.

  This is a pre-processing step for the diagonal memory optimisation
  method which significantly reduces the memory requirement of the model.
  :return: None
  """

  print("Found %d op kernels" % len(BaseOpKernel.__subclasses__()))

  safe_overlaps_found = 0
  for op in graph.ops:

    op_kernel = None
    for Kernel in BaseOpKernel.__subclasses__():
      if Kernel.matches(op):
        op_kernel = Kernel(op)

    # op_k = op_kernel_loader.find_op_kernel(op)
    if op_kernel is not None:
      safe_overlap_bytes = op_kernel.get_safe_overlap()
      if safe_overlap_bytes is not None and safe_overlap_bytes > 0:
        op.outputs[0].safe_overlap_preceding_tensor = op.inputs[0]
        op.outputs[0].safe_overlap_bytes = safe_overlap_bytes
        safe_overlaps_found += 1
        if verbose:
          print("Found a safe overlap of %12d bytes for op [%20s]" %
                (safe_overlap_bytes,
                 op.type))

    elif verbose:
      print("Warning: No op_kernel matched for operation type [%s]" %
            op.type)

  if verbose:
    print("Found %d safe overlaps within %d operations of the model" %
          (safe_overlaps_found,
           len(graph.ops)))
