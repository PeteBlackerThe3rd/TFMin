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

    script to dynamically import op_kernel objects from this path and
    any other paths specified in the TFMIN_OP_KERNEL_PATH environment variable

"""
import os
import pkgutil
import importlib.util
import textwrap

# import core op_kernels
from tf_min.op_kernels import base_op

op_kernel_env_name = 'TFMIN_OPKERNEL_PATH'


def get_op_kernels():
  return base_op.BaseOpKernel.__subclasses__()


def find_op_kernel(op, preferential_tag=''):
  """
  Find an op_kernel to match the given tensorflow operation
  with a preference for op_kernels with the given tag if multiple
  op_kernels are found
  :param op: Tensorflow operation object
  :param preferential_tag: prefered tag to resolve multiple op_kernels
  :return: op_kernel object or None if one wasn't found
  """
  op_kernels = get_op_kernels()
  found_op_kernel = None
  for k in op_kernels:
    if k.matches(op):
      if found_op_kernel is None:
        found_op_kernel = k
      elif k.tag == preferential_tag:
        found_op_kernel = k

  return found_op_kernel


def list_op_kernels():
  """
  Prints a list of all op_kernels currently loaded
  """

  wrapper = textwrap.TextWrapper(width=80 - 16)
  op_kernels = get_op_kernels()

  print("-" * 80)
  print("  %d Operations supported by this installation of TFMin." %
        len(op_kernels))
  print("-" * 80)
  for op_k in op_kernels:
      status = "[Invalid!]"
      if op_k.status() == "Production":
          status = "  \033[92mProduction\033[0m "
      elif op_k.status() == "Testing":
          status = "   \033[93mTesting\033[0m   "
      elif op_k.status() == "Development":
          status = " \033[91mDevelopment\033[0m "

      desc_lines = wrapper.wrap(op_k.description())
      wrapped_desc = ("\n" + (" " * 16)).join(desc_lines)
      print("(%s) %s" % (status, wrapped_desc))
  print("-" * 80)


def import_op_kernels(verbose=False):
  # Get list of paths from the tfmin opkernel path if it's set
  op_kernel_paths = []
  if op_kernel_env_name in os.environ:
    op_kernel_paths = os.environ[op_kernel_env_name].split(':')

  # add the TFMin builtin opkernel path (path of this script)
  op_kernel_paths.append(os.path.dirname(os.path.realpath(__file__)))

  # remove duplicated paths
  op_kernel_paths = list(set(op_kernel_paths))

  if verbose:
    print("Found %s op_kernel_paths" % len(op_kernel_paths))
    for p in op_kernel_paths:
      print(p)

  op_kernel_classes = []

  if verbose:
    print("Searching for op_kernel classes within paths")

  for kernel_path in op_kernel_paths:
    for (finder, name, ispkg) in pkgutil.iter_modules([kernel_path]):

      # skip this script to avoid infinite recurrsion
      if name == 'import_op_kernels':
        continue

      # skip the base op kernel which has already been imported
      if name == 'base_op':
        continue

      if verbose:
        print("Found finder(%s) name(%s) ispkg(%s)" %
              (finder, name, ispkg))

      spec = importlib.util.spec_from_file_location('*',
                                                    finder.path+'/'+name+'.py')
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      op_kernel_classes = base_op.BaseOpKernel.__subclasses__()

  if verbose:
    print('-' * 80)
    print("Successfully loaded [%d] op_kernel classes" % len(
      op_kernel_classes))
    print('-' * 80)
    wrapper = textwrap.TextWrapper(width=80 - 16)
    for op_k in op_kernel_classes:
      status = "[Invalid!]"
      if op_k.status() == "Production":
        status = "  \033[92mProduction\033[0m "
      elif op_k.status() == "Testing":
        status = "   \033[93mTesting\033[0m   "
      elif op_k.status() == "Development":
        status = " \033[91mDevelopment\033[0m "

      desc_lines = wrapper.wrap(op_k.description())
      wrapped_desc = ("\n" + (" " * 16)).join(desc_lines)
      print("(%s) %s" % (status, wrapped_desc))
    print("-" * 80)
