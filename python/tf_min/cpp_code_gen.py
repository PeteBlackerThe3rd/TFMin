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

    Module containing helper functions used to generate C++ code
"""

import numpy as np
import tensorflow as tf


def c_safe_identifier(name):
    """Returns a c/c++ compatible identifier string from a TensorFlow label."""
    c_safe = name.replace("/", "_")
    c_safe = c_safe.replace(":", "_")
    c_safe = c_safe.replace("-", "_")
    return c_safe


def get_c_dtype(dtype):
    """Returns the c/c++ equivalent type name of a numpy or TensorFlow dtype object."""
    c_data_type = "Error!"

    if dtype == np.bool or (type(dtype) != np.dtype and (dtype.base_dtype == tf.bool)):
        c_data_type = "bool"
    elif dtype == np.int8 or (type(dtype) != np.dtype and (dtype.base_dtype == tf.int8)):
        c_data_type = "signed char"
    elif dtype == np.int16 or (type(dtype) != np.dtype and (dtype.base_dtype == tf.int16)):
        c_data_type = "short"
    elif dtype == np.int32 or (type(dtype) != np.dtype and (dtype.base_dtype == tf.int32)):
        c_data_type = "int"
    elif dtype == np.int64 or (type(dtype) != np.dtype and (dtype.base_dtype == tf.int64)):
        c_data_type = "long"
    elif dtype == np.float32 or (type(dtype) != np.dtype and (dtype.base_dtype == tf.float32)):
        c_data_type = "float"
    elif dtype == np.float64 or (type(dtype) != np.dtype and (dtype.base_dtype == tf.float64)):
        c_data_type = "double"
    else:
        print("Error, unrecognized type [%s] type [%s]" % (str(dtype), type(dtype)))
    return c_data_type


def get_c_dtype_size(dtype):

    c_type = get_c_dtype(dtype)
    if c_type == "signed char":
        return 1
    if c_type == "short":
        return 2
    if c_type == "int" or c_type == "float":
        return 4
    if c_type == "long" or c_type == "double":
        return 8
    # safety!
    return 1


def ndarray_1d_to_literal(array, open='{', close='}'):
    """Returns a c/c++ array initialiser list generated from a 1 dimensional numpy.ndarray."""
    output_str = open + " "
    for d in range(array.shape[0]):
        if d < array.shape[0] - 1:
            output_str += str(array[d]) + ", "
        else:
            output_str += str(array[d])
    output_str += " " + close
    return output_str


def prepend_lines(text, line_prefix):
    """Returns a copy of text with <line_prefix> added to the start of every line."""
    return line_prefix + text.replace("\n", "\n"+line_prefix)
