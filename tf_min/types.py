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

    This module defines the element types supported by TFMin along with
    a set of common conversions
"""
import numpy as np
from enum import Enum


class TenDType(Enum):
    FLOAT64 = 1
    FLOAT32 = 2
    FLOAT16 = 3
    INT64 = 4
    INT32 = 5
    INT16 = 6
    INT8 = 7
    UINT64 = 8
    UINT32 = 9
    UINT16 = 10
    UINT8 = 11


NP_CONVERSION = {np.float32: TenDType.FLOAT32,
                 np.float64: TenDType.FLOAT64,
                 np.int8: TenDType.INT8,
                 np.int16: TenDType.INT16,
                 np.int32: TenDType.INT32,
                 np.int64: TenDType.INT64,
                 np.uint8: TenDType.UINT8,
                 np.uint16: TenDType.UINT16,
                 np.uint32: TenDType.UINT32,
                 np.uint64: TenDType.UINT64,}

def np_to_tfmin(src_type):
    """
    Function which converts a numpy data type to a TFMin data type
    :param np_type: numpy data type
    :return: TenDType
    """
    if not isinstance(src_type, np.dtype):
        raise TypeError("Error: np_to_tfmin, source type is not a numpy.dtype")
    #if np_type not in NP_CONVERSION.keys():
    #  raise TypeError("Error: np_to_tfmin, source type not supported [%s]" % str(np_type))
    for np_type, tm_type in NP_CONVERSION.items():
      if src_type == np_type:
        return tm_type

    raise TypeError("Error: np_to_tfmin, source type not supported [%s]" % str(src_type))


def tfmin_to_np(tfmin_type):
    """
    Function which converts a numpy data type to a TFMin data type
    :param np_type: numpy data type
    :return: TenDType
    """
    if not isinstance(tfmin_type, TenDType):
        raise TypeError("Error: tfmin_to_np, source type is not a "
                        "tf_min.TenDType")
    if tfmin_type not in NP_CONVERSION.values():
      raise TypeError("Error: tfmin_to_np, source type not supported")
    for np_type, tm_type in NP_CONVERSION.items():
      if tm_type == tfmin_type:
        return np_type


def get_dtype_size(d_type):
    """
    Function which returns the storage size of this data type in bytes.
    :param d_type: TenDType, the data type
    :return: Int, it's storage size in bytes
    """
    sizes = {TenDType.FLOAT64: 8,
             TenDType.FLOAT32: 4,
             TenDType.FLOAT16: 2,
             TenDType.INT64: 8,
             TenDType.INT32: 4,
             TenDType.INT16: 2,
             TenDType.INT8: 1,
             TenDType.UINT64: 8,
             TenDType.UINT32: 4,
             TenDType.UINT16: 2,
             TenDType.UINT8: 1}
    return sizes[d_type]


def get_dtype_c_type(d_type):
    """
    Function which returns a string describing the equivalent c type of this
    data type
    :param d_type: TenDType, the data type
    :return: String
    """
    c_types = {TenDType.FLOAT64: "double",
               TenDType.FLOAT32: "float",
               TenDType.FLOAT16: "uint16_t",  # Note c half libs use this type
               TenDType.INT64: "int64_t",
               TenDType.INT32: "int32_t",
               TenDType.INT16: "int16_t",
               TenDType.INT8: "int8_t",
               TenDType.UINT64: "uint64_t",
               TenDType.UINT32: "uint32_t",
               TenDType.UINT16: "uint16_t",
               TenDType.UINT8: "uint8_t"}
    return c_types[d_type]



def get_dtype_description(d_type):
  """
  Function which returns a string describing this datatype
  data type
  :param d_type: TenDType, the data type
  :return: String
  """
  c_types = {TenDType.FLOAT64: "float 64",
             TenDType.FLOAT32: "float 32",
             TenDType.FLOAT16: "float 16",
             TenDType.INT64: "int 64",
             TenDType.INT32: "int 32",
             TenDType.INT16: "int 16",
             TenDType.INT8: "int 8",
             TenDType.UINT64: "unsigned int 64",
             TenDType.UINT32: "unsigned int 32",
             TenDType.UINT16: "unsigned int 16",
             TenDType.UINT8: "unsigned int 8"}
  return c_types[d_type]


def get_dtype_lowest(d_type):
    """
    Function to return a string describing the lowest possible value this
    data type can store in C code.
    This returns a c integer literal for integer types and the name
    of a macro defined in float.h for float types
    :param d_type: TenDType, the data type
    :return: String
    """
    lowest_values = {TenDType.FLOAT64: "DBL_MIN",
                     TenDType.FLOAT32: "FLT_MIN",
                     TenDType.FLOAT16: "#TODO#",
                     TenDType.INT64: "-9223372036854775808",
                     TenDType.INT32: "-2147483648",
                     TenDType.INT16: "-32768",
                     TenDType.INT8: "-128",
                     TenDType.UINT64: "0",
                     TenDType.UINT32: "0",
                     TenDType.UINT16: "0",
                     TenDType.UINT8: "0"}
    return lowest_values[d_type]


def get_dtype_highest(d_type):
    """
    Function to return a string describing the highest possible value this
    data type can store in C code.
    This return an integer in a string for integer types and the name
    of a macro defined in float.h for float types
    :param d_type: TenDType, the data type
    :return: String
    """
    highest_values = {TenDType.FLOAT64: "DBL_MAX",
                      TenDType.FLOAT32: "FLT_MAX",
                      TenDType.FLOAT16: "#TODO#",
                      TenDType.INT64: "9223372036854775807",
                      TenDType.INT32: "2147483647",
                      TenDType.INT16: "32767",
                      TenDType.INT8: "127",
                      TenDType.UINT64: "18446744073709551615",
                      TenDType.UINT32: "4294967295",
                      TenDType.UINT16: "65536",
                      TenDType.UINT8: "255"}
    return highest_values[d_type]


def get_dtype_zero(d_type):
    """
    Function to return the c literal zero string for the given data type
    :param d_type: data type
    :return: String, zero literal
    """
    if d_type == TenDType.FLOAT64:
        return "0.0"
    elif d_type == TenDType.FLOAT32:
        return "0.0f"
    else:
        return "0"


def is_float(d_type):
    if d_type in [TenDType.FLOAT64, TenDType.FLOAT32, TenDType.FLOAT16]:
        return True
    else:
        return False


def is_integer(d_type):
    if d_type in [TenDType.INT64, TenDType.INT32,
                  TenDType.INT16, TenDType.INT8,
                  TenDType.UINT64, TenDType.UINT32,
                  TenDType.UINT16, TenDType.UINT8]:
        return True
    else:
        return False


def is_signed_integer(d_type):
    if d_type in [TenDType.INT64, TenDType.INT32,
                  TenDType.INT16, TenDType.INT8]:
        return True
    else:
        return False


def is_unsigned_integer(d_type):
    if d_type in [TenDType.UINT64, TenDType.UINT32,
                  TenDType.UINT16, TenDType.UINT8]:
        return True
    else:
        return False


def get_higher_range_type(d_type):
    """
    Function to return the next higher range type than the given type
    if FLOAT64, INT64 or UINT64 are given then these same types are returned.
    :param d_type: TenDType, the type
    :return: TenDType, the higher range type
    """
    higher_range_types = {TenDType.FLOAT64: TenDType.FLOAT64,
                          TenDType.FLOAT32: TenDType.FLOAT64,
                          TenDType.FLOAT16: TenDType.FLOAT32,
                          TenDType.INT64: TenDType.INT64,
                          TenDType.INT32: TenDType.INT64,
                          TenDType.INT16: TenDType.INT32,
                          TenDType.INT8: TenDType.INT16,
                          TenDType.UINT64: TenDType.UINT64,
                          TenDType.UINT32: TenDType.UINT64,
                          TenDType.UINT16: TenDType.UINT32,
                          TenDType.UINT8: TenDType.UINT16}
    return higher_range_types[d_type]


def get_dtype_struct_type(d_type):
    """
    Function to return the data type string used in pythons struct pack and
    unpack functions.
    :param d_type: TenDType, data type
    :return: String
    """
    c_types = {TenDType.FLOAT64: "d",
               TenDType.FLOAT32: "f",
               TenDType.FLOAT16: "#Error#",
               TenDType.INT64: "q",
               TenDType.INT32: "i",
               TenDType.INT16: "h",
               TenDType.INT8: "b",
               TenDType.UINT64: "Q",
               TenDType.UINT32: "I",
               TenDType.UINT16: "H",
               TenDType.UINT8: "B"}
    return c_types[d_type]
