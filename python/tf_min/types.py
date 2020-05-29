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


def get_dtype_size(d_type):
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


def get_dtype_lowest(d_type):
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


def get_dtype_struct_type(d_type):
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