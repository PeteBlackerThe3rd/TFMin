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

    This module defines the activation functions supported by TFMin. This
    includes types of activation and the c code which implements them.
"""
from enum import Enum
import tf_min.types as types


class ActType(Enum):
  """
  Enum defining the types of activation functions supported by TFMin
  """
  NONE = 0
  RELU = 1
  RELUN1TO1 = 2
  RELU6 = 3
  TANH = 4
  SIGNBIT = 5
  LEAKY_RELU = 6


def gen_act_code(act_type, d_type, args=None):
  """
  Function to return the c code implementation of the given activation type
  Note this assumes that the element value is held in a variable called 'value'
  :param act_type: ActType, the type of activation function to generate
  :param d_type: TenDType, the datatype of the value being activated.
  :param args: List, or arguments or None used for leaky relu alpha.
  :return: String, containing the c code implementation
  """
  if act_type == ActType.NONE:
    return "// None"
  elif act_type == ActType.RELU:
    return "// Relu\nif (value < 0)\n  value = 0;"
  elif act_type == ActType.RELUN1TO1:
    return ("// Relu -1 to 1\nif (value < -1)\n  value = -1;\n"
            "if (value > 1)\n  value = 1;")
  elif act_type == ActType.RELU6:
    return ("// Relu 6\nif (value < 0)\n  value = 0;\n"
            "if (value > 6)\n  value = 6;")
  elif act_type == ActType.TANH:
    if d_type == types.TenDType.FLOAT64:
      return "// relu tanh (double precision float)\nvalue = tanh(value);"
    else:
      return "// relu tanh (single precision float)\nvalue = tanhf(value);"
  elif act_type == ActType.SIGNBIT:
    print("Error: SignBit activation function is not supported at this time.")
    return ("// Error: SignBit activation function is not "
            "supported at this time.")
  elif act_type == ActType.LEAKY_RELU:
    if args is None or 'act_alpha' not in args:
      print("Error: LeakyRelu specified with not Alpha coefficient!")
      return "// Error: LeakyRelu specified with no Alpha coefficient!"
    else:
      code = "// Leaky Relu, Alpha = %f\n" % args['act_alpha']
      code += "if (value < 0)\n  value *= %f;\n" % args['act_alpha']
      return code
