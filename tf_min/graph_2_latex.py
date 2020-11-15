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

    This module contains the LatexWriter object used to save model topologies
    as latex compatible tables. Written primairally to generate an Appedix
    for my thesis the lazy way!
"""
import svgwrite as svg
import tf_min.graph as tg
from .types import get_dtype_description


class LatexWriter:

  HEADER = '''\\begin{{longtable}}{{| p{{ {0}\\textwidth}} | 
p{{ {1}\\textwidth}} | p{{ {2}\\textwidth}} | p{{ {3}\\textwidth}} |}} 
\\hline
\\bfseries Layer & \\bfseries Output Size & \\bfseries Data Type & 
\\bfseries Details \\\\ \\hline \\hline
'''

  FOOTER = '''\caption{{ {0} }}
\label{{ {1} }}
\end{{longtable}}
'''

  ROW = '''{0} & {1} & {2} & {3} \\\\ \hline
'''

  DEFAULT_WIDTHS = [0.24, 0.14, 0.12, 0.36]

  def __init__(self, graph):
    self.graph = graph

  def generate_latex(self, caption="TFMin model", ref_label="", widths=None):
    """
    Function to convert the model graph as latex. Note this only works nicely
    with purely sequential graphs. It'll produce output from any graph,
    but it will be really messy if it's a highly connected graph!
    :return: String, containing the a latex table with a detailed
             description of the graph
    """

    self.graph.sequence_ops()

    if widths is None:
      column_widths = self.DEFAULT_WIDTHS
    else:
      column_widths = widths
    latex = self.HEADER.format(*tuple(column_widths))

    # Create one initial row for each input tensor
    inputs = self.graph.get_inputs()
    for idx, input in enumerate(inputs):
      if len(inputs) == 1:
        label = "Input"
      else:
        label = "Input %d" % idx
      latex += self.ROW.format(
        label,
        input.shape,
        get_dtype_description(input.d_type),
        ""
      )
    latex += '\hline\n'

    # Create a row for each operation
    for opr in self.graph.ops:
      desc_lines = self.generate_opr_description(opr)
      description = self.make_table_multiline(desc_lines)
      latex += self.ROW.format(
        opr.type,
        opr.outputs[0].shape,
        get_dtype_description(opr.outputs[0].d_type),
        description
      )
    latex += '\hline\n'

    # Create a final row for each output tensor
    outputs = self.graph.get_outputs()
    for idx, output in enumerate(outputs):
      if len(outputs) == 1:
        label = "Output"
      else:
        label = "Output %d" % idx
      latex += self.ROW.format(
        label,
        output.shape,
        get_dtype_description(output.d_type),
        ""
      )

    latex += self.FOOTER.format(caption, ref_label)

    return latex

  @staticmethod
  def generate_opr_description(opr):
    description = []
    if opr.type in ["Conv2D", "TransposedConv2D"]:
      description.append("Filter shape %s" % opr.inputs[1].shape[1:])
      description.append("Padding: %s" % opr.params['padding'])
      description.append("Stride: [%d %d]" % (opr.params['stride_width'],
                                              opr.params['stride_height']))
      if (opr.params['dilation_width_factor'] != 1 or
              opr.params['dilation_height_factor'] != 1):
        description.append("Stride: [%d %d]" % (
          opr.params['dilation_width_factor'],
          opr.params['dilation_height_factor'])
        )
      if ('fused_activation_fn' in opr.params and
              opr.params['fused_activation_fn'] is not None):
        description.append('Activation fn: %s' %
                           opr.params['fused_activation_fn'])
      if len(opr.inputs) > 2:
        description.append("Bias added.")
    if opr.type in ["MaxPool", 'MinPool', 'AvgPool']:
      description.append('Stride: [%d %d]' % (opr.params['stride_width'],
                                              opr.params['stride_height']))
      description.append("Padding: %s" % opr.params['padding'])
      description.append('Kernel size: [%d %d]' % (opr.params['kernel_width'],
                                                   opr.params['kernel_height']))
    if opr.type == "MatMul":
      description.append('Weights %s' % opr.inputs[1].shape)
      if ('fused_activation_fn' in opr.params and
              opr.params['fused_activation_fn'] is not None):
        description.append('Activation fn: %s' %
                           opr.params['fused_activation_fn'])
      if len(opr.inputs) > 2:
        description.append("Bias added.")
    return description

  @staticmethod
  def make_table_multiline(lines):
    if not lines:
      return ""
    else:
      output = "\\vtop{"
      for line in lines:
        output += "\\hbox{\\strut %s}" % line
      output += "}"
      return output
