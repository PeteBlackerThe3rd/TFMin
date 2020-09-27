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

  HEADER = '''\\begin{table}[h]
\caption{<caption>}
\label{<label>}
\centering
\\begin{tabular}{|c||c|c|c|}
\hline
\\bfseries Layer & \\bfseries Size & \\bfseries Data Type & 
\\bfseries Details \\\\
\hline\hline
'''

  FOOTER = '''\end{tabular}
\end{table}
'''

  def __init__(self, graph):
    self.graph = graph

  def generate_latex(self, caption="TFMin model", label=""):
    """
    Function to convert the model graph as latex. Note this only works nicely
    with purely sequential graphs. It'll produce output from any graph,
    but it will be really messy if it's a highly connected graph!
    :return: String, containing the a latex table containing a detailed
             description of the graph
    """

    self.graph.sequence_ops()
    latex = LatexWriter.HEADER
    latex = latex.replace("<caption>", caption)
    latex = latex.replace("<label>", label)

    # Create one initial row for each input tensor
    inputs = self.graph.get_inputs()
    for idx, input in enumerate(inputs):
      if len(inputs) == 1:
        label = "Input"
      else:
        label = "Input %d" % idx
      latex += "%s & %s & %s & %s \\\\\n" % (
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
      latex += "%s & %s & %s & %s \\\\\n" % (
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
      latex += "%s & %s & %s & %s \\\\\n" % (
        label,
        output.shape,
        get_dtype_description(output.d_type),
        ""
      )
      latex += '\hline\n'

    latex += LatexWriter.FOOTER

    return latex

  @staticmethod
  def generate_opr_description(opr):
    description = []
    if opr.type in ["Conv2D", "TransposedConv2D"]:
      description.append("Filter shape %s" % opr.inputs[1].shape[1:])
      if len(opr.inputs) > 2:
        description.append("Biases used.")
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
    if opr.type in ["MaxPool", 'MinPool', 'AvgPool']:
      description.append('Stride: [%d %d]' % (opr.params['stride_width'],
                                              opr.params['stride_height']))
      description.append("Padding: %s" % opr.params['padding'])
      description.append('Kernel size: [%d %d]' % (opr.params['kernel_width'],
                                                   opr.params['kernel_height']))
    if opr.type == "MatMul":
      if len(opr.inputs) > 2:
        description.append("Biases used.")
      if ('fused_activation_fn' in opr.params and
              opr.params['fused_activation_fn'] is not None):
        description.append('Activation fn: %s' %
                           opr.params['fused_activation_fn'])
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
