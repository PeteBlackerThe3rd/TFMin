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

    This module contains a specialised code generator which produces a model
    implementation which outputs timing information to stdout
"""
from .graph_c_gen import CodeGenerator


class TimingCodeGenerator(CodeGenerator):
    """
    TFMin ansi-c code generator object.
    """

    def __init__(self, graph, base_name="model_", prefix="", path=".",
                 clang_format=None, byte_order="@", batch_size=1,
                 fake_weights=None):
      """
      Create a CodeGenerator object for the give model and settings
      :param graph: tf_min.Graph, model to export
      :param base_name: String, prefix for all c itentifiers
      :param prefix: String,
      :param path: String, base path to generate code files in.
      :param clang_format: String, or None, the code style to format to.
      :param byte_order: String, struct compatable byte order string, used
                         when writing weight literals as 32 bit hex.
      :param batch_size: Int, the batch size to generate this model with.
      :param fake_weights: Int or None, if an integer then a fake weights
                           buffer of the given size will be used. Allows
                           testing of models on platforms without enough
                           memory to store a models weights.
      """
      CodeGenerator.__init__(self, graph=graph, base_name=base_name,
                             prefix=prefix, path=path, clang_format="google",
                             byte_order=byte_order, batch_size=1,
                             fake_weights=100000)

    def __call__(self, silent=False, debug=False):
      """
      Method which actually generates the c source files of the model and
      its weights.
      :return: True if export successful, False operation
      """
      # disable the use of debug print statements because that
      # will break the timing.
      CodeGenerator.__call__(self, silent=silent, debug=False)

    def get_preamble_dependencies(self):
      """
      Overide the preamble dependencies function to add time.h
      :return:
      """
      return {'time.h': True, 'stdio.h': True}

    def gen_model_fn_preamble(self):
      """
      Overide the function preable method to create the timing initialisation
      code.
      :return: String, c code to add.
      """
      return "clock_t timing_op, timing_start = clock();\n"

    def gen_opr_complete_code(self, op_idx):
      """
      Method used to generate a specific block of code after each operation
      has completed, used in child classes.
      :param op_idx: Sequence index of the operation completed
      :return: String, c code to add.
      """
      code = "timing_op = clock();\n"
      code += "printf(\"%d, %%20f\\n\", " \
              "(timing_op-(double)timing_start)/CLOCKS_PER_SEC);\n" % op_idx
      if op_idx == len(self.graph.op_sequence)-1:
        code += "printf(\"timing complete\\n\");\n"
      return code
