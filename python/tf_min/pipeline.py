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

    This module defines the Pipeline object which represents Graph
    Translation Pipelines. This object includes functions to build, save,
    load, and execute these pipelines.
"""
import os
import xml.dom.minidom as xmldom
import copy
import numpy as np
import operator
from tf_min.graph_translators.graph_translator import GraphTranslator
from tf_min.graph_translators import *
import tf_min.graph as tfm_g


# predefined graph translation pipelines defining some of the useful
# common options.
BuiltinPipelines = {"GreedyHeap": """<tfmin>
  <pipeline>
    <SequenceOps Execution="Greedy" />
    <HeapSmartOrder Order="Backwards" />
  </pipeline>
</tfmin>""",
                    "SimplifyTFGraph": """<tfmin>
  <pipeline>
    <RemoveIdentityOps />
    <FuseBiasAdds />
    <FuseActivations />
    <RemoveReshapeOps />
  </pipeline>
</tfmin>"""}


class Pipeline:
    """
    The Graph Translation Pipeline object, this includes a sequence of
    graph translators and their paraemeters.
    """

    def __init__(self, filename=None, builtin=None):
        """
        Creates a graph translation pipeline. Either blank of from the
        given XML file, or builtin constant.
        :param filename: String, name of a pipeline XML file to load
        :param builtin: String, Key into BuiltinPipelines defining a
                        builtin pipeline to instantiate.
        """
        self.pipeline = []
        self.code_gen = None

        assert not(filename is not None and builtin is not None), \
            "Error: Cannot provide both and XML filename and builtin " \
            "constant when creating a Pipeline object."

        if filename is not None:
            self.load_xml(filename)

        if builtin is not None:
            assert builtin in BuiltinPipelines.keys(), \
                "Error: Builtin pipeline given \"%s\" not found." % builtin
            self.from_xml(BuiltinPipelines[builtin])

    def __call__(self, input_graph, inplace=True):
      """
      Execute this pipeline on the given graph, either on a clone of the graph
      or inplace.
      :param input_graph: TFMin.Graph model
      :param inplace: Boolean
      :return:
      """
      if inplace:
        output_graph = input_graph
      else:
        output_graph = input_graph.clone()

      for tg in self.pipeline:
        tg(output_graph, inplace=True)

      if inplace:
        return
      else:
        return output_graph

    def load_xml_file(self, filename):
        """
        Method to clear this pipeline and load a new one from the given
        XML file.
        :param filename: String, name of the XML file to load
        :return: None
        """
        self.from_xml(open(filename, 'r').read())

    def from_xml(self, xml_str):
      """
      Method to clear this pipeline and read a new one from the given XML
      string
      :param xml_str: XML content to load
      :return:
      """
      doc = xmldom.parseString(xml_str)

      # get and verify base element
      base_node = doc.documentElement
      # print(doc.documentElement.childNodes)
      if base_node.tagName != "tfmin":
          print("base nodename [%s]" % base_node.tagName)
          print("type [%s]" % type(base_node.tagName))
          raise IOError("Error reading pipeline XML file, document "
                        "element is not a tfmin element.")

      # get and verify pipeline element
      pipeline_ele = None
      for child_node in base_node.childNodes:
          if (isinstance(child_node, xmldom.Element) and
                  child_node.tagName == "pipeline"):
              pipeline_ele = child_node
      if pipeline_ele is None:
          raise IOError("Error reading pipeline XML file, no pipeline "
                        "element found.")

      # generate graph translators for each step of the pipeline
      self.pipeline = []
      for step in pipeline_ele.childNodes:
          if isinstance(step, xmldom.Element):
              gt = self.get_graph_translator_from_name(step.nodeName)
              assert gt is not None, \
                  "Error reading pipeline XML file unrecognized " \
                  "Graph Translator \"%s\"" % step.nodeName
              self.pipeline.append(gt(step))

      # if a valid code_gen node exists then load it
      for ele in base_node.childNodes:
          if isinstance(ele, xmldom.Element) and ele.tagName == "code_gen":
              self.code_gen = dict(ele.attributes.items())
              assert "name_prefix" in self.code_gen, \
                  "Error reading pipeline XML, no name_prefix attribute in " \
                  "code_gen element "

    def save_xml(self, filename):
        """
        Method to write this pipeline to an XML file
        :param filename: String, name of XML file to create
        :return: None
        """
        doc = xmldom.Document()
        tfmin_ele = doc.createElement("tfmin")

        for step in self.pipeline:
            tfmin_ele.appendChild(step.to_xml(doc))
        if self.code_gen is not None:
            code_gen_ele = doc.createElement("code_gen")
            for key, value in self.code_gen.items():
                code_gen_ele.setAttribute(key, value)
            tfmin_ele.appendChild(code_gen_ele)

        doc.appendChild(tfmin_ele)
        doc.writexml(filename)

    def summary(self):
        """
        Method to generate a text summary of this pipeline
        :return: String
        """
        summary = "========\n"
        summary += "Graph Translation Pipeline with %d steps.\n" % \
            len(self.pipeline)
        summary += "========\n"
        for step in self.pipeline:
            summary += "%s\n%s\n" % (step.get_type(),
                                     step.DESCRIPTION)
            for key, value in step.parameters.items():
                summary += "[%s] %s\n" % (key, value)
            if step.summary != "":
                summary += "----\n%s\n" % step.summary
            summary += "========\n"
        if self.code_gen is not None:
            summary += "Code generation defined with parameters:\n"
            for key, value in self.code_gen.items():
                summary += "[%s] %s\n" % (key, value)
            summary += "========\n"
        return summary

    @staticmethod
    def get_graph_translator_from_name(name):
        """
        Search through all descendants of the GraphTranslator object
        looking for one with a name which matches 'name'
        :param name: String, the name of GraphTranslator to search for
        :return: None of not found or the class type of the GraphTranslator
                 if found.
        """
        for gt in GraphTranslator.__subclasses__():
            if gt.TYPE == name:
              return gt

        return None
