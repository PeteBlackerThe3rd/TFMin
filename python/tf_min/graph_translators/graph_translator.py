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

    This module defines the graph translator base object from which all
    graph translators are descended.
"""
import xml.dom as xmldom
import copy
from enum import Enum
import numpy as np
import operator
import tf_min.types as types


class GraphTranslator:
  """
  GraphTranslator functionoid base class, defines the mechanism for
  loading settings and serializing and deserializing to and from XML
  """

  DEFAULT_PARAMS = {}
  TYPE = 'BaseTranslator'
  DESCRIPTION = 'Base Graph Translator object, should not be' \
                'instantiated or used!'

  def __init__(self, source={}):
    """
    Constructor which configures this graph translator, either deserializing
    from XML or adding settings from a dictionary.
    :param source: Optional, either a dictionary of settings or an
                   xml.dom.Node to read settings from.
    """
    # If a dictionary was given then copy it to settings and merge with
    # and undefined default settings
    self.parameters = self.DEFAULT_PARAMS
    self.summary = ""

    # if a dictionary was passed to source then add this parameters
    if isinstance(source, dict):
      self.parameters.update(source)

    # if an XML node was passed to source then read attributes from the
    # XML element
    elif isinstance(source, xmldom.Node):
      self.parameters.update(dict(source.attributes.items()))

    # if any other type was passed to source then throw and assertion
    else:
      assert false, "Error cannot instantiate GraphTranslator from a " \
                    "\"%s\" type." % type(source)

  @classmethod
  def get_type(cls):
      return cls.TYPE  # "GraphTranslator"


  def to_xml(self, doc):
    """
    Function to create an XML node representing this translator and
    its current settings.
    :param doc: an xml.dom.Document object used to create this node
    :return:
    """
    xml_node = doc.createNode(self.get_type())
    for setting, value in self.settings:
      xml_node.setAttribute(setting, value)
    return xml_node

  def __call__(self, input_graph, inplace=False):
    """
    Overload making this object callable, base class implementation
    takes an input Graph clones it and returns it unchanged.
    :param input_graph:
    :return:
    """
    self.summary = "Base Graph Translator Object called. " \
                   "Returning unchanged graph."
    if inplace:
        return None
    else:
        return input_graph.clone()
