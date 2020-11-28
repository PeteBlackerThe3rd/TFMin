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

    This module contains a set of custom exception classes used by the
    TFMin framework.
"""


class TFMinException(Exception):
  """ Base class of all TFMin exceptions """
  pass


class TFMinImportError(TFMinException):
  """ Exception thrown when a model fails to import to a Graph object """
  pass


class TFMinInvalidObject(TFMinException):
  """ Exception thrown when an object fails self validation """
  pass


class TFMinLayerBuildFailed(TFMinException):
  """ Exception thrown when creating a Tensorflow layer using """
  pass


class TFMinDeploymentFailed(TFMinException):
  """ Exception thrown whenever a c deployment fails """
  pass


class TFMinDeploymentExecutionFailed(TFMinException):
  """
  Exception thrown whenever a test model deployment fails to execute or
  fails during execution
  """
  pass


class TFMinVerificationExecutionFailed(TFMinException):
  """
  Exception thrown whenever a graph verification could not be performed for
  some reason. Note this does not signal that the model failed verification,
  but rather that the verification itself could be performed.
  """
  pass
