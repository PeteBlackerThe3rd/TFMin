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

    This module contains the base operation kernel for generating ansi-c
    code for the v2 architecture.
"""
import tf_min.graph as tg


class BaseOpKernel:

    def __init__(self, operation):
        """
        create an instance of this op kernel which will generate the code
        for the given operation instance
        :param operation: tf_min.Operation object
        """
        assert self.matches(operation), "Error, cannot instantiate %s op " \
                                        "kernel from a %s operation" % (
            type(self),
            operation.type
        )

        self.operation = operation

    @staticmethod
    def matches(operation):
        print("Error call to matches on base class with op [%s]!" %
              operation.type)
        return False

    @staticmethod
    def description():
        return "Error description called on base class!"

    @staticmethod
    def status():
        """
        Development status of this op kernel.
        Either development, testing, production, or base.
        :return: String
        """
        return "base"

    def generate(self, batch_size=1):
        """
        Overridable method to generate the ansi-c code of this operation.
        :return: String,
        """
        return "/* generate called on BaseOpKernel! */\n"

    @staticmethod
    def process_template(template, values):
        """

        :param template:
        :param values:
        :return:
        """
        output = template
        for key, value in values.items():
            output = output.replace(key, str(value))
        return output
