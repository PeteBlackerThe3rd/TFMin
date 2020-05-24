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

    Simply utility to show the TFMin version, and a list of the operations
    this installation supports.
"""
import tf_min
import tf_min.exporter as tfm_ex


def main():
    """
    Entry point function. Prints the version of TFMin and lists all supported
    operations and their development status.
    :return:
    """

    print("-" * 80)
    print("  TFMin version %s" % tf_min.__version__)

    tfm_ex.Exporter.list_supported_operations()


if __name__ == '__main__':
    main()
