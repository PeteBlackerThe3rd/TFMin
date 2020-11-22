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

    Helper module to display a pretty progress bar on the command line
"""


def update_progress_bar(progress, pre_msg='', post_msg='', size=40,
                        show_times=False, c_return='\r'):
    """Function to display a pretty progress bar on the command line,
    along with a message and optional elapsed and estimated time
    remaining."""

    progress_string = '%s [\033[92m' % pre_msg
    for i in range(size):
        if progress * size > i:
            progress_string = '%s#' % progress_string
        elif progress * size > i-0.25:
            progress_string = '%s=' % progress_string
        elif progress * size > i-0.5:
            progress_string = '%s~' % progress_string
        elif progress * size > i-0.75:
            progress_string = '%s-' % progress_string
        else:
            progress_string = '%s ' % progress_string
    progress_string = '%s\033[0m] %6.2f%% %s' % (progress_string,
                                                 progress*100, post_msg)

    if show_times:
        progress_string = '%s' % progress_string

    print(progress_string, end=c_return, flush=True)


def finish_progress_bar(progress):

    update_progress_bar(progress, c_return='\n')
