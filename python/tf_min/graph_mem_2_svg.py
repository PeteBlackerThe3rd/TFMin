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

    This module contains the SVGMemoryWriter object used to export annotated 
    SVG visualisation of the tensor buffer pre-allocations. 
"""
import svgwrite as svg
import tf_min.graph as tg


class SVGMemoryWriter:

    def __init__(self, graph):

        self.graph = graph
        self.dwg = None

        self.plot_width = 500
        self.row_height = 18  # 9
        self.plot_offset_x = 190
        self.plot_offset_y = 60
        self.op_labels_with = 200
        self.margin = 25
        self.label_length_limit = 20

    def draw_memory_layout(self):
      """
      Function to save the allocation buffer locations and their scopes to a
      vector svg file. This includes labels of operation scopes and the
      scale of the memory axis
      :param filename: name of te svg file to generate
      :return: True on success false otherwise
      """
      memory_size = self.graph.get_peak_memory()

      plot_width = self.plot_width
      row_height = self.row_height
      plot_offset_x = self.plot_offset_x
      plot_offset_y = self.plot_offset_y
      operation_count = len(self.graph.ops)

      if memory_size == 0:
          print("Error: Cannot save memory layout when no blocks "
                "are allocated")
          return

      # draw zebra background
      for i in range(operation_count + 1):
        if (i % 2) == 0:
          fill = svg.rgb(95 ,95 ,95 ,'%')
        else:
          fill = svg.rgb(90 ,90 ,90 ,'%')
        self.dwg.add(self.dwg.rect((20, plot_offset_y + i* row_height),
                         (plot_width + plot_offset_x,
                          row_height),
                         fill=fill))

      # add operation labels
      for i, opr in enumerate(self.graph.ops):
        label = "[%s] %s" % (opr.type, opr.label)
        if len(label) > self.label_length_limit:
          label = "..." + label[-(self.label_length_limit - 3):]
        self.dwg.add(self.dwg.text(label,
                         insert=(40, plot_offset_y + ((i + 1) * row_height) - 5),
                         fill='black'))

      # add axes and memory scale
      self.dwg.add(self.dwg.line((plot_offset_x - row_height,
                      plot_offset_y - row_height),
                     (plot_offset_x - row_height,
                      plot_offset_y + row_height * (operation_count + 2)),
                     stroke='black'))
      self.dwg.add(self.dwg.line((plot_offset_x - row_height,
                      plot_offset_y - row_height),
                     (plot_offset_x + plot_width + row_height,
                      plot_offset_y - row_height),
                     stroke='black'))
      self.dwg.add(self.dwg.line((plot_offset_x,
                      plot_offset_y - row_height),
                     (plot_offset_x,
                      plot_offset_y - 2 * row_height),
                     stroke='black'))
      self.dwg.add(self.dwg.line((plot_offset_x + plot_width,
                        plot_offset_y - row_height),
                       (plot_offset_x + plot_width,
                        plot_offset_y - 2 * row_height),
                       stroke='black'))

      self.dwg.add(self.dwg.text('0 KB',
                       insert=(plot_offset_x,
                               plot_offset_y - 2 * row_height),
                       fill='black'))
      self.dwg.add(self.dwg.text('%d KB' % int(memory_size / 1024),
                       insert=(plot_offset_x + plot_width,
                               plot_offset_y - 2 * row_height),
                       fill='black'))

      # draw allocated blocks
      for i, tensor in enumerate(self.graph.tensors):
        if tensor.allocated():
          # print("Rendering allocated block with mem_offset %d" % b.mem_offset)
          mem_start = (tensor.memory_offset * plot_width) / memory_size
          mem_width = (tensor.buffer_size * plot_width) / memory_size

          op_start = row_height * tensor.creation_idx
          op_range = row_height * (tensor.last_use_idx -
                                   tensor.creation_idx + 1)

          block_stroke = svg.rgb(0, 0, 0)
          block_fill = svg.rgb(255, 128, 128)
          if tensor.highlight_color is not None:
            block_fill = svg.rgb(tensor.highlight_color[0],
                                 tensor.highlight_color[1],
                                 tensor.highlight_color[2], '%')

          self.dwg.add(
            self.dwg.rect((plot_offset_x + mem_start, plot_offset_y + op_start),
                     (mem_width, op_range),
                     fill=block_fill,
                     stroke=block_stroke))

    def write(self, filename):
      """
      Function to save a visualisation of the tensor graph in a
      vector svg file
      :param filename: filename and path to save. svg will be added if needed
      :return: True on success, False on failure
      """
      self.dwg = svg.Drawing(filename,
                             size=(750,
                                   90 + self.row_height * len(self.graph.ops)),
                             profile='tiny')
      # TODO need to test for failure here

      self.draw_memory_layout()

      self.dwg.save()
      return True
