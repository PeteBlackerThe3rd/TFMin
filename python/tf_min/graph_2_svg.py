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

    This module contains the SVGWriter object used to save vector visualisation
    of the tensor graphs stored within TFMin.
"""
import svgwrite as svg
import tf_min.graph as tg


class SVGWriter:

    def __init__(self, graph):

        self.graph = graph
        self.dwg = None
        self.dwg_width = 1000
        self.dwg_height = 1000
        self.dwg_margin = 50

        self.op_locations = []
        self.tensor_locations = []

        self.tensor_height = 70
        self.tensor_width = 240
        self.tensor_fill = svg.rgb(85, 85, 100, '%')
        self.tensor_stroke = svg.rgb(50, 50, 80, '%')

        self.input_tensor_fill = svg.rgb(85, 100, 85, '%')
        self.input_tensor_stroke = svg.rgb(50, 80, 50, '%')

        self.output_tensor_fill = svg.rgb(100, 100, 85, '%')
        self.output_tensor_stroke = svg.rgb(80, 80, 50, '%')

        self.weights_tensor_fill = svg.rgb(100, 85, 100, '%')
        self.weights_tensor_stroke = svg.rgb(80, 50, 80, '%')

        self.op_height = 70
        self.op_width = 240
        self.op_fill = svg.rgb(90, 90, 90, '%')
        self.op_stroke = svg.rgb(80, 80, 40, '%')

        self.h_padding = 20
        self.v_padding = 30
        self.highlight_width = 10

        self.font_size = 16

        self.row_spacing = ((self.tensor_height + self.op_height) / 2 +
                            self.v_padding)

        self.max_label_len = 25

    def get_operation_height(self, opr):
        lines = 1 + len(opr.params)
        if opr.label is not None:
            lines += 1
        if opr.params:
            lines += 1
        return (lines * self.font_size) + 20

    def draw_operation_highlight(self, opr, pos):
        if opr.highlight_color is not None:
          op_height = self.get_operation_height(opr)
          (x, y) = pos
          highlight = svg.rgb(opr.highlight_color[0],
                              opr.highlight_color[1],
                              opr.highlight_color[2], '%')
          wid = self.highlight_width
          self.dwg.add(
            self.dwg.rect((x - (self.op_width / 2) - wid,
                           y - (op_height / 2) - wid),
                          (self.op_width + (wid * 2), op_height + (wid * 2)),
                          fill=highlight,
                          rx=self.font_size / 2 + wid,
                          ry=self.font_size / 2 + wid))
          if opr.sequence_index is not None:
              seq_x = x + (self.op_width / 2)
              seq_y = y - (op_height / 2)
              self.dwg.add(
                  self.dwg.rect((seq_x - self.font_size - wid,
                                 seq_y - self.font_size - wid),
                                (2 * self.font_size + (wid * 2),
                                 2 * self.font_size + (wid * 2)),
                                fill=highlight,
                                rx=self.font_size * 2 + wid,
                                ry=self.font_size * 2 + wid))

    def draw_operation(self, opr, pos):
        (x, y) = pos
        op_height = self.get_operation_height(opr)
        self.dwg.add(
            self.dwg.rect((x - (self.op_width/2),
                           y - (op_height/2)),
                          (self.op_width, op_height),
                          fill=self.op_fill,
                          rx=self.font_size/2,
                          ry=self.font_size/2,
                          stroke=self.op_stroke))

        line_y = y - (op_height/2) + 20
        self.dwg.add(self.dwg.text(opr.type,
                                   insert=(x, line_y),
                                   fill='black',
                                   text_anchor='middle'))

        if opr.label is not None:
            trunc_label = opr.label
            if len(trunc_label) > self.max_label_len:
                trunc_label = "..." + trunc_label[-(self.max_label_len-3):]
            line_y += self.font_size
            self.dwg.add(self.dwg.text(trunc_label,
                                       insert=(x, line_y),
                                       fill='black',
                                       text_anchor='middle'))

        if opr.params:
            line_y += self.font_size
            self.dwg.add(self.dwg.text("Parameters: . . .",
                                       insert=((x - (self.op_width/2) +
                                                self.font_size),
                                               line_y),
                                       fill='black'))
            for param, value in opr.params.items():
                line_y += self.font_size
                self.dwg.add(self.dwg.text("%s : %s" % (param, value),
                                           insert=((x - (self.op_width/2) +
                                                    self.font_size),
                                                   line_y),
                                           fill='black'))

        # draw sequence index if it exists
        if opr.sequence_index is not None:
            seq_x = x + (self.op_width / 2)
            seq_y = y - (op_height / 2)
            self.dwg.add(
                self.dwg.rect((seq_x - self.font_size,
                               seq_y - self.font_size),
                              (2 * self.font_size, 2 * self.font_size),
                              fill=svg.rgb(100, 100, 100, '%'),
                              stroke='black',
                              rx=self.font_size * 2,
                              ry=self.font_size * 2))
            self.dwg.add(self.dwg.text(str(opr.sequence_index),
                                       insert=(seq_x,
                                               seq_y + (self.font_size * 0.25)),
                                       fill='black',
                                       text_anchor='middle'))

    def draw_tensor_highlight(self, tensor, pos):
      if tensor.highlight_color is not None:
        (x, y) = pos
        wid = self.highlight_width
        self.dwg.add(
          self.dwg.rect((x - (self.tensor_width / 2) - wid,
                         y - (self.tensor_height / 2) - wid),
                        (self.tensor_width + (wid * 2),
                         self.tensor_height + (wid * 2)),
                        fill=svg.rgb(tensor.highlight_color[0],
                                     tensor.highlight_color[1],
                                     tensor.highlight_color[2], '%')))

    def draw_tensor(self, tensor, pos):
      (x, y) = pos
      fill = self.tensor_fill
      stroke = self.tensor_stroke
      if tensor.type == tg.TenType.INPUT:
          fill = self.input_tensor_fill
          stroke = self.input_tensor_stroke
      if tensor.type == tg.TenType.OUTPUT:
          fill = self.output_tensor_fill
          stroke = self.output_tensor_stroke
      if tensor.type == tg.TenType.CONSTANT:
          fill = self.weights_tensor_fill
          stroke = self.weights_tensor_stroke
      self.dwg.add(
        self.dwg.rect((x - (self.tensor_width / 2),
                       y - (self.tensor_height / 2)),
                      (self.tensor_width, self.tensor_height),
                      fill=fill,
                      stroke=stroke))

      trunc_label = tensor.label
      if len(trunc_label) > self.max_label_len:
          trunc_label = "..." + trunc_label[-(self.max_label_len-3):]

      self.dwg.add(self.dwg.text(trunc_label,
                                 insert=(x, y - self.font_size),
                                 fill='black',
                                 text_anchor='middle'))
      self.dwg.add(self.dwg.text(tensor.d_type,
                                 insert=((x - (self.tensor_width/2) +
                                          self.font_size),
                                         y),
                                 fill='black'))
      mem_str = ""
      if tensor.get_buffer_size(1) is not None:
          mem_str = " (%d bytes)" % tensor.get_buffer_size()
      self.dwg.add(self.dwg.text("Shape %s%s" % (tensor.shape, mem_str),
                                 insert=((x - (self.tensor_width/2) +
                                          self.font_size),
                                         y + self.font_size),
                                 fill='black'))

    def get_index_of_tensor(self, tensor):
        for i, g_tensor in enumerate(self.graph.tensors):
            if g_tensor == tensor:
                return i
        return None

    def get_index_of_op(self, opr):
        for i, g_opr in enumerate(self.graph.ops):
            if g_opr == opr:
                return i
        return None

    def tensor_placed(self, tensor):
        idx = self.get_index_of_tensor(tensor)
        return self.tensor_locations[idx] is not None

    def op_placed(self, opr):
        idx = self.get_index_of_op(opr)
        if idx is None:
            return False
        return self.op_locations[idx] is not None

    def place_row_of_tensors(self, tensors_to_add, y_pos):

        # compute the width of this row
        row_width = 0
        sub_tensors_present = False
        for i, tensor in enumerate(tensors_to_add):
            if i != 0:
                row_width += self.h_padding
            if tensor.meta_type == tg.TenMetaType.SINGLE:
                row_width += self.tensor_width
            elif tensor.meta_type == tg.TenMetaType.SUPER:
                sub_tensors_present = True
                sub_count = len(tensor.sub_tensors)
                row_width += sub_count * (self.tensor_width + self.h_padding)
                row_width -= self.h_padding
        x_start = 0 - (row_width / 2)
        op_heights = []
        for tensor in tensors_to_add:
            if tensor.creating_op is not None:
                op_heights.append(self.get_operation_height(tensor.creating_op))
            elif tensor.meta_type == tg.TenMetaType.SUPER:
                for sub_tensor in tensor.sub_tensors:
                    if sub_tensor.creating_op is not None:
                        op_heights.append(
                            self.get_operation_height(sub_tensor.creating_op))
        max_op_height = 0
        if op_heights:
            max_op_height = max(op_heights)
        op_row_spacing = (self.row_spacing + max_op_height) / 2 + self.v_padding
        if sub_tensors_present:
            #print("Adding row of tensors, super tensor present")
            op_row_spacing += self.row_spacing
        for i, tensor in enumerate(tensors_to_add):
            # x_pos = x_start + i * (self.op_width + self.h_padding)

            if tensor.meta_type == tg.TenMetaType.SINGLE:
                tensor_idx = self.get_index_of_tensor(tensor)
                y = y_pos
                if sub_tensors_present:
                  y -= self.tensor_height/2
                self.tensor_locations[tensor_idx] = (x_start, y)

                if tensor.creating_op is not None:
                  creating_op = tensor.creating_op
                  creating_idx = self.get_index_of_op(creating_op)
                  self.op_locations[creating_idx] = (x_start,
                                                     y_pos - op_row_spacing)

                x_start += self.tensor_width + self.h_padding

            elif tensor.meta_type == tg.TenMetaType.SUPER:
                    # if tensor doesn't have a creating op but it is a super
                    # tensor then it may have multiple sub-tensors with creating
                    # ops
                #print("Drawing super tensor with [%d] sub-tensors" % len(tensor.sub_tensors))
                # if this is a pre-super tensor where the super tensor is
                # populated first then sub-tensors read
                if tensor.creating_op is not None:
                    super_tensor_y = y_pos - self.tensor_height
                    sub_tensor_y = y_pos
                else:  # if this is a post-super tensor where the sub
                       # tensors are populated first
                    super_tensor_y = y_pos
                    sub_tensor_y = y_pos - self.tensor_height

                tensor_idx = self.get_index_of_tensor(tensor)
                super_tensor_x = (x_start +
                                  ((self.tensor_width + self.h_padding) *
                                   (len(tensor.sub_tensors)-1)) / 2)
                self.tensor_locations[tensor_idx] = (super_tensor_x,
                                                     super_tensor_y)

                if tensor.creating_op is not None:
                    creating_op = tensor.creating_op
                    creating_idx = self.get_index_of_op(creating_op)
                    self.op_locations[creating_idx] = (super_tensor_x,
                                                       y_pos - op_row_spacing)

                for sub_tensor in tensor.sub_tensors:
                  sub_tensor_idx = self.get_index_of_tensor(sub_tensor)
                  self.tensor_locations[sub_tensor_idx] = (
                    x_start, sub_tensor_y)

                  if sub_tensor.creating_op is not None:
                    creating_op = sub_tensor.creating_op
                    creating_idx = self.get_index_of_op(creating_op)
                    self.op_locations[creating_idx] = (x_start,
                                                       y_pos - op_row_spacing)

                  x_start += self.tensor_width + self.h_padding
            else:
                print("ignoring sub tensor passed to place_row_of_tensors")

        return 2 * op_row_spacing

    def place_blocks(self):
        """
        Function which returns two lists containing the plot locations of each
        operation and each tensor respectively
        :return:
        """
        self.op_locations = [None] * len(self.graph.ops)
        self.tensor_locations = [None] * len(self.graph.tensors)

        y_pos = 0

        # place output tensors initially
        outputs = self.graph.get_outputs()
        row_height = self.place_row_of_tensors(outputs, y_pos)
        y_pos -= row_height

        # Start with the outputs and place rows of tensors and operations
        # iteratively
        blocks_added = True
        while blocks_added:
            # Find any tensors which have not been placed but that all
            # consuming operations have been placed
            tensors_to_place = []
            for ten in self.graph.tensors:

                # ignore sub-tensors
                if ten.meta_type == tg.TenMetaType.SUB:
                    continue

                # if tensor has no outputs then ignore it
                deps = ten.dependent_ops.copy()
                if ten.meta_type == tg.TenMetaType.SUPER:
                    for sub_ten in ten.sub_tensors:
                        deps.extend(sub_ten.dependent_ops)

                if not deps:
                    #print("skipping tensor [%s] type [%s] with no deps" %
                    #      (ten.label,
                    #       ten.meta_type))
                    continue

                ready_to_place = not self.tensor_placed(ten)
                for output in deps:
                    if not self.op_placed(output):
                        ready_to_place = False
                if ready_to_place:
                    tensors_to_place.append(ten)

            #print("Placing %d tensors." % len(tensors_to_place))
            #for ten in tensors_to_place:
            #  print("-> %s [%s]" % (ten.label, ten.meta_type))

            row_height = self.place_row_of_tensors(tensors_to_place, y_pos)
            y_pos -= row_height
            blocks_added = tensors_to_place  # using boolean value of non
            #                                  empty list.

        # find range of blocks and centre in a document
        min_x = None
        max_x = None
        min_y = None
        max_y = None
        for idx, tensor in enumerate(self.graph.tensors):
            if self.tensor_locations[idx] is not None:
                if min_x is None:
                    min_x = (self.tensor_locations[idx][0] -
                             self.tensor_width / 2)
                    max_x = (self.tensor_locations[idx][0] +
                             self.tensor_width / 2)
                    min_y = (self.tensor_locations[idx][1] -
                             self.tensor_height / 2)
                    max_y = (self.tensor_locations[idx][1] +
                             self.tensor_height / 2)
                else:
                    min_x = min(min_x, (self.tensor_locations[idx][0] -
                                        self.tensor_width / 2))
                    max_x = max(max_x, (self.tensor_locations[idx][0] +
                                        self.tensor_width / 2))
                    min_y = min(min_y, (self.tensor_locations[idx][1] -
                                        self.tensor_height / 2))
                    max_y = max(max_y, (self.tensor_locations[idx][1] +
                                        self.tensor_height / 2))
        # print("Initial placed range:")
        # print("X [%f - %f]" % (min_x, max_x))
        # print("Y [%f - %f]" % (min_y, max_y))
        # set drawing size and move tensors and op into final locations
        self.dwg_width = max_x - min_x + (2 * self.dwg_margin)
        self.dwg_height = max_y - min_y + (2 * self.dwg_margin)
        for i, loc in enumerate(self.tensor_locations):
            if loc is not None:
                loc = list(loc)
                # loc[0] -= (min_x + max_x) / 2
                # loc[1] -= (min_y + max_y) / 2
                loc[0] -= min_x - self.dwg_margin
                loc[1] -= min_y - self.dwg_margin
                self.tensor_locations[i] = tuple(loc)
        for i, loc in enumerate(self.op_locations):
            if loc is not None:
                loc = list(loc)
                # loc[0] -= (min_x + max_x) / 2
                # loc[1] -= (min_y + max_y) / 2
                loc[0] -= min_x - self.dwg_margin
                loc[1] -= min_y - self.dwg_margin
                self.op_locations[i] = tuple(loc)

    def draw_tensor_to_op(self, tensor, opr):
        """
        marker-end:url(#Arrow2Lend)
        :param tensor:
        :param opr:
        :return:
        """
        tensor_idx = self.get_index_of_tensor(tensor)
        opr_idx = self.get_index_of_op(opr)
        if tensor_idx is None or opr_idx is None:
            print("Warning cannot draw tensor to op, one or both are None.")
            return
        if self.tensor_locations[tensor_idx] is None:
            print("Warning cannot draw connection, tensor not placed.")
            return
        if self.op_locations[opr_idx] is None:
            print("Warning cannot draw ocnnection, operation not placed.")
            return
        start = (self.tensor_locations[tensor_idx][0],
                 self.tensor_locations[tensor_idx][1] + self.tensor_height / 2)
        end = (self.op_locations[opr_idx][0],
               (self.op_locations[opr_idx][1] -
                self.get_operation_height(opr) / 2))

        self.dwg.add(self.dwg.line(start,
                                   end,
                                   stroke='black'))

    def draw_op_to_tensor(self, opr, tensor):
        """

        :param tensor:
        :param opr:
        :return:
        """
        tensor_idx = self.get_index_of_tensor(tensor)
        opr_idx = self.get_index_of_op(opr)
        if tensor_idx is None or opr_idx is None:
            print("Warning cannot draw tensor to op, one or both are None.")
            return
        if opr_idx is None:
            print("Error cannot get index of op [%s]" % opr.label)
            return

        if self.tensor_locations[tensor_idx] is None:
            print("Warning cannot draw connection, tensor not placed.")
            return
        if self.op_locations[opr_idx] is None:
            print("Warning cannot draw ocnnection, operation not placed.")
            return
        start = (self.op_locations[opr_idx][0],
                 (self.op_locations[opr_idx][1] +
                  self.get_operation_height(opr) / 2))
        end = (self.tensor_locations[tensor_idx][0],
               self.tensor_locations[tensor_idx][1] - self.tensor_height / 2)

        self.dwg.add(self.dwg.line(start,
                                   end,
                                   stroke='black'))

    def write(self, filename):
        """
        Function to save a visualisation of the tensor graph in a
        vector svg file
        :param filename: filename and path to save. svg will be added if needed
        :return: True on success, False on failure
        """
        self.place_blocks()
        self.dwg = svg.Drawing(filename,
                               size=(self.dwg_width, self.dwg_height),
                               profile='tiny')
        # TODO need to test for failure here

        # draw tensor and operation highlights
        for i, tensor in enumerate(self.graph.tensors):
            if self.tensor_locations[i] is not None:
                self.draw_tensor_highlight(tensor, self.tensor_locations[i])
        for i, opr in enumerate(self.graph.ops):
            if self.op_locations[i] is not None:
                self.draw_operation_highlight(opr, self.op_locations[i])

        # draw arcs between tensors and operations
        for tensor in self.graph.tensors:
            if tensor.creating_op is not None:
                self.draw_op_to_tensor(tensor.creating_op, tensor)
            for output in tensor.dependent_ops:
                self.draw_tensor_to_op(tensor, output)

        # draw tensors
        for i, tensor in enumerate(self.graph.tensors):
            if self.tensor_locations[i] is not None:
                self.draw_tensor(tensor, self.tensor_locations[i])

        # draw operations
        for i, opr in enumerate(self.graph.ops):
            if self.op_locations[i] is not None:
                self.draw_operation(opr, self.op_locations[i])

        self.dwg.save()
        return True
