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

    Base memory optimisation object of the TFMin library

    This object is used to derive all concrete memory optimisation algorithms
    used to shuffle operation order and allocate buffer locations reducing
    the amount of memory required.
"""
import tensorflow as tf
import numpy as np
import pprint as pp
import copy
# import cv2
from tf_min.op_kernels import *


class Operation:

  def __init__(self, object, buffer_list):

    self._underlying_op = object

    if type(object) == tf.Operation:
      self.name = object.name
      self.type = object.type
      self.input_buffers = []
      self.output_buffers = []
      self.refresh_buffers(buffer_list)

      # find op_kernel which matches this operation
      # to determine its memory requirements
      op_kernels = base_op.BaseOpKernel.__subclasses__()
      supported = False
      for k in op_kernels:
        if k.matches(object):
          supported = True

          # if the output tensor can be defined by mapping its contents
          # onto the memory space of one of the input tensors, then this
          # is the index of the input tensor. None in all other cases
          self.inplace_reference = k.can_inplace_reference()

          # if this calculation can be performed inplace, so the output is
          # written back to one of the input tensors. This is either a list
          # of input indices which can be used or None if no in-place
          # memory use can be used.
          self.inplace_clobber = k.can_inplace_clobber()

          # get size of safe buffer overlap in bytes.
          self.buffer_overlap = k.get_safe_overlap(object)

      assert supported, ("Error, Unsupported operation type [%s] passed"
                         "to mem_opt Operation." % object.type)

    else:
      assert False, ("Error cannot create Operation object from an "
                     "unsupported type of %s. Must be a tf.Operation object"
                     % str(type(object)))

  def refresh_buffers(self, buffer_list):
    # create input list of references to objects in buffer_list
    self.input_buffers = []
    for input_tensor in self._underlying_op.inputs:
      if input_tensor.op.type == "Identity":
        input_tensor = input_tensor.op.inputs[0]
      # found = False
      for buffer in buffer_list:
        if buffer.name == input_tensor.name:
          # found = True
          self.input_buffers += [buffer]

    # create output list of references to objects in buffer_list
    self.output_buffers = []
    for output_tensor in self._underlying_op.outputs:
      # found = False
      for buffer in buffer_list:
        if buffer.name == output_tensor.name:
          # found = True
          self.output_buffers += [buffer]


class Buffer:

  def __init__(self, object):

    self._underlying_buffer = object
    self.overlap_offset = None

    if type(object) == tf.Tensor:
      # get details of tensorflow Tensor buffer
      self.name = object.name

      dims = object.shape.as_list()
      dims = np.array(list(filter(None, dims)))

      # if this is a scalar
      if len(dims) == 0:
        self.size = object.dtype.size
      else:  # if this is a tensor
        self.size = object.dtype.size * dims.prod()

      # init creating op and final_use op
      # this are populated using the ____() method
      self.creation = None
      self.final_use = None

      # buffer needs to be allocated by default
      # unused buffers or compiled in constants will set this to False
      self.allocation_needed = True
      self.offset = None
      self.old_offset = None
    else:
      assert False, ("Error cannot create Buffer object from an "
                     "unsupported type of %s. Must be a tf.Tensor object"
                     % str(type(object)))

  def allocated(self):
    return self.offset is not None

  def overlaps(self, buffer):
    return buffer.final_use >= self.creation and \
           buffer.creation <= self.final_use

  def overlaps_time_mem(self, buffer):

    if buffer.final_use < self.creation or \
          buffer.creation > self.final_use:
      return False

    # Sort buffers by execution order
    top = self
    bottom = buffer
    if bottom.creation < top.creation:
      top = buffer
      bottom = self

    top_start = top.offset
    top_end = top.offset + top.size

    # if an overlap is possible reduce the size of `top`
    if (top.final_use == bottom.creation and
            bottom.overlap_offset is not None):
      top_start += bottom.size - bottom.overlap_offset

    bottom_start = bottom.offset
    bottom_end = bottom.offset + bottom.size

    return max(top_start, bottom_start) < min(top_end, bottom_end)


class BaseMemoryOptimiser:

  def __init__(self, op_list, buffer_list, alignment=4):
    self.op_list = op_list
    self.buffer_list = buffer_list
    self.alignment = alignment

    self.count = 0

    """print("BaseMemoryOptimiser constructor being called.\nOperations:")
    for op in self.op_list:
      print("[%s] %s : overlap (%s)" %
            (op.type,
             op.name,
             op.buffer_overlap))"""

  @staticmethod
  def name():
    return "BaseOptimiser"

  @staticmethod
  def description():
    return "Error description called on base optimser class!"

  # @classmethod
  def optimise(self):
    print("Error cannot call optimise on base class!")
    return 0

  def get_buffer_idx_by_name(self, name):
    for i, b in enumerate(self.buffer_list):
      if b.name == name:
        return i

    assert False, "Error failed to lookup buffer with name [%s]" % name

  def get_buf_creating_op(self, buf):
    for op in self.op_list:
      if buf in op.output_buffers:
        return op

    assert False, "Error, couldn't find creating operation of given buffer."

  def alpha_poly(self, img, points, color, alpha):
    points_np = np.array([points], 'int32')
    color_np = np.array([color], np.uint8)

    range_min = points[0]
    range_max = points[0]
    for point in points:
      range_min = np.minimum(range_min, point)
      range_max = np.maximum(range_max, point)
    range_size = range_max - range_min

    for point in points_np:
      point -= range_min

    alpha_buf = np.zeros([range_size[1], range_size[0], 3], np.float32)
    cv2.fillPoly(alpha_buf,
                 points_np,
                 [0.2, 0.2, 0.2],
                 4)

    img_roi = img[range_min[1]:range_max[1],
                  range_min[0]:range_max[0],
                  :]

    img[range_min[1]:range_max[1],
        range_min[0]:range_max[0],
        :] = cv2.add(img_roi * (1.0-alpha_buf), color_np * alpha_buf)

    return img

  def save_memory_pattern(self,
                          filename,
                          width=1000,
                          row_height=30,
                          buf_high=[]):

    # generate empty zebra pattern
    """img = np.zeros((row_height * len(self.op_list), width, 3), np.uint8)
    #yel_buf = np.ones((row_height * len(self.op_list), width, 3), np.uint8)
    #yel_buf[:, :, 0] = 0
    #yel_buf[:, :, 1] = 255
    #alpha_buf = np.zeros((row_height * len(self.op_list), width, 1),
                          np.float32)

    for i in range(len(self.op_list)):
      if i%2 == 0:
        cv2.rectangle(img,
                      (0, i*row_height),
                      (width, (i+1)*row_height),
                      (255, 255, 255), -1)
      else:
        cv2.rectangle(img,
                      (0, i*row_height),
                      (width, (i+1)*row_height),
                      (235, 235, 235), -1)

    bytes_to_pixel = float(width) / self.required_memory()

    # Add allocated buffers to the image
    for idx, b in enumerate(self.buffer_list):
      if b.offset is not None and b.allocation_needed:
        top = b.creation * row_height
        bottom = (b.final_use+1) * row_height
        left = int(b.offset * bytes_to_pixel)
        right = int((b.offset+b.size) * bytes_to_pixel)

        if idx in buf_high:
          color = (100, 100, 255)
        else:
          color = (100, 235, 100)

        overlap_ratio = 0
        if b.overlap_offset is not None:
          overlap_ratio = (b.size - b.overlap_offset) / b.size

        lower_overlap_ratio = 0
        for dep_buf in self.buffer_list:
          creating_op = self.op_list[b.final_use]
          if dep_buf.creation == b.final_use and \
                  creating_op.buffer_overlap is not None:
            overlap = dep_buf.size - creating_op.buffer_overlap
            lower_overlap_ratio = overlap / b.size

        horz_chamfer = left * overlap_ratio + right * (1-overlap_ratio)
        # vert_chamfer = top + (row_height * overlap_ratio)
        lower_chamfer = left*(1-lower_overlap_ratio) + right*lower_overlap_ratio

        points = [(left, top),
                  (int(horz_chamfer), top),
                  (right, top + row_height),
                  (right, bottom),
                  (int(lower_chamfer), bottom),
                  (left, bottom - row_height),
                  (left, top)]

        img = self.alpha_poly(img, points, (0, 255, 255), 0.25)

        for i in range(len(points)-1):
          pt_a = points[i]
          pt_b = points[i+1]
          cv2.line(img, pt_a, pt_b, color, 2, 4)

        if b.old_offset is not None and \
              b.old_offset != b.offset:
          centre = (int((left + right) / 2),
                    int((top + bottom) / 2))
          shift = b.offset-b.old_offset
          shift_pixels = int(shift * bytes_to_pixel)
          old_centre = (centre[0] - shift_pixels,
                        centre[1])

          print("shift [%d %d] [%d %d]" %
                (centre[0], centre[1],
                 old_centre[0], old_centre[1]))

          cv2.line(img,
                   old_centre,
                   centre,
                   (0, 0, 255),
                   2, 4)

    cv2.imwrite(filename, img)"""

  def swap_buffers_in_ops(self, old_buffer, new_buffer=None):
    """
    Method to swap all references to a buffer in the operation list from
    the old one to the new one. NOTE if the new buffer is none, then the
    old buffer is simply removed.

    :param old_buffer:
    :param new_buffer:
    :return: None
    """

    if new_buffer is None:
      for op in self.op_list:
        try:
          op.input_buffers.remove(old_buffer)
        except ValueError:
          pass
        try:
          op.output_buffers.remove(old_buffer)
        except ValueError:
          pass

    else:
      for op in self.op_list:
        op.input_buffers = [new_buffer if x == old_buffer else x for x in op.input_buffers]
        op.output_buffers = [new_buffer if x == old_buffer else x for x in op.output_buffers]

  def sort_and_update_buffer_scope(self):
    """
    Method to update the creation and final_use ops of all buffers
    and sort them in ascending order of creation op
    :return: nothing
    """

    debug = False

    if debug:
      print("----------")
      for op in self.op_list:
        print("Operation [%s]:" % op.name)
        for output in op.output_buffers:
          print("-- Output buffer [%s]:" % output.name)
      print("----------")

    # merge buffers created by operations which can store their results
    # as references to previous buffers (inplace_reference). If the buffer
    # preceding the operation is an input then the outut buffer can be removed.
    # new_buffer_list = []
    for op in self.op_list:
      if op.inplace_reference and op.output_buffers != []:

        if debug:
          print("Removing buffer [%s] from inplace operation %s(%s)" %
                (op.output_buffers[0].name,
                 op.name,
                 op._underlying_op.type))

        # the outputs from this operation will not need thier own buffer
        buffer_to_remove = op.output_buffers[0]
        try:
          self.buffer_list.remove(buffer_to_remove)
        except ValueError:
          print("Failed to remove buffer [%s] from buffer list!" %
                buffer_to_remove.name)

        # if there are no input buffers (it's an input) remove buffer
        if not op.input_buffers:
          if debug:
            print("Removing inplace reference buffer [%s]" %
                  buffer_to_remove.name)
          self.swap_buffers_in_ops(buffer_to_remove, None)
        else:  # if there is an input merge the output buffer with the input
          if debug:
            print("Replacing buffer [%s] with buffer [%s]" %
                  (buffer_to_remove.name,
                   op.input_buffers[0].name))
          self.swap_buffers_in_ops(buffer_to_remove, op.input_buffers[0])

        if debug:
          print("----------")
          for d_op in self.op_list:
            print("Operation [%s]:" % d_op.name)
            for input in d_op.input_buffers:
              print("-- Input buffer [%s]:" % input.name)
            for output in d_op.output_buffers:
              print("-- Output buffer [%s]:" % output.name)
          print("----------")

    if debug:
      print("Buffer list :")
      for b in self.buffer_list:
        print("[%s]" % b.name)

    # reset creation and final_use
    for buffer in self.buffer_list:
      buffer.creation = None
      buffer.final_use = None

    # update creation and final_use
    for i, op in enumerate(self.op_list):
      for input_buf in op.input_buffers:
        if input_buf.final_use is None or input_buf.final_use < i:
          input_buf.final_use = i
      for output_buf in op.output_buffers:
        if output_buf.creation is None or output_buf.creation > i:
          output_buf.creation = i

    if debug:
      print("--buffers---")
      for buf in self.buffer_list:
        print("[%s] (%s - %s)" %
              (buf._underlying_buffer.name,
               buf.creation,
               buf.final_use))

    # sort in ascending order of creation
    self.buffer_list.sort(key=lambda buf: buf.creation)

  def required_memory(self):
    """
    Returns the amount of memory required
    :return: required memory in bytes
    """
    upper_limit = 0
    for buffer in self.buffer_list:
      if buffer.allocated():
        upper_limit = max(upper_limit, buffer.offset + buffer.size)

    return upper_limit

  # @classmethod
  def optimise_memory(self):

    op_list_copy = self.op_list
    buffer_list_copy = self.buffer_list

    mem_size = self.optimise()

    # verify that all required tensors have been allocated
    for buffer in self.buffer_list:
      assert buffer.allocated(), ("Error, optimiser completed but buffer [%s] "
                                  "has not been allocated. Export failed!" %
                                  buffer.name)

    # TODO verify that the same set of operations are present

    # write memory offsets into extra members of the underlying objects
    # There may be more operations than buffers if some have been merged
    # so it is important here that we iterate through the operations and
    # set the address of each of their outputs
    for op in op_list_copy:
      for o_idx in range(len(op.output_buffers)):
        op._underlying_op.outputs[o_idx]._tfmin_memory_offset = op.output_buffers[o_idx].offset

    # for buffer in buffer_list_copy:
    #  buffer._underlying_buffer._tfmin_memory_offset = buffer.offset
    #  buffer._underlying_buffer._tfmin_memory_size = buffer.size

    return [mem_size, op_list_copy, buffer_list_copy]
