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

    HeapAllocator - simple heap allocation memory optimisation algorithm
    this isn't a very good algorithm but is used to generate an upper bound
    used when comparing other algorithms.
"""
from tf_min.mem_opt.base_optimiser import BaseMemoryOptimiser
from tf_min.mem_opt.memory_region import MemoryRegion


class HeapAllocator(BaseMemoryOptimiser):

  def __init__(self,
               op_list,
               buffer_list,
               alignment=4):
    super(HeapAllocator, self).__init__(op_list,
                                        buffer_list,
                                        alignment=alignment)
    self.img_count = 0

  @staticmethod
  def name():
    return "Heap Memory Allocator"

  @staticmethod
  def description():
    return "A simple memory pre-allocation althorithm which " \
           "uses a heap allocation strategy"

  def heap_allocate_buffer(self, buffer):
    """
    Add a single block to the allocated block pattern using
     a heap allocation method. I.e. the first free space.
    :param buffer: the buffer object to allocate
    :return: None
    """

    # test if this operation can be inplace using the input buffers
    # memory space

    # create a list of all free regions of memory around the
    # blocks currently allocated which overlap with this block
    free_regions = [MemoryRegion(0, None)]
    for b in self.buffer_list:
      if b.allocated() and b.overlaps(buffer):
        new_free_regions = []
        block_region = MemoryRegion(b.offset, b.offset + b.size)
        for region in free_regions:
          new_free_regions += region.get_carve_result(block_region)
        free_regions = new_free_regions

    # add this block to the first region it fits into
    new_block_region = MemoryRegion(0, buffer.size)
    for region in free_regions:
      if new_block_region.can_fit_inside(region):
        buffer.offset = region.start
        break

    filename = "allocation_step_%04d.png" % self.img_count
    self.img_count += 1
    self.save_memory_pattern(filename,
                             buf_high=[
                               self.get_buffer_idx_by_name(buffer.name)])

  def merge_inplace_op_buffers(self):

    # for each buffer who's final use op is inplace clobber safe merge
    # it's first input and first output buffer into one.

    print("")

    merged = True
    while merged:

      merged = False

      for buf in self.buffer_list:
        op = self.op_list[buf.final_use]

        print("Checking buffer [%s] with final use in [%s:%s]" %
              (buf.name,
               op._underlying_op.type,
               op.name))

        print("inplace [%s] output len %d" %
              (op.inplace_clobber,
               len(op.output_buffers)))

        if op.inplace_clobber and op.output_buffers != []:

          print("--- replacing buffer %s with %s" %
                (op.output_buffers[0],
                 op.input_buffers[0]))

          # update final use index
          buf.final_use = op.output_buffers[0].final_use

          # swap all references to the old output to the new extended input
          old_output_buffer = op.output_buffers[0]
          self.swap_buffers_in_ops(old_output_buffer, op.input_buffers[0])

          # remove old output buffer.
          self.swap_buffers_in_ops(old_output_buffer, None)
          self.buffer_list.remove(old_output_buffer)

          merged = True
          break

      print("\n--Operation list after merging inplace clobbers--------")
      for op in self.op_list:
        print("%10s Operation [%s]:" % (op._underlying_op.type, op.name))
        for input in op.input_buffers:
          print("        -> Input buffer [%s]:" % input.name)
        for output in op.output_buffers:
          print("        <- Output buffer [%s]:" % output.name)
      print("----------")

  def optimise(self):
    """
    Overriden method to optimise buffer memory locations using a simple
    heap based method.
    :return:
    """

    # sort buffers and calculate buffer scopes
    self.sort_and_update_buffer_scope()

    self.merge_inplace_op_buffers()

    # allocate buffers in order of creation using heap strategy
    for buffer in self.buffer_list:
      self.heap_allocate_buffer(buffer)

    # return super(HeapAllocator, self).required_memory()
    return self.required_memory()
