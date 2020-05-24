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

    Insertion based memory allocator - Intermediate buffers are sequentially
    inserted into the memory pattern. Each insertion attempts to place at
    each possible gap in the pattern and the most optimal gap is chosen. Here
    the most optimal gap is defined as the one which increases peak memory
    the least, in the case that peak memory is identacle then the gap which
    results in the largest contiguous memory area is chosen.

    This memory optimisor can be run with or without diagonal memory
    optimisation, which overlaps input and output buffers for certain
    operations where it is safe to do so.
"""
import copy
from tf_min.mem_opt.base_optimiser import BaseMemoryOptimiser
from tf_min.mem_opt.memory_region import MemoryRegion


class InsertionMemOptimiser(BaseMemoryOptimiser):

  def __init__(self,
               op_list,
               buffer_list,
               alignment=4):
    super(InsertionMemOptimiser, self).__init__(op_list,
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

  def duplicate(self):

    new_op_list = []
    new_buffer_list = []

    for buf in self.buffer_list:
      new_buffer_list.append(copy.copy(buf))
    for op in self.op_list:
      new_op = copy.copy(op)
      new_op.refresh_buffers(new_buffer_list)
      new_op_list.append(new_op)

    return InsertionMemOptimiser(new_op_list,
                                 new_buffer_list,
                                 self.alignment)

  def calculate_lower_bound(self, diagonal=True):
    lower_bound = 0
    for i, op in enumerate(self.op_list):

      # find this operations input and outputs can be overlapped
      overlapped_buffers = []
      overlap_size = 0
      if diagonal and op.buffer_overlap is not None and \
              len(op.input_buffers) > 0 and \
              len(op.output_buffers) > 0 and op.input_buffers[0].final_use == i:
        overlapped_buffers.append(op.input_buffers[0])
        overlapped_buffers.append(op.output_buffers[0])
        overlap_size = op.buffer_overlap + op.input_buffers[0].size

      # collect buffers required for the duration of this operation
      req_bufs = []
      req_bufs_size = 0
      for b in self.buffer_list:
        if b.creation <= i <= b.final_use and b not in overlapped_buffers:
          req_bufs.append(b)
          req_bufs_size += b.size

      this_size = overlap_size + req_bufs_size
      print("%24s [%d] is %d : %d bufs taking %d bytes" %
            (op.type,
             i,
             this_size,
             len(req_bufs),
             req_bufs_size), end='')
      if len(overlapped_buffers) == 2:
        print(" and %d bufs overlapped by %s taking %d bytes "
              "(saving %d bytes)." %
              (len(overlapped_buffers),
               op.buffer_overlap,
               overlap_size,
               op.output_buffers[0].size - op.buffer_overlap))
      else:
        print("")

      if this_size > lower_bound:
        lower_bound = this_size

    if diagonal:
      print("Lower bound for this model using diagonal "
            "memory optimisation is %d bytes." % lower_bound)
    else:
      print("Lower bound for this model is %d bytes." % lower_bound)

    return lower_bound

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

  def merge_inplace_op_buffers(self):

    # for each buffer who's final use op is inplace clobber safe merge
    # it's first input and first output buffer into one.

    print("")
    debug = False

    merged = True
    while merged:

      merged = False

      for buf in self.buffer_list:
        op = self.op_list[buf.final_use]

        if debug:
          print("Checking buffer [%s] with final use in [%s:%s]" %
                (buf.name,
                 op._underlying_op.type,
                 op.name))

          print("inplace [%s] output len %d" %
                (op.inplace_clobber,
                 len(op.output_buffers)))

        if op.inplace_clobber and op.output_buffers != []:

          if debug:
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

      if debug:
        print("\n--Operation list after merging inplace clobbers--------")
        for op in self.op_list:
          print("%10s Operation [%s]:" % (op._underlying_op.type, op.name))
          for input in op.input_buffers:
            print("        -> Input buffer [%s]:" % input.name)
          for output in op.output_buffers:
            print("        <- Output buffer [%s]:" % output.name)
        print("----------")

  def evaluate_gap(self, buf_idx, in_scope_buffers, gap):

    # create a copy of this memory optimisation, insert buffer and
    # then evaluate the result.
    proposed_pattern = self.duplicate()
    proposed_pattern.insert_into_gap(buf_idx,
                                     in_scope_buffers,
                                     gap)
    peak_mem = proposed_pattern.required_memory()
    largest_gap = 0
    # TODO calculate gap correctly

    return [peak_mem, largest_gap]

  def re_locate_buffer(self, buffer):

    overlapping_buffers = []
    for buf in self.buffer_list:
      if buf != buffer and buf.overlaps(buffer):
        overlapping_buffers.append(buf)

    overlapping_buffers.sort(key=lambda b: b.offset)

    print("Relocating buffer [%s] found %d overlapping buffers" %
          (buffer.name, len(overlapping_buffers)))

    if not overlapping_buffers:
      buffer.offset = 0
      return

    # special case if there is enough free space at the start, use it
    gap_size = overlapping_buffers[0].offset
    if buffer.creation == overlapping_buffers[0].final_use and \
            buffer.overlap_offset is not None:
      print("re_locating buffer, found safe overlap at start [%d]" %
            buffer.overlap_offset)
      gap_size -= buffer.size - buffer.overlap_offset

    if gap_size >= buffer.size:
      buffer.offset = 0

    else:  # general case find a large enough gap and locate buffer there
      for i in range(len(overlapping_buffers)):
        buf = overlapping_buffers[i]
        if i+1 == len(overlapping_buffers):
          buffer.offset = buf.offset + buf.size
        else:
          next_buf = overlapping_buffers[i+1]
          gap_size = next_buf.offset - (buf.offset + buf.size)
          if buffer.creation == next_buf.final_use and \
                  buffer.overlap_offset is not None:
            print("re_locating buffer, found safe overlap of [%d]" %
                  buffer.overlap_offset)
            gap_size -= buffer.size - buffer.overlap_offset
          if gap_size >= buffer.size:
            buffer.offset = buf.offset + buf.size
            break

  def identify_buffers_on_right(self, seed_buffers, ignore_buf = None):

    for buf in self.buffer_list:
      buf.flag = False
    edge_buffers = seed_buffers

    while edge_buffers:
      first = edge_buffers[0]
      edge_buffers.remove(first)
      first.flag = True
      for buf in self.buffer_list:
        if not buf.flag and buf != ignore_buf and \
                buf.offset is not None and buf not in edge_buffers:
          if buf.offset >= first.offset and buf.overlaps(first):
            edge_buffers.append(buf)

    right_buffers = []
    for buf in self.buffer_list:
      if buf.flag:
        right_buffers.append(buf)
    return right_buffers

  def shuffle_left(self, buffer):

    print("## shuffing buffer [%s] size %d" % (buffer.name, buffer.size))

    # find set of available memory regions where this buffer is needed
    free_space = [MemoryRegion()]
    for buf in self.buffer_list:
      if buf != buffer and buf.offset is not None and buf.overlaps(buffer):
        start = buf.offset
        end = start + buf.size
        if buffer.creation == buf.final_use and buffer.overlap_offset is not None:
          start += buffer.size - buffer.overlap_offset
        if buf.creation == buffer.final_use and buf.overlap_offset is not None:
          end -= buf.size - buf.overlap_offset
        used_space = MemoryRegion(start, end)
        new_free_space = []
        for space in free_space:
          new_free_space += space.get_carve_result(used_space)
        free_space = new_free_space
        print("@ subtracted buffer [%d - %s] (%s)" % (start, end, buf.name))
        for gap in free_space:
          print("@@ [%d - %s]" % (gap.start, gap.end))

    print("===== Found %d free regions." % len(free_space))
    free_space.sort(key=lambda r: r.start)
    for gap in free_space:
      size = None
      if gap.end is not None:
        size = gap.end - gap.start
      print("Free region [%d - %s] %s" % (gap.start, gap.end, size))

    # find the lowest free gap large enough for this buffe and place it
    buffer_region = MemoryRegion(buffer.offset, buffer.offset + buffer.size)
    for gap in free_space:
      if buffer_region.can_fit_inside(gap):
        buffer.offset = gap.start
        break

    #pass
    #TODO make this. . .

  def insert_into_gap(self, buf_idx, in_scope_buffers, gap):
    new_buffer = self.buffer_list[buf_idx]

    print(" ---> Inserting buffer %s in gap %d (%s)" %
          (new_buffer.name,
           gap,
           new_buffer.overlap_offset))
    for i, s_idx in enumerate(in_scope_buffers):
      scope_buf = self.buffer_list[s_idx]
      print(" ---# [%d] in scope %s [%d %d]" %
            (i,
             scope_buf.name,
             scope_buf.offset,
             scope_buf.size))

    # set offset of buffer being placed
    new_offset = 0
    if gap > 0:
      preceeding_buffer = self.buffer_list[in_scope_buffers[gap-1]]
      new_offset = preceeding_buffer.offset + preceeding_buffer.size
    new_buffer.offset = new_offset

    #print(" ---@ new offset %s" % new_offset)

    # create a list of buffers which are directly overlapping with new one
    overlapping_buffers = []
    for buf in self.buffer_list:
      if buf != new_buffer and buf.offset is not None and \
        buf.overlaps_time_mem(new_buffer):
        overlapping_buffers.append(buf)

    #print("Found %d directly overlapping buffers" % len(overlapping_buffers))

    if not overlapping_buffers:
      return

    max_shift = 0
    for buf in overlapping_buffers:
      shift = (new_buffer.offset + new_buffer.size) - buf.offset
      #print("-> Found a normal shift of %d" % shift)
      if new_buffer.creation == buf.final_use and \
              new_buffer.overlap_offset is not None:
        shift -= new_buffer.size - new_buffer.overlap_offset
        #print("-> Reduced shift to %d due to safe overlap of" %
        #      shift)
      if shift > max_shift:
        max_shift = shift

    #print("Found a required shift of %dbytes to clear the overlap" % max_shift)

    # add buffers directly touching on the higher side of these buffers
    overlapping_buffers = self.identify_buffers_on_right(overlapping_buffers,
                                                         new_buffer)

    for buf in overlapping_buffers:
      buf.offset += max_shift

    # shuffle buffers left again as far as possible
    # the order of this shuffling is in offset order with reverse execution
    # order to settle ties.
    def sort_fn(buf):
      sort_value = buf.offset
      fraction = buf.final_use / len(self.buffer_list)
      sort_value += 1 - fraction
      return sort_value

    overlapping_buffers.sort(key=sort_fn)
    for buf in overlapping_buffers:
      old_offset = buf.offset
      self.shuffle_left(buf)
      shift = buf.offset - old_offset
      print("~~{ shuffed buffer %d bytes [%s]" %
            (shift, buf.name))

  def insert_op_buffer(self, op, idx):
    """
    Method to add the output buffer of the given operation to the pre-allocation
    pattern. potentially shifting already allocated buffers to make space.
    If the operation's output buffer is already allocated, in the case of
    inplace reference ops such as reshape or slice then no action is performed.
    :param op: operation to add
    :param idx: index of this operation in the execution order.
    :return:
    """

    # position movement debugging
    for buf in self.buffer_list:
      if buf.offset is not None:
        buf.old_offset = buf.offset

    # if there are no output buffer or the output has already been allocated
    if len(op.output_buffers) == 0 or op.output_buffers[0].creation != idx:
      return
    new_buffer = op.output_buffers[0]

    print("Allocating buffer %s size [%d]" %
          (new_buffer.name,
           new_buffer.size))

    # get list of allocated blocks who's scopes overlap with this buffer
    in_scope_buffers = []
    for i, b in enumerate(self.buffer_list):
      if b.offset is not None:
        if b.creation <= idx <= b.final_use:
          # if b.creation <= idx and b.final_use >= idx:
          in_scope_buffers.append(i)
          print(" ** scope [%s, %d, %d]" %
                (b.name,
                 b.offset,
                 b.size))

    # if there were no overlapping buffers allocate this buffer at zero
    if not in_scope_buffers:
      new_buffer.offset = 0

    # sort buffers into ascending order of memory offset
    in_scope_buffers.sort(key=lambda buf: self.buffer_list[buf].offset)

    # identify possible buffer overlap
    new_buffer.overlap_buffer_idx = None
    new_buffer.overlap_offset = None
    if op.buffer_overlap is not None and len(op.input_buffers) > 0 and \
            op.input_buffers[0].final_use == idx:
      new_buffer.overlap_buffer_idx = \
        self.get_buffer_idx_by_name(op.input_buffers[0].name)
      new_buffer.overlap_offset = op.buffer_overlap
      print(" ~~~ Found a safe offset of %d bytes test (%s)" %
            (new_buffer.overlap_offset,
             op.output_buffers[0].overlap_offset))

    # evaluate each of the possible N + 1 gaps
    buf_idx_to_add = self.get_buffer_idx_by_name(new_buffer.name)
    peak_mems = []
    largest_contiguous_gaps = []
    for gap in range(len(in_scope_buffers) + 1):
      [peak_mem, contiguous_gap] = self.evaluate_gap(buf_idx_to_add,
                                                     in_scope_buffers,
                                                     gap)
      print(" -- [%d, %d]" % (peak_mem, contiguous_gap))
      peak_mems.append(peak_mem)
      largest_contiguous_gaps.append(contiguous_gap)

    # find optimal gap
    lowest_peak_mem = min(peak_mems)
    print(" == lowest peak mem [%d]" % lowest_peak_mem)
    largest_contiguous_gap = None
    optimal_gap = None
    for i in range(len(peak_mems)):
      if peak_mems[i] == lowest_peak_mem:
        if largest_contiguous_gap is None or \
                largest_contiguous_gap < largest_contiguous_gaps[i]:
          largest_contiguous_gap = largest_contiguous_gaps[i]
          optimal_gap = i

    print(" ++ optimal gap index %s" % optimal_gap)

    # add buffer in optimal gap and shift buffer as needed
    self.insert_into_gap(buf_idx_to_add,
                         in_scope_buffers,
                         optimal_gap)

    filename = "allocation_step_%04d.png" % self.img_count
    self.img_count += 1
    self.save_memory_pattern(filename,
                             buf_high=[self.get_buffer_idx_by_name(new_buffer.name)])

    print("Allocated buffer [%s] at offset %s" %
          (new_buffer.name,
           new_buffer.offset))

  def optimise(self):
    """
    Overriden method to optimise buffer memory locations using a simple
    heap based method.
    :return:
    """

    # sort buffers and calculate buffer scopes
    self.sort_and_update_buffer_scope()

    self.calculate_lower_bound(diagonal=True)

    self.img_count = 0

    # insert buffers in reverse sequence of execution
    def sort_fn(op):
      if op.output_buffers:
        return 0-op.output_buffers[0].final_use
      else:
        return 0-(len(self.op_list)+1)

    """reordered_ops = copy.copy(self.op_list)
    reordered_ops.sort(key=sort_fn)
    for op in reordered_ops:
      if (op.output_buffers):
        idx = self.get_buffer_idx_by_name(op.output_buffers[0].name)
        self.insert_op_buffer(op, idx)"""

    for idx, op in enumerate(self.op_list):
      self.insert_op_buffer(op, idx)

    # self.save_memory_pattern("mem_pattern.png")

    # return super(HeapAllocator, self).required_memory()
    return self.required_memory()
