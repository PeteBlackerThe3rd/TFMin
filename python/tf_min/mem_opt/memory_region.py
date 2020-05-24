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

    MemoryRegion helper object, used by the memory optimisation sub system

    This object defines a region of memory which can be from zero to a
    finite or infitie end. The object supports the common set operations
    union, intersect, subtract
"""


# TODO rename operations and add union and intersect.


class MemoryRegion:
  def __init__(self, start=0, end=None):
    self.start = start
    self.end = end

  def get_carve_result(self, new_region):
    """
    returns a list of memory regions left over after this region has had the
    new region removed from it can return an empty list if the new region
    completely overlaps this one, a single region if it is clipped or two
    regions if this region is bisected
    :param new_region:
    :return: a list of remaining regions
    """

    # if there is no overlap return the original region
    if (self.end is not None and self.end <= new_region.start) or \
            (new_region.end is not None and new_region.end <= self.start):
      return [self]

    # if the new region overlaps completely with this one.
    if new_region.start <= self.start and \
            ((
                     new_region.end is None and self.end is None) or new_region.end is None or (
                     self.end is not None and new_region.end >= self.end)):
      # print("Carve returning empty set")
      return []

    # if the new region overlaps with the start of this region
    if new_region.start <= self.start and \
            (self.end is None or (new_region.end < self.end)):
      # print("Carve shortening the start of this region")
      return [MemoryRegion(new_region.end, self.end)]

    # if the new region overlaps with the end of this region
    if self.end is not None and new_region.start < self.end and new_region.end >= self.end:
      # print("Carve shortening the end of this region")
      return [MemoryRegion(self.start, new_region.start)]

    # The only option left now is that the new rigion bisects this one, so return the two parts
    # print("Carve bisecting this region")
    return [MemoryRegion(self.start, new_region.start),
            MemoryRegion(new_region.end, self.end)]

  def can_fit_inside(self, super_region):
    """
    returns true if this region can fit inside the super_region
    :param super_region:
    :return: boolean
    """
    # if the super region is infinite then this is always true
    if super_region.end is None:
      return True

    # this region is infinite and the super region isn't
    #  then this must be false
    if self.end is None:
      return False

    this_size = self.end - self.start
    super_size = super_region.end - super_region.start
    return this_size <= super_size