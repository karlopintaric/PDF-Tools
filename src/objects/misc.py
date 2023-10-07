from dataclasses import dataclass
from functools import cached_property
from typing import Optional


@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    _pad: int = 0

    @cached_property
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def expand(self, amount):
        self.xmin = self.xmin - amount
        self.ymin = self.ymin - amount
        self.xmax = self.xmax + amount
        self.ymax = self.ymax + amount
    
    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, amount):
        self._pad = amount 

    def __contains__(self, other: "BBox") -> bool:
        """
        Check if at least 50% of the area of the other box is inside the current box.

        Parameters:
            other (BBox): The other bounding box to check against.

        Returns:
            bool: True if at least 50% of the area of the other box is inside the current box, False otherwise.
        """
        # Calculate the area of overlap
        overlap_width = max(
            0, min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        )
        overlap_height = max(
            0, min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        )
        area_overlap = overlap_width * overlap_height

        # Calculate half of the area of the other box
        half_other_area = 0.5 * other.area

        return area_overlap >= half_other_area

    def __sub__(self, other: "BBox") -> "BBox":

        xmin = max(0, self.xmin - other.xmin + other.pad)
        ymin = max(0, self.ymin - other.ymin + other.pad)
        xmax = max(0, self.xmax - other.xmin + other.pad)
        ymax = max(0, self.ymax - other.ymin + other.pad)

        return BBox(xmin, ymin, xmax, ymax)
