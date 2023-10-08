from dataclasses import dataclass
from functools import cached_property


@dataclass
class BBox:
    """
    Represents a bounding box with coordinates.

    :param xmin: The minimum x-coordinate.
    :type xmin: int
    :param ymin: The minimum y-coordinate.
    :type ymin: int
    :param xmax: The maximum x-coordinate.
    :type xmax: int
    :param ymax: The maximum y-coordinate.
    :type ymax: int
    :param _pad: The padding value (optional, defaults to 0).
    :type _pad: int
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    _pad: int = 0

    @cached_property
    def area(self) -> int:
        """
        Computes the area of the bounding box.

        :return: The area of the bounding box.
        :rtype: int
        """
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def expand(self, amount):
        """
        Expands the bounding box by the specified amount.

        :param amount: The amount by which to expand the bounding box.
        :type amount: int
        """

        self.xmin = self.xmin - amount
        self.ymin = self.ymin - amount
        self.xmax = self.xmax + amount
        self.ymax = self.ymax + amount

    @property
    def pad(self):
        """
        Gets the pad value.

        :return: The pad value.
        :rtype: int
        """

        return self._pad

    @pad.setter
    def pad(self, amount: int):
        self._pad = amount

    def __contains__(self, other: "BBox") -> bool:
        """
        Checks if at least 50% of the other bounding box is inside \
            the current bounding box.

        :param other: The other bounding box to check.
        :type other: BBox
        :return: True if at least 50% of the other bounding box is inside \
            the current bounding box, False otherwise.
        :rtype: bool
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
        """
        Computes the difference of two bounding boxes.

        :param other: The bounding box to subtract from this bounding box.
        :type other: BBox
        :return: The resulting bounding box after subtraction.
        :rtype: BBox
        """

        xmin = max(0, self.xmin - other.xmin + other.pad)
        ymin = max(0, self.ymin - other.ymin + other.pad)
        xmax = max(0, self.xmax - other.xmin + other.pad)
        ymax = max(0, self.ymax - other.ymin + other.pad)

        return BBox(xmin, ymin, xmax, ymax)
