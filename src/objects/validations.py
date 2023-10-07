import io
from pathlib import Path
from typing import Union

import fitz
import numpy as np

PdfSource = Union[fitz.Document, str, io.BytesIO, bytes, Path]
ImageSource = Union[str, io.BytesIO, bytes, Path]


def is_image_array(arr: np.ndarray) -> bool:
    """
    Check if the input array is a valid image array.

    :param arr: The input array to be checked.
    :type arr: np.ndarray
    :return: True if the input is a valid image array, False otherwise.
    :rtype: bool
    """

    if not isinstance(arr, np.ndarray):
        return False

    if arr.ndim not in (2, 3):
        return False

    if arr.ndim == 3 and arr.shape[-1] not in (1, 3):
        return False

    if arr.dtype != np.uint8:
        return False

    return True


def is_jpg(data) -> bool:
    """
    Check if the input data represents a JPG image.

    :param data: The input data to be checked.
    :type data: bytes
    :return: True if the input data represents a JPG image, False otherwise.
    :rtype: bool
    """

    return data[:2] == b"\xFF\xD8"


def is_png(data) -> bool:
    """
    Check if the input data represents a PNG image.

    :param data: The input data to be checked.
    :type data: bytes
    :return: True if the input data represents a PNG image, False otherwise.
    :rtype: bool
    """

    return data[:8] == b"\x89PNG\r\n\x1a\n"


def get_image_format(file) -> str | None:
    """
    Get the format of an image file based on its header.

    :param file: The file to determine the image format for.
    :type file: file-like object
    :return: The format of the image (e.g., 'jpg' or 'png'), \
        or None if the format is not recognized.
    :rtype: str | None
    """

    header = file.read(8)

    if is_jpg(header):
        return "jpg"

    elif is_png(header):
        return "png"

    else:
        return None
