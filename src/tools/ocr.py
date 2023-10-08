from typing import Callable

import numpy as np
import pytesseract
from pytesseract import Output

from src.objects.misc import BBox


def tesserect_ocr(custom_config: str = "") -> Callable:
    """
    Provides functions for performing OCR using Tesseract.

    :param custom_config: Custom configuration for Tesseract OCR \
        (optional, defaults to "").
    :type custom_config: str
    :return: Function for performing OCR on an image.
    :rtype: Callable
    """

    def ocr_image(img: np.ndarray) -> dict:
        """
        Performs OCR on the provided image using Tesseract.

        :param img: The image to perform OCR on.
        :type img: Any
        :return: OCR results including bounding boxes, text, and confidence levels.
        :rtype: dict
        """

        data = pytesseract.image_to_data(
            img, config=custom_config, output_type=Output.DICT
        )
        cleaned_data = {"box": [], "text": data["text"], "conf": data["conf"]}

        for i, _ in enumerate(data["text"]):
            xmin = data["left"][i]
            ymin = data["top"][i]
            xmax = data["width"][i] + xmin
            ymax = data["height"][i] + ymin

            cleaned_data["box"].append(BBox(xmin, ymin, xmax, ymax))

        return cleaned_data

    return ocr_image
