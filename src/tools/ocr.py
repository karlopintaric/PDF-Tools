import pytesseract
from pytesseract import Output
from functools import partial
from src.objects.misc import BBox


def tesserect_ocr(custom_config: str = ""):
    def ocr_image(img):
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
