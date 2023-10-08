import io
from pathlib import Path
from typing import List, Optional

import cv2
import fitz
import numpy as np

from src.objects.validations import ImageSource, get_image_format


def read_img(src: ImageSource) -> np.ndarray:
    if isinstance(src, Path):
        src = str(Path)

    if isinstance(src, str):
        return cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)

    if isinstance(src, io.BytesIO):
        src = src.read()

    if isinstance(src, bytes):
        decoded = cv2.imdecode(np.frombuffer(src, np.uint8), 1)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    raise TypeError(
        "The 'src' argument must be a string, io.BytesIO, bytes or Path."
    )


def open_image_as_document(src: ImageSource) -> fitz.Document:
    """
    Open an image file or stream and convert it to a PyMuPDF Document object.

    :param src: The source of the image, which can be a string file path, \
        a Path object, bytes, or io.BytesIO.
    :type src: Union[str, Path, bytes, io.BytesIO]

    :return: A PyMuPDF Document object representing the image.
    :rtype: fitz.Document

    :raises ValueError: If there is an error opening the image source.
    :raises TypeError: If the 'src' argument is of an unsupported type.
    """

    if isinstance(src, (str, Path)):
        try:
            doc = fitz.open(src)
        except Exception as e:
            raise ValueError(f"Error opening {src}: {str(e)}")

    elif isinstance(src, (io.BytesIO, bytes)):
        try:
            if isinstance(src, io.BytesIO):
                filename = src.name if hasattr(src, "name") else None
                src = src.getvalue()

            img_format = get_image_format(src) if not filename else None
            doc = fitz.open(filename=filename, filetype=img_format, stream=src)

        except Exception as e:
            raise ValueError(
                f"Error opening image: {str(e)}. "
                "Check if the object is a valid image."
            )

    else:
        raise TypeError(
            "The 'src' argument must be a string, io.BytesIO, bytes, or Path."
        )

    return doc


def images_to_pdf(img_list: List[ImageSource]) -> fitz.Document:
    """
    Convert a list of image files or streams to a single PDF document.

    :param img_list: List of image sources, which can be strings (file paths) \
        or io.BytesIO objects.
    :type img_list: List[Union[str, io.BytesIO]]

    :return: A PyMuPDF Document object representing the generated PDF.
    :rtype: fitz.Document
    """

    doc = fitz.open()  # PDF with the pictures

    for f in img_list:
        img = open_image_as_document(f)

        rect = img[0].rect  # pic dimension
        pdfbytes = img.convert_to_pdf()  # make a PDF stream
        img.close()  # no longer needed
        imgPDF = fitz.open("pdf", pdfbytes)  # open stream as PDF
        page = doc.new_page(
            width=rect.width, height=rect.height  # new page with ...
        )  # pic dimension
        page.show_pdf_page(rect, imgPDF, 0)  # image fills the page

    return doc


def pdf_to_images(
    doc: fitz.Document, dpi: int, pages: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Extract images from a PDF document and return them as \
        a list of NumPy arrays.

    :param doc: The input PDF document.
    :type doc: fitz.Document

    :param pages: Optional list of page numbers to extract images from. \
        If None, extract images from all pages.
    :type pages: List[int], optional

    :return: A list of NumPy arrays, each representing an extracted image.
    :rtype: List[np.array]
    """

    page_images = []

    for page in doc:
        page_number = page.number
        if not pages or page_number in pages:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, -1)
            )
            page_images.append(img)

    return page_images
