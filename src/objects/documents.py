from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Iterable, Dict
from collections import defaultdict

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np

from src.objects.validations import ImageSource, PdfSource, is_image_array
from src.tools.images import images_to_pdf, pdf_to_images, read_img
from src.tools.pdf import merge_docs, open_pdf
from src.objects.misc import BBox


@dataclass
class DocumentImage:
    """
    Represents an image extracted from a document.

    :param img: The image data as a NumPy array.
    :type img: numpy.ndarray
    :param page_num: The page number from which the image was extracted, \
        defaults to 0.
    :type page_num: int, optional
    :param size: The size of the image, initialized automatically.
    :type size: numpy.ndarray
    """

    img: np.ndarray
    page_num: int = field(default=0)
    size: np.ndarray = field(init=False)
    fname: str = field(init=False)
    dpi: Optional[int] = None
    page_text: Optional[fitz.TextPage] = None

    def __post_init__(self):
        if not is_image_array(self.img):
            raise ValueError("Input array is not a valid image array.")

        if self.page_text and not self.dpi:
            raise ValueError(
                "Provide the DPI of the image if extracted from document."
            )

        self.fname = f"doc_{self.page_num + 1}"
        self.size = self.img.shape[:2]

    @cached_property
    def word_positions(self):
        if self.page_text is None:
            return None

        words = self.page_text.extractWORDS()

        if not words:
            return None

        scale_factor = self.dpi / 72

        d = {"box": [], "text": []}
        for word in words:
            xmin, ymin, xmax, ymax = [
                int(round(p * scale_factor)) for p in word[:4]
            ]

            d["box"].append(BBox(xmin, ymin, xmax, ymax))
            d["text"].append(word[4])

        return d

    def words_inside_box(self, box: BBox) -> dict:
        if self.word_positions is None:
            return None

        boxes = self.word_positions.get('box')
        words = self.word_positions.get('text')

        if boxes is None or words is None:
            return None

        words_inside = defaultdict(list)

        for word_box, word in zip(boxes, words):
            if word_box in box:
                words_inside['box'].append(word_box - box)
                words_inside['text'].append(word)

        return dict(words_inside)

    def save(self, fname: Optional[str] = None) -> None:
        """
        Save the image to a file.

        :param fname: The filename to save the image to, \
            defaults to "doc_{self.page_num}.png".
        :type fname: str, optional
        """

        bgr_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

        if fname is None:
            fname = self.fname

        cv2.imwrite(f"{fname}.png", bgr_img)

    def show(self) -> None:
        """
        Display the image using Matplotlib.
        """

        plt.figure(figsize=(10, 14))
        plt.imshow(self.img)
        plt.axis("off")  # Hide the axis
        plt.show()

    def __iter__(self):
        return self


@dataclass
class PageImage(DocumentImage):
    """
    Represents an image extracted from a document page.

    :param img: The image data as a NumPy array.
    :type img: numpy.ndarray
    :param page_num: The page number from which the image was extracted, \
        defaults to 0.
    :type page_num: int, optional
    """

    def __post_init__(self):
        super().__post_init__()
        self.fname = f"page_{self.page_num + 1}"


@dataclass
class TableImage(DocumentImage):
    """
    Represents an image of a table extracted from a document.

    :param img: The image data as a NumPy array.
    :type img: numpy.ndarray
    :param page_num: The page number from which the image was extracted, \
        defaults to 0.
    :type page_num: int, optional
    :param table_num: The table number, defaults to 0.
    :type table_num: int, optional
    """

    table_num: int = field(default=0)

    def __post_init__(self):
        super().__post_init__()
        self.fname = f"table_{self.table_num + 1}_page_{self.page_num + 1}"

    @cached_property
    def preprocess(self) -> np.ndarray:
        """
        Apply preprocessing steps to the table image.

        :return: The preprocessed table image.
        :rtype: numpy.ndarray
        """

        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE()
        clahe_img = clahe.apply(gray)

        _, thresh = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_BINARY)

        return thresh


class PdfDocument:
    def __init__(
        self,
        src: PdfSource,
        dpi: int = 300,
        images: Optional[Iterable[np.ndarray]] = None,
        pages: Optional[List[int]] = None,
    ):
        """
        Represents a PDF document.

        :param src: The source of the PDF document.
        :type src: PdfSource
        :param image_list: A list of image sources \
            to be used for PDF creation, defaults to None.
        :type image_list: List[ImageSource], optional
        :param pages: A list of page numbers to extract, defaults to None.
        :type pages: List[int], optional
        """

        self.doc = src
        self.pages = pages
        self.dpi = dpi
        self._page_images = None

        if images is not None:
            self._page_images = (
                PageImage(img=img, page_num=i) for i, img in enumerate(images)
            )

    @property
    def page_images(self) -> Iterable[PageImage]:
        """
        Get a list of page images extracted from the PDF document.

        :return: List of page images.
        :rtype: List[PageImage]
        """
        if self._page_images is not None:
            return self._page_images

        images = pdf_to_images(self.doc, self.dpi)
        texts = [page.get_textpage() for page in self.doc]

        return (
            PageImage(img=img, page_text=text, page_num=i, dpi=self.dpi)
            for i, (img, text) in enumerate(zip(images, texts))
        )
    
    @page_images.setter
    def page_images(self, page_images):
        self._page_images = page_images

    @property
    def doc(self) -> fitz.Document:
        """
        Get the underlying PDF document object.

        :return: The PDF document object.
        :rtype: fitz.Document
        """

        return self._doc

    @doc.setter
    def doc(self, src):
        self._doc = open_pdf(src)

    @property
    def pages(self) -> List[int] | None:
        """
        Get the list of page numbers to extract.

        :return: List of page numbers.
        :rtype: list
        """

        return self._pages

    @pages.setter
    def pages(self, pages):
        if pages is None:
            self._pages = pages

        elif not all(isinstance(page, int) for page in pages):
            raise TypeError("The 'pages' argument must be a list of integers.")

        else:
            self._pages = pages

    def save(self, fname: str) -> None:
        """
        Save the PDF document to a file.

        :param fname: The filename to save the PDF document to.
        :type fname: str
        """

        self.doc.save(f"{fname}.pdf")

    def merge(self, srcs: List[PdfSource]) -> None:
        """
        Merge the PDF document with one or more additional sources.

        :param srcs: List of PDF sources to merge with.
        :type srcs: List[PdfSource]
        """

        self._doc = merge_docs([self.doc, *srcs])

        if "page_images" in self.__dict__:
            del self.page_images

    def __len__(self):
        """
        Get the number of pages in the PDF document.

        :return: Number of pages.
        :rtype: int
        """

        return len(self.doc)

    def __getitem__(self, idx: int):
        """
        Get a specific page from the PDF document.

        :param idx: Page index.
        :type idx: int
        :return: Page object.
        :rtype: fitz.Page
        """

        return self.doc[idx]

    @classmethod
    def from_images(cls, image_list: List[ImageSource]):
        """
        Create a PDF document from a list of image sources.

        :param image_list: List of image sources.
        :type image_list: List[ImageSource]
        :return: PDFDocument instance.
        :rtype: PdfDocument
        """

        doc = images_to_pdf(image_list)
        images = (read_img(img) for img in image_list)

        return cls(src=doc, images=images)
