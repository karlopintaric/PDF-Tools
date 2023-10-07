import io
from pathlib import Path
from typing import List

import fitz

from src.objects.validations import PdfSource


def open_pdf(src: PdfSource):
    """
    Open a PDF document from various sources.

    :param src: The source of the PDF document, can be a fitz.Document, \
        string, io.BytesIO, bytes, or Path.
    :type src: Union[fitz.Document, str, Path, io.BytesIO, bytes]

    :raises ValueError: If there is an error opening the PDF.
    :raises TypeError: If the 'src' argument is of an unsupported type.

    :return: The opened PDF document.
    :rtype: fitz.Document
    """

    if isinstance(src, fitz.Document):
        return src

    if isinstance(src, (str, Path)):
        try:
            doc = fitz.open(src)
        except Exception as e:
            raise ValueError(f"Error opening {src}: {str(e)}")

    elif isinstance(src, (io.BytesIO, bytes)):
        if isinstance(src, io.BytesIO):
            src = src.getvalue()

        try:
            doc = fitz.open(stream=src)
        except Exception as e:
            raise ValueError(
                f"Error opening PDF: {str(e)}. "
                "Check if the object is a valid PDF."
            )

    else:
        raise TypeError(
            "The 'src' argument must be a fitz.Document, string, "
            "io.BytesIO, bytes, or Path."
        )

    return doc


def merge_docs(srcs: List[PdfSource]) -> fitz.Document:
    """
    Merge multiple PDF documents into a single PDF document.

    :param srcs: The list of PDF sources to merge.
    :type srcs: Union[List[PdfSource]]

    :return: The merged PDF document.
    :rtype: fitz.Document
    """

    merged_doc = fitz.open()
    merged_pages = 0
    merged_toc = None

    for src in srcs:
        doc = open_pdf(src)
        toc = doc.get_toc(False)

        for t in toc:
            t[2] += merged_pages

        if merged_toc is None:
            merged_toc = toc
        else:
            merged_toc += toc

        merged_doc.insert_pdf(doc)
        merged_pages += len(doc)

    merged_doc.set_toc(merged_toc)

    return merged_doc


def split_doc(
    src: PdfSource, split_on_pages: List[int]
) -> List[fitz.Document]:
    """
    Split a PDF document into multiple smaller PDF documents \
        based on the specified page numbers.

    :param src: The source PDF document to split.
    :type src: PdfSource
    :param split_on_pages: A list of page numbers on \
        which to split the document.
    :type split_on_pages: List[int]

    :raises TypeError: If the 'split_on_pages' argument \
        is not a list of integers.
    :raises ValueError: If the page numbers are out of range.

    :return: A list of split PDF documents.
    :rtype: List[fitz.Document]
    """

    doc = open_pdf(src)
    toc = doc.get_toc(False)
    page_count = doc.page_count

    if not isinstance(split_on_pages, list) or not all(
        isinstance(page, int) for page in split_on_pages
    ):
        raise TypeError(
            "The 'split_on_pages' argument must be of type List[int]."
        )

    if min(split_on_pages) < 1 or max(split_on_pages) >= page_count:
        raise ValueError(
            "The page number can't be a negative number and must be "
            "smaller than the total number of pages"
        )

    split_docs = []
    from_page = 0

    for page in split_on_pages:
        new_doc = fitz.open()
        new_toc = [t for t in toc if from_page < t[2] <= page]

        for t in new_toc:
            t[2] -= from_page

        new_doc.insert_pdf(doc, from_page=from_page, to_page=page - 1)
        new_doc.set_toc(new_toc)
        split_docs.append(new_doc)

        toc = [t for t in toc if t not in new_toc]

        from_page = page

    new_doc = fitz.open()
    found_start = False
    toc = [t for t in toc if (found_start := t[0] == 1) or found_start]

    for t in toc:
        t[2] -= from_page

    new_doc.insert_pdf(doc, from_page=from_page)
    new_doc.set_toc(toc)
    split_docs.append(new_doc)

    return split_docs
