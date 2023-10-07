from typing import List, Iterable, Tuple, Callable

import cv2
import numpy as np
import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from src.objects.documents import PageImage, PdfDocument, TableImage
from src.objects.tables import TableData, Table
from src.objects.misc import BBox
from abc import ABC, abstractmethod

from src.tools.ocr import tesserect_ocr
import re


class BaseExtractor(ABC):
    """
    A class for extracting tables from PDF documents or images.

    :param threshold: The confidence threshold for table detection, \
        defaults to 0.7
    :type threshold: float, optional
    :param pad: The amount of padding to add around the detected tables, \
        defaults to 25
    :type pad: int, optional
    :param expand: The expansion amount for the table bounding boxes, \
        defaults to 25
    :type expand: int, optional
    """

    def __init__(
        self,
        threshold,
        batch_size,
    ):
        self.model = self._init_model()
        self.feature_extractor = DetrImageProcessor()
        self.threshold = threshold
        self.batch_size = batch_size

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def _validate_input(self):
        pass

    def _batch_input(self, inputs):
        batch = []

        for item in inputs:
            batch.append(item)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Yield the last batch with remaining elements
        if batch:
            yield batch

    @abstractmethod
    def extract(self):
        pass

    def _process_batch(self, batch) -> list:
        encoding = self._preprocess(batch)
        model_outputs = self._forward(encoding)
        batch_tables = self._post_process(model_outputs, batch)

        return batch_tables

    @abstractmethod
    def _preprocess(self):
        pass

    def _forward(self, model_inputs: dict):
        """
        Perform forward pass through the table detection model.

        :param model_inputs: Input data for the model.
        :type model_inputs: dict
        """

        return self.model(**model_inputs)

    @abstractmethod
    def _post_process(self):
        pass

    def _post_process_results(self, model_outputs, inputs):
        target_sizes = torch.tensor([inpt.size for inpt in inputs])
        return self.feature_extractor.post_process_object_detection(
            model_outputs, threshold=self.threshold, target_sizes=target_sizes
        )


class TableExtractor(BaseExtractor):
    """
    A class for extracting tables from PDF documents or images.

    :param threshold: The confidence threshold for table detection, \
        defaults to 0.7
    :type threshold: float, optional
    :param pad: The amount of padding to add around the detected tables, \
        defaults to 25
    :type pad: int, optional
    :param expand: The expansion amount for the table bounding boxes, \
        defaults to 25
    :type expand: int, optional
    """

    def __init__(
        self,
        threshold: float = 0.92,
        pad: int = 50,
        expand: int = 25,
        batch_size: int = 2,
    ):
        super().__init__(threshold, batch_size)

        self.pad = pad
        self.expand = expand

    def extract(
        self,
        inputs: PdfDocument
        | PageImage
        | np.ndarray
        | Iterable[PageImage | np.ndarray],
    ) -> Iterable[TableImage]:
        """
        Extract tables from input data.

        :param inputs: Input data, which can be a PdfDocument, PageImage, \
            numpy array, or a list of numpy arrays.
        :type inputs: PdfDocument | PageImage | np.ndarray | List[np.ndarray]
        :return: A list of TableImage objects representing extracted tables.
        :rtype: List[TableImage]
        :raises ValueError: If the input is not one of the supported types.
        """

        inputs = self._validate_input(inputs)

        for batch in self._batch_input(inputs):
            yield from self._process_batch(batch)

    def _validate_input(self, inputs):
        if isinstance(inputs, PdfDocument):
            inputs = inputs.page_images

        elif isinstance(inputs, np.ndarray):
            inputs = PageImage(img=inputs)

        elif isinstance(inputs, list) and all(
            isinstance(inpt, np.ndarray) for inpt in inputs
        ):
            inputs = [PageImage(img=inpt) for inpt in inputs]

        else:
            raise ValueError(
                "Input must be a PdfDocument object, PageImage object, \
                numpy array, or a list of numpy arrays."
            )

        return inputs

    def _init_model(self):
        return TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )

    def _preprocess(self, inputs: PageImage | Iterable[PageImage]) -> dict:
        """
        Preprocess input data for table detection.

        :param inputs: A list of PageImage objects.
        :type inputs: List[PageImage]
        :return: Preprocessed input data.
        :rtype: dict
        """

        return self.feature_extractor(
            [inpt.img for inpt in inputs], return_tensors="pt"
        )

    def _post_process(
        self, model_outputs, inputs: PageImage | Iterable[PageImage]
    ) -> List[TableImage]:
        """
        Post-process model outputs to extract tables.

        :param model_outputs: Output from the table detection model.
        :param inputs: Input PageImage objects.
        :type inputs: List[PageImage]
        :return: A list of TableImage objects representing extracted tables.
        :rtype: List[TableImage]
        """

        tables = []

        results = self._post_process_results(model_outputs, inputs)

        for inpt, lines in zip(inputs, results):
            for i, box in enumerate(lines["boxes"]):
                box = BBox(*[int(round(i)) for i in box.tolist()])
                
                table = self._extract_with_padding(inpt.img, box)
                
                table_image = TableImage(img=table, page_num=inpt.page_num, table_num=i)
                table_image.word_positions = inpt.words_inside_box(box)

                tables.append(table_image)

        return tables

    def _crop(self, img: np.ndarray, box: BBox) -> np.ndarray:
        """
        Crop an image based on a bounding box.

        :param img: The input image.
        :type img: np.ndarray
        :param box: The bounding box coordinates [xmin, ymin, xmax, ymax].
        :type box: List[float | int]
        :return: The cropped image.
        :rtype: np.ndarray
        """

        return img[box.ymin:box.ymax, box.xmin:box.xmax]

    def _pad(self, img: np.ndarray) -> np.ndarray:
        """
        Add padding to an image.

        :param img: The input image.
        :type img: np.ndarray
        :return: The padded image.
        :rtype: np.ndarray
        """

        return cv2.copyMakeBorder(
            img,
            self.pad,
            self.pad,
            self.pad,
            self.pad,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def _extract_with_padding(
        self, img: np.ndarray, box: List[float | int]
    ) -> np.ndarray:
        """
        Extract a table from an image with padding.

        :param img: The input image.
        :type img: np.ndarray
        :param box: The bounding box coordinates [xmin, ymin, xmax, ymax].
        :type box: List[float | int]
        :return: The extracted table with padding.
        :rtype: np.ndarray
        """
        box.expand(self.expand)
        img = self._crop(img, box)

        box.pad = self.pad
        img = self._pad(img)

        return img


class TableDataExtractor(BaseExtractor):
    """
    A class for extracting tables from PDF documents or images.

    :param threshold: The confidence threshold for table detection, \
        defaults to 0.7
    :type threshold: float, optional
    :param pad: The amount of padding to add around the detected tables, \
        defaults to 25
    :type pad: int, optional
    :param expand: The expansion amount for the table bounding boxes, \
        defaults to 25
    :type expand: int, optional
    """

    def __init__(
        self,
        threshold: float = 0.8,
        batch_size: int = 2,
        use_ocr: bool = False,
        ocr: Callable = tesserect_ocr(custom_config = '--psm 6'),
    ):
        super().__init__(threshold, batch_size)

        self.use_ocr = use_ocr
        self.ocr = ocr

    def extract(
        self,
        inputs: TableImage | np.ndarray | Iterable[TableImage | np.ndarray],
    ) -> Iterable[Table]:
        """
        Extract tables from input data.

        :param inputs: Input data, which can be a PdfDocument, PageImage, \
            numpy array, or a list of numpy arrays.
        :type inputs: PdfDocument | PageImage | np.ndarray | List[np.ndarray]
        :return: A list of TableImage objects representing extracted tables.
        :rtype: List[TableImage]
        :raises ValueError: If the input is not one of the supported types.
        """

        inputs = self._validate_input(inputs)

        for batch in self._batch_input(inputs):
            yield from self._process_batch(batch)

    def _validate_input(self, inputs):
        if isinstance(inputs, TableImage):
            return inputs

        if isinstance(inputs, list) and all(
            isinstance(inpt, TableImage) for inpt in inputs
        ):
            return inputs

        if isinstance(inputs, np.ndarray):
            return TableImage(img=inputs)

        if isinstance(inputs, list) and all(
            isinstance(inpt, np.ndarray) for inpt in inputs
        ):
            return [TableImage(img=inpt) for inpt in inputs]

        raise ValueError(
            "Input must be a PdfDocument object, PageImage object, \
            numpy array, or a list of numpy arrays."
        )

    def _init_model(self):
        return TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )

    def _preprocess(self, inputs: TableImage | Iterable[TableImage]) -> dict:
        """
        Preprocess input data for table detection.

        :param inputs: A list of PageImage objects.
        :type inputs: List[PageImage]
        :return: Preprocessed input data.
        :rtype: dict
        """

        return self.feature_extractor(
            [inpt.img for inpt in inputs], return_tensors="pt"
        )

    def _post_process(
        self, model_outputs, inputs: TableImage | Iterable[TableImage]
    ) -> Iterable[Table]:
        """
        Post-process model outputs to extract tables.

        :param model_outputs: Output from the table detection model.
        :param inputs: Input PageImage objects.
        :type inputs: List[PageImage]
        :return: A list of TableImage objects representing extracted tables.
        :rtype: List[TableImage]
        """
        tables = []

        results = self._post_process_results(model_outputs, inputs)
        for inpt, result in zip(inputs, results):
            labeled_boxes = self._extract_labeled_boxes(result)
            header, data = self._extract_text_from_table(labeled_boxes, inpt)
            table_data = TableData(header, data)

            tables.append(Table(inpt, table_data))

        return tables

    def _extract_labeled_boxes(self, results: dict) -> dict:
        labeled_boxes = {}
        for label_id, box in zip(
            results["labels"].tolist(), results["boxes"].tolist()
        ):
            label = self.model.config.id2label[label_id].replace(" ", "_")
            labeled_boxes[label] = labeled_boxes.get(label, []) + [box]

        return labeled_boxes

    def _extract_text_from_table(
        self, labeled_boxes: dict, input: TableImage
    ) -> Tuple[dict, List[dict]]:
        # Header coords
        header_position = self.get_header_position(labeled_boxes)

        if header_position is None:
            header_ymax = 0
        else:
            header_ymax = header_position.ymax

        # Table seperation lines
        vlines, hlines = self.get_data_separations(
            labeled_boxes, header_ymax
        )

        header = self._extract_header(input, header_position, vlines)
        data = self._extract_all_data(input, hlines, vlines)

        return header, data

    def get_data_separations(self, labeled_boxes, header_ymax):
        vlines = [0] + sorted(
            [box[2] for box in labeled_boxes["table_column"]]
        )
        hlines = sorted(
            [
                box[3]
                for box in labeled_boxes["table_row"]
                if box[3] - header_ymax > 0
            ]
        )

        vlines = self._remove_close_lines(vlines)
        hlines = self._remove_close_lines(hlines)

        return vlines, hlines

    def get_header_position(self, labeled_boxes):
        header = labeled_boxes.get("table_column_header")
        
        if header is not None:
            xmin, ymin, xmax, ymax = (
                int(round(i)) for i in header[0]
            )

            return BBox(xmin, ymin, xmax, ymax)

    def _extract_header(self, table_img, header_position, vlines):
        if header_position is None:
            return {i: str(i) for i in range(len(vlines) - 1)}
        
        ymin, ymax = header_position.ymin, header_position.ymax

        header_img = table_img.preprocess[ymin:ymax, :]

        data = None
        if not self.use_ocr:
            data = table_img.words_inside_box(header_position)
        
        if not data:
            data = self.ocr(header_img)

        bboxes = [
            BBox(vlines[i], 0, vlines[i + 1], header_img.shape[0])
            for i in range(len(vlines) - 1)
        ]

        return self._structure_text_for_table(data, bboxes)

    def _structure_text_for_table(self, data, bboxes):
        row_cells = {i: "" for i, _ in enumerate(bboxes)}

        for i, (word_box, word) in enumerate(zip(data["box"], data["text"])):
            conf = data.get("conf", [100] * len(data["text"]))[i]

            if not re.search(r"[\w]+", word) or conf < 50:
                continue

            word = word.strip()

            for j, bbox in enumerate(bboxes):
                if word_box in bbox:
                    row_cells[j] += f"{word} "

        for k, v in row_cells.items():
            row_cells[k] = v.strip()

        return row_cells

    def _extract_all_data(
        self,
        table_img: TableImage,
        hlines: List[int],
        vlines: List[int],
    ) -> List[dict]:
        all_data = []

        for i in range(len(hlines) - 1):
            ymin = int(hlines[i])
            ymax = int(hlines[i + 1])

            row_position = BBox(0, ymin, table_img.size[1], ymax)
            row_img = table_img.preprocess[ymin:ymax, :]

            data = None
            if not self.use_ocr:
                data = table_img.words_inside_box(row_position)
            
            if not data:
                data = self.ocr(row_img)
                
            bboxes = [
                BBox(vlines[i], 0, vlines[i + 1], row_img.shape[0])
                for i in range(len(vlines) - 1)
            ]

            row_cells = self._structure_text_for_table(data, bboxes)
            all_data.append(row_cells)

        return all_data

    def _remove_close_lines(self, lines):
        # Initialize a lines list to store the selected integers
        clean_lines = []

        # Iterate through the list of integers
        for i in range(len(lines)):
            # Cast the float to integer and keep the first integer
            if i == 0:
                clean_lines.append(int(lines[i]))
            else:
                # Check if the absolute difference with the previous integer is >= 5
                if abs(int(lines[i]) - lines[i - 1]) >= 5:
                    clean_lines.append(int(lines[i]))

        return clean_lines


if __name__ == "__main__":
    table_extractor = TableExtractor()
    data_extractor = TableDataExtractor(use_ocr=False)

    PDF_PATH = "/home/karlo/Coding/DataScience/PDF-Tools/test_files/ast_sci_data_tables_sample.pdf"
    pdf = PdfDocument(PDF_PATH)

    tables = list(table_extractor.extract(pdf))
    for i, extracted_table in enumerate(data_extractor.extract(tables)):
        print(extracted_table.df.to_markdown(index=False))
        print()
