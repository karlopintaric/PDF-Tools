import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from src.objects.documents import PageImage, PdfDocument, TableImage
from src.objects.misc import BBox
from src.objects.tables import Table, TableData
from src.tools.ocr import tesserect_ocr


class BaseExtractor(ABC):
    """
    A class for extracting tables from PDF documents or images.

    :param threshold: The confidence threshold for table detection
    :type threshold: float
    :param batch_size: Number of items in batch during model inference
    :type batch_size: int
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
        """
        Initialize the table extraction model.

        :return: The initialized table extraction model.
        :rtype: Any
        """

        pass

    @abstractmethod
    def _validate_input(self):
        """
        Validate the input data for table extraction.

        :raises ValueError: If the input data is invalid.
        """

        pass

    def _batch_input(self, inputs):
        """
        Generate batches of input data.

        :param inputs: List of input data.
        :type inputs: list
        :return: Generator yielding batches of input data.
        :rtype: generator
        """

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
    def extract(self, inputs):
        """
        Extract tables from the provided inputs.

        :param inputs: Input data for table extraction.
        :type inputs: Any
        :return: Extracted tables.
        :rtype: Any
        """

        pass

    def _process_batch(self, batch) -> list:
        """
        Process a batch of data for table extraction.

        :param batch: Batch of input data.
        :type batch: list
        :return: Processed tables for the batch.
        :rtype: list
        """

        encoding = self._preprocess(batch)
        model_outputs = self._forward(encoding)
        batch_tables = self._post_process(model_outputs, batch)

        return batch_tables

    @abstractmethod
    def _preprocess(self, batch):
        """
        Preprocess the input data before feeding it to the model.

        :param batch: Batch of input data.
        :type batch: list
        :return: Preprocessed data.
        :rtype: Any
        """

        pass

    def _forward(self, model_inputs: dict):
        """
        Perform forward pass through the table detection model.

        :param model_inputs: Input data for the model.
        :type model_inputs: dict
        :return: Model outputs.
        :rtype: Any
        """

        return self.model(**model_inputs)

    @abstractmethod
    def _post_process(self, model_outputs, batch):
        """
        Post-process the model outputs to obtain tables.

        :param model_outputs: Model outputs.
        :type model_outputs: Any
        :param batch: Batch of input data.
        :type batch: list
        :return: Extracted tables for the batch.
        :rtype: Any
        """

        pass

    def _post_process_results(self, model_outputs, inputs):
        """
        Post-process the model outputs to obtain final results.

        :param model_outputs: Model outputs.
        :type model_outputs: Any
        :param inputs: Input data for table extraction.
        :type inputs: Any
        :return: Post-processed results.
        :rtype: Any
        """

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
    :param batch_size: Number of items in batch during model inference, \
        defaults to 2
    :type batch_size: int, optional
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
        """
        Initializes the table detection model.
        """

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

                table_image = TableImage(
                    img=table, page_num=inpt.page_num, table_num=i
                )
                table_image.word_positions = inpt.words_inside_box(box)

                tables.append(table_image)

        return tables

    def _crop(self, img: np.ndarray, box: BBox) -> np.ndarray:
        """
        Crop an image based on a bounding box.

        :param img: The input image.
        :type img: np.ndarray
        :param box: The bounding box.
        :type box: BBox
        :return: The cropped image.
        :rtype: np.ndarray
        """

        return img[box.ymin : box.ymax, box.xmin : box.xmax]

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

    def _extract_with_padding(self, img: np.ndarray, box: BBox) -> np.ndarray:
        """
        Extract a table from an image with padding.

        :param img: The input image.
        :type img: np.ndarray
        :param box: The bounding box.
        :type box: BBox
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
    Extracts tabular data from images using TableTransformerForObjectDetection.

    :param threshold: Detection threshold for table components, defaults to 0.8
    :type threshold: float, optional
    :param batch_size: Batch size for processing, defaults to 2
    :type batch_size: int, optional
    :param use_ocr: Flag to enable OCR for text extraction, defaults to False
    :type use_ocr: bool, optional
    :param ocr: OCR function to use, defaults to Tesseract OCR with custom configuration
    :type ocr: Callable, optional
    """

    def __init__(
        self,
        threshold: float = 0.8,
        batch_size: int = 2,
        use_ocr: bool = False,
        ocr: Callable = tesserect_ocr(custom_config="--psm 6"),
    ):
        super().__init__(threshold, batch_size)

        self.use_ocr = use_ocr
        self.ocr = ocr

    def extract(
        self,
        inputs: TableImage | np.ndarray | Iterable[TableImage | np.ndarray],
    ) -> Iterable[Table]:
        """
        Extracts tabular data from the provided inputs.

        :param inputs: Input data, can be a TableImage, numpy array, \
            or an iterable of TableImages/numpy arrays
        :type inputs: TableImage | np.ndarray | Iterable[TableImage | np.ndarray]
        :return: Iterable of extracted tables
        :rtype: Iterable[Table]
        """

        inputs = self._validate_input(inputs)

        for batch in self._batch_input(inputs):
            yield from self._process_batch(batch)

    def _validate_input(self, inputs):
        """
        Validates the input data for extraction.

        :param inputs: Input data
        :raises ValueError: If input is not valid
        """

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
        """
        Initializes the table detection model.
        """

        return TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )

    def _preprocess(self, inputs: TableImage | Iterable[TableImage]) -> dict:
        """
        Preprocesses the input data for model processing.

        :param inputs: Input data
        :return: Preprocessed data
        :rtype: dict
        """

        return self.feature_extractor(
            [inpt.img for inpt in inputs], return_tensors="pt"
        )

    def _post_process(
        self, model_outputs, inputs: TableImage | Iterable[TableImage]
    ) -> Iterable[Table]:
        """
        Post-processes the model outputs to extract table information.

        :param model_outputs: Model outputs
        :param inputs: Input data
        :return: Iterable of extracted tables
        :rtype: Iterable[Table]
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
        """
        Extracts labeled bounding boxes from model results.

        :param results: Model output results
        :type results: dict
        :return: Labeled boxes
        :rtype: dict
        """

        labeled_boxes: dict = {}
        for label_id, box in zip(
            results["labels"].tolist(), results["boxes"].tolist()
        ):
            label = self.model.config.id2label[label_id].replace(" ", "_")
            labeled_boxes[label] = labeled_boxes.get(label, []) + [box]

        return labeled_boxes

    def _extract_text_from_table(
        self, labeled_boxes: dict, input: TableImage
    ) -> Tuple[dict, List[dict]]:
        """
        Extracts text from the labeled boxes to form table header and data.

        :param labeled_boxes: Labeled boxes
        :type labeled_boxes: dict
        :param input: TableImage input
        :type input: TableImage
        :return: Tuple of header and data
        :rtype: Tuple[dict, List[dict]]
        """

        # Header coords
        header_position = self.get_header_position(labeled_boxes)

        if header_position is None:
            header_ymax = 0
        else:
            header_ymax = header_position.ymax

        # Table seperation lines
        vlines, hlines = self.get_data_separations(labeled_boxes, header_ymax)

        header = self._extract_header(input, header_position, vlines)
        data = self._extract_all_data(input, hlines, vlines)

        return header, data

    def get_data_separations(self, labeled_boxes: dict, header_ymax: int):
        """
        Extracts vertical and horizontal separation lines for table data.

        :param labeled_boxes: Labeled boxes
        :type labeled_boxes: dict
        :param header_ymax: Y coordinate of the header
        :type header_ymax: int
        :return: Tuple of vertical and horizontal lines
        :rtype: tuple
        """

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

    def get_header_position(self, labeled_boxes: dict) -> BBox | None:
        """
        Determines the position of the header in the labeled boxes.

        :param labeled_boxes: Labeled boxes
        :type labeled_boxes: dict
        :return: Header position
        :rtype: BBox
        """

        header = labeled_boxes.get("table_column_header")

        if header is not None:
            xmin, ymin, xmax, ymax = (int(round(i)) for i in header[0])

            return BBox(xmin, ymin, xmax, ymax)

        return None

    def _extract_header(
        self,
        table_img: TableImage,
        header_position: BBox | None,
        vlines: List[int],
    ) -> dict:
        """
        Extracts the header from the table image.

        :param table_img: Table image
        :type table_img: TableImage
        :param header_position: Header position
        :type header_position: BBox | None
        :param vlines: Vertical lines
        :type vlines: List[int]
        :return: Extracted header
        :rtype: dict
        """

        if header_position is None:
            return {i: str(i) for i in range(len(vlines) - 1)}

        ymin, ymax = header_position.ymin, header_position.ymax

        header_img = table_img.preprocess[ymin:ymax, :]

        data = None
        if not self.use_ocr:
            data = table_img.words_inside_box(header_position)

        if data is None:
            data = self.ocr(header_img)

        bboxes = [
            BBox(vlines[i], 0, vlines[i + 1], header_img.shape[0])
            for i in range(len(vlines) - 1)
        ]

        return self._structure_text_for_table(data, bboxes)

    def _structure_text_for_table(
        self, data: dict, bboxes: List[BBox]
    ) -> dict:
        """
        Structures the extracted text for the table based on bounding boxes.

        :param data: Extracted text data
        :type data: dict
        :param bboxes: Bounding boxes
        :type bboxes: List[BBox]
        :return: Structured text for the table
        :rtype: dict
        """

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
        """
        Extracts all data rows from the table image.

        :param table_img: Table image
        :type table_img: TableImage
        :param hlines: Horizontal lines
        :type hlines: List[int]
        :param vlines: Vertical lines
        :type vlines: List[int]
        :return: Extracted data rows
        :rtype: List[dict]
        """

        all_data = []

        for i in range(len(hlines) - 1):
            ymin = int(hlines[i])
            ymax = int(hlines[i + 1])

            row_position = BBox(0, ymin, table_img.size[1], ymax)
            row_img = table_img.preprocess[ymin:ymax, :]

            data = None
            if not self.use_ocr:
                data = table_img.words_inside_box(row_position)

            if data is None:
                data = self.ocr(row_img)

            bboxes = [
                BBox(vlines[i], 0, vlines[i + 1], row_img.shape[0])
                for i in range(len(vlines) - 1)
            ]

            row_cells = self._structure_text_for_table(data, bboxes)
            all_data.append(row_cells)

        return all_data

    def _remove_close_lines(self, lines: List[float]) -> List[int]:
        """
        Removes closely spaced lines.

        :param lines: Lines to be cleaned
        :type lines: List[float]
        :return: Cleaned lines
        :rtype: list
        """

        # Initialize a lines list to store the selected integers
        clean_lines = []

        # Iterate through the list of integers
        for i in range(len(lines)):
            # Cast the float to integer and keep the first integer
            if i == 0:
                clean_lines.append(int(lines[i]))
            else:
                # Check if the difference with the previous integer is >= 5
                if abs(int(lines[i]) - lines[i - 1]) >= 5:
                    clean_lines.append(int(lines[i]))

        return clean_lines
