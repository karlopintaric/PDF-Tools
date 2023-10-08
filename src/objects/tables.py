from dataclasses import dataclass
from typing import List

import pandas as pd

from src.objects.documents import TableImage


@dataclass
class TableData:
    """
    Represents the data structure for table header and rows.

    :param header: The header of the table as a dictionary.
    :type header: dict
    :param data: The data rows of the table as a list of dictionaries.
    :type data: List[dict]
    """

    header: dict
    data: List[dict]


@dataclass
class Table:
    """
    Represents a table with associated image and data.

    :param image: The image associated with the table.
    :type image: TableImage
    :param data: The data structure for table header and rows.
    :type data: TableData
    """

    image: TableImage
    data: TableData

    def __post_init__(self):
        self._df = pd.DataFrame(self.data.data).rename(
            columns=self.data.header
        )

    @property
    def markdown(self):
        """
        Converts the table data to a Markdown formatted string.

        :return: Markdown formatted table.
        :rtype: str
        """

        return self.df.to_markdown(index=False)

    @property
    def df(self):
        """
        Gets the DataFrame representation of the table.

        :return: DataFrame representing the table.
        :rtype: pd.DataFrame
        """

        return self._df

    @df.setter
    def df(self, df):
        self._df = df

    def save(self, fname):
        """
        Saves the table data to a CSV file.

        :param fname: The file name for the CSV file (excluding extension).
        :type fname: str
        """

        self.df.to_csv(f"{fname}.csv", index=False)
