from dataclasses import dataclass
from src.objects.documents import TableImage
import pandas as pd
from typing import List


@dataclass
class TableData:
    header: dict
    data: dict


@dataclass
class Table:
    image: TableImage
    data: TableData

    def __post_init__(self):
        self._df = pd.DataFrame(self.data.data).rename(
            columns=self.data.header
        )

    @property
    def markdown(self):
        return self.df.to_markdown(index=False)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df

    def save(self, fname):
        self.df.to_csv(f"{fname}.csv", index=False)
