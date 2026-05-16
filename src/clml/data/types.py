from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    task: str
    description: str
    feature_columns: list[str]
    target_column: str | None
    source: str
    rows: int
    columns: int


@dataclass(frozen=True)
class DatasetBundle:
    info: DatasetInfo
    frame: pd.DataFrame

    @property
    def x(self) -> pd.DataFrame:
        return self.frame[self.info.feature_columns]

    @property
    def y(self) -> pd.Series | None:
        if self.info.target_column is None:
            return None
        return self.frame[self.info.target_column]
