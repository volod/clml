import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

Record = dict[str, Any]


class DictReader(Protocol):
    def read(self, path: Path) -> list[Record]: ...


class DictWriter(Protocol):
    def write(
        self,
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str] | None = None,
    ) -> None: ...


class CsvDictReader:
    """Read CSV files as dictionaries; CSV-specific logic stays in this adapter."""

    def read(self, path: Path) -> list[Record]:
        with path.open(newline="", encoding="utf-8") as fh:
            return [
                {key: _normalize_csv_value(value) for key, value in row.items()}
                for row in csv.DictReader(fh)
            ]


class CsvDictWriter:
    """Write dictionary rows to CSV files with stable field ordering."""

    def write(
        self,
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str] | None = None,
    ) -> None:
        materialized = [dict(row) for row in rows]
        columns = list(fieldnames) if fieldnames is not None else _fieldnames(materialized)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(materialized)


_READERS: dict[str, DictReader] = {"csv": CsvDictReader()}
_WRITERS: dict[str, DictWriter] = {"csv": CsvDictWriter()}


def read_records(path: Path, *, data_format: str | None = None) -> list[Record]:
    return _reader_for(path, data_format).read(path)


def write_records(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    data_format: str | None = None,
) -> None:
    _writer_for(path, data_format).write(path, rows, fieldnames=fieldnames)


def read_frame(path: Path, *, data_format: str | None = None) -> pd.DataFrame:
    return _coerce_frame(pd.DataFrame(read_records(path, data_format=data_format)))


def write_frame(
    path: Path,
    frame: pd.DataFrame,
    *,
    include_index: bool = False,
    data_format: str | None = None,
) -> None:
    output = frame.reset_index() if include_index else frame
    write_records(
        path,
        output.where(pd.notna(output), None).to_dict("records"),
        fieldnames=output.columns.astype(str).tolist(),
        data_format=data_format,
    )


def write_series(
    path: Path,
    series: pd.Series,
    *,
    include_index: bool = False,
    data_format: str | None = None,
) -> None:
    write_frame(path, series.to_frame(), include_index=include_index, data_format=data_format)


def infer_data_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in _READERS:
        return suffix
    raise ValueError(f"Unsupported tabular data format for `{path}`.")


def _reader_for(path: Path, data_format: str | None) -> DictReader:
    return _READERS[_normalize_format(path, data_format)]


def _writer_for(path: Path, data_format: str | None) -> DictWriter:
    return _WRITERS[_normalize_format(path, data_format)]


def _normalize_format(path: Path, data_format: str | None) -> str:
    name = (data_format or infer_data_format(path)).lower()
    if name not in _READERS:
        supported = ", ".join(sorted(_READERS))
        raise ValueError(f"Unsupported tabular data format `{name}`. Supported: {supported}.")
    return name


def _normalize_csv_value(value: str | None) -> str | None:
    return None if value == "" else value


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        original = frame[column]
        values = pd.to_numeric(original, errors="coerce")
        present = original.notna()
        if present.any() and values[present].notna().all():
            frame[column] = values
    return frame
