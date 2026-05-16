import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from clml.config.settings import get_settings
from clml.data.adapters import write_frame


@dataclass(frozen=True)
class RunRecord:
    method: str
    dataset: str
    task: str
    run_dir: Path
    metrics: dict[str, Any]
    best_params: dict[str, Any]
    comparisons: dict[str, Any]

    @property
    def timestamp(self) -> str:
        return self.run_dir.name


def find_runs(method: str | None = None) -> list[RunRecord]:
    settings = get_settings()
    roots = [settings.data_dir / method] if method else sorted(settings.data_dir.iterdir())
    records: list[RunRecord] = []
    for root in roots:
        if not root.is_dir():
            continue
        for run_json in sorted(root.glob("*/run.json")):
            records.append(_load_run(run_json))
    return sorted(records, key=lambda record: str(record.run_dir), reverse=True)


def last_run(method: str | None = None) -> RunRecord | None:
    runs = find_runs(method)
    return runs[0] if runs else None


def runs_frame(method: str | None = None) -> pd.DataFrame:
    rows = []
    for record in find_runs(method):
        metric_preview = {
            key: value
            for key, value in record.metrics.items()
            if isinstance(value, int | float | str)
        }
        rows.append(
            {
                "method": record.method,
                "dataset": record.dataset,
                "task": record.task,
                "timestamp": record.timestamp,
                "run_dir": str(record.run_dir),
                **metric_preview,
            }
        )
    return pd.DataFrame(rows)


def write_run_all_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    write_frame(output_dir / "summary.csv", pd.DataFrame(rows))


def _load_run(path: Path) -> RunRecord:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return RunRecord(
        method=raw.get("method", path.parent.parent.name),
        dataset=raw.get("dataset", ""),
        task=raw.get("task", ""),
        run_dir=Path(raw.get("run_dir", path.parent)),
        metrics=raw.get("metrics", {}),
        best_params=raw.get("best_params", {}),
        comparisons=raw.get("comparisons", {}),
    )
