"""Shared data types for the pipeline execution layer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clml.data.catalog import DatasetBundle
from clml.methods.registry import MethodSpec


@dataclass(frozen=True)
class RunResult:
    method: str
    dataset: str
    task: str
    run_dir: Path
    metrics: dict[str, float | int | str]
    best_params: dict[str, Any]
    comparisons: dict[str, dict[str, float | int | str]]


@dataclass(frozen=True)
class RunContext:
    """Shared state threaded through the entire method execution pipeline."""

    spec: MethodSpec
    bundle: DatasetBundle
    run_dir: Path
    trials: int
    feature_engineering: bool
    feature_rules: list[dict[str, Any]] = field(default_factory=list)
