import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RuleBasedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for explicit, auditable feature rules."""

    def __init__(self, rules: list[dict[str, Any]] | None = None):
        self.rules = rules or []

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        for rule in self.rules:
            frame = _apply_rule(frame, rule)
        return frame


def load_rules(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        return payload
    return list(payload.get("rules", []))


def _apply_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    kind = rule.get("type")
    if kind == "datetime_parts":
        return _datetime_parts(frame, rule)
    if kind == "numeric_bins":
        return _numeric_bins(frame, rule)
    if kind == "ratio":
        return _ratio(frame, rule)
    if kind == "product":
        return _product(frame, rule)
    if kind == "log1p":
        return _log1p(frame, rule)
    if kind == "clip":
        return _clip(frame, rule)
    if kind == "text_length":
        return _text_length(frame, rule)
    raise ValueError(f"Unsupported feature rule type: {kind}")


def _datetime_parts(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    column = rule["column"]
    if column not in frame.columns:
        return frame
    values = pd.to_datetime(frame[column], errors="coerce")
    prefix = rule.get("prefix", column)
    parts = rule.get("parts", ["day_of_week", "month", "quarter", "is_weekend"])
    if "day_of_week" in parts:
        frame[f"{prefix}_day_of_week"] = values.dt.dayofweek
    if "month" in parts:
        frame[f"{prefix}_month"] = values.dt.month
    if "quarter" in parts:
        frame[f"{prefix}_quarter"] = values.dt.quarter
    if "year" in parts:
        frame[f"{prefix}_year"] = values.dt.year
    if "is_weekend" in parts:
        frame[f"{prefix}_is_weekend"] = values.dt.dayofweek.isin([5, 6]).astype(int)
    if rule.get("drop_original", True):
        frame = frame.drop(columns=[column])
    return frame


def _numeric_bins(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    column = rule["column"]
    if column not in frame.columns:
        return frame
    output = rule.get("output_column", f"{column}_bin")
    labels = rule.get("labels")
    frame[output] = pd.cut(
        frame[column],
        bins=rule["bins"],
        labels=labels,
        include_lowest=True,
        right=rule.get("right", False),
    ).astype("object")
    if rule.get("drop_original", False):
        frame = frame.drop(columns=[column])
    return frame


def _ratio(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    numerator = rule["numerator"]
    denominator = rule["denominator"]
    if numerator not in frame.columns or denominator not in frame.columns:
        return frame
    output = rule.get("output_column", f"{numerator}_to_{denominator}")
    epsilon = float(rule.get("epsilon", 1e-9))
    frame[output] = frame[numerator] / (frame[denominator].replace(0, np.nan) + epsilon)
    return frame


def _product(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    columns = rule["columns"]
    if any(column not in frame.columns for column in columns):
        return frame
    output = rule.get("output_column", "_x_".join(columns))
    value = pd.Series(1.0, index=frame.index)
    for column in columns:
        value = value * frame[column]
    frame[output] = value
    return frame


def _log1p(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    column = rule["column"]
    if column not in frame.columns:
        return frame
    output = rule.get("output_column", f"log1p_{column}")
    minimum = float(rule.get("clip_min", 0.0))
    frame[output] = np.log1p(frame[column].clip(lower=minimum))
    if rule.get("drop_original", False):
        frame = frame.drop(columns=[column])
    return frame


def _clip(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    column = rule["column"]
    if column not in frame.columns:
        return frame
    output = rule.get("output_column", column)
    frame[output] = frame[column].clip(lower=rule.get("lower"), upper=rule.get("upper"))
    return frame


def _text_length(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    column = rule["column"]
    if column not in frame.columns:
        return frame
    output = rule.get("output_column", f"{column}_length")
    frame[output] = frame[column].astype(str).str.len()
    return frame
