import pandas as pd

from clml.constants import (
    CORRELATION_WARN_THRESHOLD_INFERENCE,
    DATETIME_PARSE_MIN_NOTNA_FRACTION,
    TARGET_CONTINUOUS_MIN_NUNIQUE,
)


def _infer_time_column(frame: pd.DataFrame) -> str | None:
    candidates = []
    for column in frame.columns:
        name = column.lower()
        if "date" in name or name.endswith("_at") or name in {"day", "month", "year"}:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().mean() > DATETIME_PARSE_MIN_NOTNA_FRACTION:
                candidates.append(column)
    return candidates[0] if candidates else None


def _infer_target_column(frame: pd.DataFrame, time_column: str | None) -> str | None:
    explicit_names = ["target", "label", "y", "class", "event"]
    for name in explicit_names:
        matches = [col for col in frame.columns if col.lower() == name]
        if matches:
            return matches[0]
    numeric = [
        col
        for col in frame.select_dtypes(include="number").columns
        if col != time_column and frame[col].nunique(dropna=True) > 2
    ]
    if not numeric:
        return None
    monotonic_scores = []
    for column in numeric:
        series = frame[column].dropna()
        is_monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing
        monotonic_scores.append((float(is_monotonic), column))
    if any(score for score, _ in monotonic_scores):
        return max(monotonic_scores, key=lambda item: (item[0], frame[item[1]].abs().max()))[1]
    return max(numeric, key=lambda col: frame[col].nunique(dropna=True))


def _infer_task(frame: pd.DataFrame, target_column: str | None, time_column: str | None) -> str:
    if time_column:
        numeric = frame.select_dtypes(include="number")
        monotonic_count = sum(
            numeric[col].dropna().is_monotonic_increasing
            or numeric[col].dropna().is_monotonic_decreasing
            for col in numeric.columns
        )
        if monotonic_count >= max(2, len(numeric.columns) // 3):
            return "time_series_cumulative_regression"
    if target_column is None:
        return "unsupervised"
    target = frame[target_column]
    is_continuous_numeric = (
        pd.api.types.is_numeric_dtype(target)
        and target.nunique(dropna=True) > TARGET_CONTINUOUS_MIN_NUNIQUE
    )
    if time_column and is_continuous_numeric:
        return "timeseries"
    if is_continuous_numeric:
        return "regression"
    return "classification"


def _bundle_task(inferred_task: str) -> str:
    if "classification" in inferred_task:
        return "classification"
    if "regression" in inferred_task:
        return "regression"
    if inferred_task == "timeseries":
        return "regression"
    return "clustering"


def _profile_notes(
    frame: pd.DataFrame, target_column: str | None, time_column: str | None
) -> list[str]:
    notes = [
        f"Rows: {len(frame)}; columns: {len(frame.columns)}.",
        f"Numeric columns: {len(frame.select_dtypes(include='number').columns)}.",
        f"Categorical/text columns: {len(frame.select_dtypes(exclude='number').columns)}.",
    ]
    missing = frame.isna().mean().sort_values(ascending=False)
    if not missing.empty and missing.iloc[0] > 0:
        notes.append(f"Highest missing fraction is {missing.iloc[0]:.1%} in `{missing.index[0]}`.")
    if time_column:
        notes.append(f"`{time_column}` looks like a time column; use time-aware validation.")
    if target_column:
        notes.append(f"`{target_column}` is selected as target; override if this is not desired.")
    numeric = frame.select_dtypes(include="number")
    if (
        len(numeric.columns) >= 2
        and numeric.corr(numeric_only=True).abs().max().max()
        > CORRELATION_WARN_THRESHOLD_INFERENCE
    ):
        notes.append(
            "Several numeric columns are highly correlated; cumulative counters may dominate."
        )
    return notes
