import pandas as pd

from clml.constants import (
    IMBALANCE_MINORITY_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_IQR_Q1,
    OUTLIER_IQR_Q3,
    OUTLIER_MIN_SAMPLES,
    SKEW_THRESHOLD,
)
from clml.recommendations._models import MethodRecommendation


def _rec(
    method: str,
    why_suggested: str,
    use_case: str,
    expected_result: str,
    interpretation: str,
    target_column: str | None,
    priority: str,
) -> MethodRecommendation:
    target_note = f" --target-column {target_column}" if target_column else ""
    return MethodRecommendation(
        method=method,
        why_suggested=why_suggested,
        use_case=use_case,
        expected_result=expected_result,
        interpretation=interpretation,
        command=f"clml run-data --method {method} --data <path>{target_note} --trials 0",
        priority=priority,
    )


def _largest_numeric_column(frame: pd.DataFrame) -> str | None:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return None
    return numeric.abs().max().sort_values(ascending=False).index[0]


def _is_nonnegative(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return not values.empty and bool((values >= 0).all())


def _is_strictly_positive(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return not values.empty and bool((values > 0).all())


def _is_right_skewed(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return len(values) >= OUTLIER_MIN_SAMPLES and bool(values.skew() > SKEW_THRESHOLD)


def _has_outliers(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < OUTLIER_MIN_SAMPLES:
        return False
    q1 = values.quantile(OUTLIER_IQR_Q1)
    q3 = values.quantile(OUTLIER_IQR_Q3)
    iqr = q3 - q1
    if iqr <= 0:
        return False
    lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
    upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
    return bool(((values < lower) | (values > upper)).any())


def _is_imbalanced(series: pd.Series) -> bool:
    frequencies = series.dropna().value_counts(normalize=True)
    return len(frequencies) > 1 and bool(frequencies.min() < IMBALANCE_MINORITY_THRESHOLD)
