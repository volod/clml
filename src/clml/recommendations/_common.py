import pandas as pd

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
    return len(values) >= 10 and bool(values.skew() > 1.0)


def _has_outliers(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 10:
        return False
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return False
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return bool(((values < lower) | (values > upper)).any())


def _is_imbalanced(series: pd.Series) -> bool:
    frequencies = series.dropna().value_counts(normalize=True)
    return len(frequencies) > 1 and bool(frequencies.min() < 0.2)
