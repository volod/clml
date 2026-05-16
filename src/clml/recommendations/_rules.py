import pandas as pd

from clml.recommendations._models import MethodRecommendation
from clml.recommendations._supervised import recommend_classification, recommend_regression
from clml.recommendations._time_series import (
    recommend_cumulative_series,
    recommend_timeseries,
)
from clml.recommendations._unsupervised import recommend_unsupervised


def _recommend_methods(
    frame: pd.DataFrame,
    *,
    target_column: str | None,
    time_column: str | None,
    inferred_task: str,
) -> list[MethodRecommendation]:
    if inferred_task == "time_series_cumulative_regression":
        return recommend_cumulative_series(frame, target_column)
    if inferred_task == "timeseries":
        return recommend_timeseries(frame, target_column)
    if inferred_task == "regression":
        return recommend_regression(frame, target_column)
    if inferred_task == "classification":
        return recommend_classification(frame, target_column, time_column)
    return recommend_unsupervised()
