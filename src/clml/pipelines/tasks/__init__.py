"""Task runner registry — maps task name to its runner function."""

from collections.abc import Callable

from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks.feature_selection import run_feature_selection
from clml.pipelines.tasks.incremental import run_incremental_classification
from clml.pipelines.tasks.optimization import (
    run_cvxpy_portfolio,
    run_cvxpy_quadratic,
    run_linear_programming,
    run_nonlinear_optimization,
)
from clml.pipelines.tasks.specialized import (
    run_categorical_encoding,
    run_imbalanced_classification,
)
from clml.pipelines.tasks.statistical import (
    run_statsmodels_classification,
    run_statsmodels_regression,
)
from clml.pipelines.tasks.supervised import run_classification, run_regression, run_regression_1d
from clml.pipelines.tasks.survival import run_survival_cox, run_survival_kaplan_meier
from clml.pipelines.tasks.timeseries import run_timeseries
from clml.pipelines.tasks.unsupervised import (
    run_anomaly,
    run_clustering,
    run_density,
    run_dimensionality,
)

TaskRunner = Callable[[RunContext], RunResult]

TASK_RUNNERS: dict[str, TaskRunner] = {
    # Supervised
    "classification": run_classification,
    "feature_selection": run_feature_selection,
    "incremental_classification": run_incremental_classification,
    "regression": run_regression,
    "regression_1d": run_regression_1d,
    # Unsupervised
    "clustering": run_clustering,
    "dimensionality": run_dimensionality,
    "anomaly": run_anomaly,
    "density": run_density,
    # Statistical (statsmodels)
    "statsmodels_regression": run_statsmodels_regression,
    "statsmodels_classification": run_statsmodels_classification,
    # Specialized
    "imbalanced_classification": run_imbalanced_classification,
    "categorical_encoding": run_categorical_encoding,
    # Optimisation
    "linear_programming": run_linear_programming,
    "nonlinear_optimization": run_nonlinear_optimization,
    "cvxpy_portfolio": run_cvxpy_portfolio,
    "cvxpy_quadratic": run_cvxpy_quadratic,
    # Survival
    "survival_cox": run_survival_cox,
    "survival_kaplan_meier": run_survival_kaplan_meier,
    # Time series
    "timeseries": run_timeseries,
}

__all__ = ["TASK_RUNNERS", "TaskRunner"]
