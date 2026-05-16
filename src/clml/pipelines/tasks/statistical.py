"""Statsmodels task runners: OLS/GLM/RLM regression and Logit classification."""

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from clml.config.settings import get_settings
from clml.constants import SM_HIGH_CARDINALITY_THRESHOLD, SM_LOGIT_MAX_ITER, TEST_SIZE
from clml.data.adapters import write_frame
from clml.features.rules import RuleBasedFeatureEngineer
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks._shared import (
    _finish,
    _regression_metrics,
    _write_json,
    _write_statsmodels_artifacts,
)
from clml.reporting.plots import plot_confusion_matrix, plot_regression_predictions


def _statsmodels_design(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm

    # Drop object columns with high cardinality (dates, IDs) to avoid singular Hessians.
    high_card = [
        col
        for col in x_train.select_dtypes(exclude="number").columns
        if x_train[col].nunique() > SM_HIGH_CARDINALITY_THRESHOLD
    ]
    x_train = x_train.drop(columns=high_card)
    x_test = x_test.drop(columns=high_card, errors="ignore")

    train = pd.get_dummies(x_train, drop_first=True, dtype=float)
    test = pd.get_dummies(x_test, drop_first=True, dtype=float)
    test = test.reindex(columns=train.columns, fill_value=0.0)
    train = sm.add_constant(train, has_constant="add")
    test = sm.add_constant(test, has_constant="add")
    return train.astype(float), test.astype(float)


def _make_sm_regression_fitter(name: str) -> Callable[[Any, Any], Any]:
    """Return a callable that fits the correct statsmodels estimator by method name."""
    import statsmodels.api as sm

    _fitters: dict[str, Callable[[Any, Any], Any]] = {
        "statsmodels_ols": lambda y, x: sm.OLS(y, x).fit(),
        "statsmodels_glm_regression": lambda y, x: sm.GLM(
            y, x, family=sm.families.Gaussian()
        ).fit(),
        "statsmodels_robust_regression": lambda y, x: sm.RLM(
            y, x, M=sm.robust.norms.HuberT()
        ).fit(),
    }
    factory = _fitters.get(name)
    if factory is None:
        raise ValueError(f"Unsupported statsmodels regression method: {name}")
    return factory


def run_statsmodels_regression(ctx: RunContext) -> RunResult:
    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state
    )
    if ctx.feature_engineering:
        engineer = RuleBasedFeatureEngineer(ctx.feature_rules)
        x_train = engineer.fit_transform(x_train)
        x_test = engineer.transform(x_test)
    train_design, test_design = _statsmodels_design(x_train, x_test)
    fitter = _make_sm_regression_fitter(ctx.spec.name)
    fitted = fitter(y_train, train_design)
    predictions = fitted.predict(test_design)
    metrics = _regression_metrics(y_test, predictions)
    write_frame(
        ctx.run_dir / "predictions.csv", pd.DataFrame({"actual": y_test, "predicted": predictions})
    )
    _write_statsmodels_artifacts(fitted, ctx.run_dir)
    plot_regression_predictions(y_test, predictions, ctx.run_dir / "prediction_scatter.png")
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, fitted, metrics, {}, {})


def run_statsmodels_classification(ctx: RunContext) -> RunResult:
    import statsmodels.api as sm

    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state, stratify=y
    )
    if ctx.feature_engineering:
        engineer = RuleBasedFeatureEngineer(ctx.feature_rules)
        x_train = engineer.fit_transform(x_train)
        x_test = engineer.transform(x_test)
    train_design, test_design = _statsmodels_design(x_train, x_test)
    fitted = sm.Logit(y_train, train_design).fit(disp=False, maxiter=SM_LOGIT_MAX_ITER)
    probabilities = fitted.predict(test_design)
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    write_frame(
        ctx.run_dir / "predictions.csv",
        pd.DataFrame({"actual": y_test, "predicted": predictions, "probability": probabilities}),
    )
    _write_json(
        ctx.run_dir / "classification_report.json",
        classification_report(y_test, predictions, output_dict=True),
    )
    _write_statsmodels_artifacts(fitted, ctx.run_dir)
    plot_confusion_matrix(y_test, predictions, ctx.run_dir / "confusion_matrix.png")
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, fitted, metrics, {}, {})
