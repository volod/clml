"""Supervised learning task runners: classification and regression."""

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from clml.config.settings import get_settings
from clml.constants import TEST_SIZE
from clml.data.adapters import write_frame
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks._shared import (
    _finish,
    _optimize_supervised,
    _pipeline,
    _regression_metrics,
    _run_third_party_comparisons,
    _write_json,
)
from clml.reporting.plots import plot_confusion_matrix, plot_regression_predictions


def run_classification(ctx: RunContext) -> RunResult:
    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state, stratify=y
    )
    pipeline = _pipeline(ctx, x_train)
    best_params = _optimize_supervised(ctx, pipeline, x_train, y_train, "classification")
    pipeline.set_params(**best_params)
    pipeline.fit(x_train, y_train)
    predictions = _predict_regression(pipeline, x_test)
    metrics: dict = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }
    if hasattr(pipeline, "predict_proba") and y.nunique() == 2:
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    _write_json(
        ctx.run_dir / "classification_report.json",
        classification_report(y_test, predictions, output_dict=True),
    )
    write_frame(
        ctx.run_dir / "predictions.csv", pd.DataFrame({"actual": y_test, "predicted": predictions})
    )
    plot_confusion_matrix(y_test, predictions, ctx.run_dir / "confusion_matrix.png")
    comparisons = _run_third_party_comparisons(ctx, x_train, x_test, y_train, y_test)
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, best_params, comparisons)


def run_regression(ctx: RunContext) -> RunResult:
    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state
    )
    pipeline = _pipeline(ctx, x_train)
    best_params = _optimize_supervised(ctx, pipeline, x_train, y_train, "regression")
    pipeline.set_params(**best_params)
    pipeline.fit(x_train, y_train)
    predictions = _predict_regression(pipeline, x_test)
    metrics = _regression_metrics(y_test, predictions)
    write_frame(
        ctx.run_dir / "predictions.csv", pd.DataFrame({"actual": y_test, "predicted": predictions})
    )
    _write_distribution_predictions(pipeline, x_test, y_test, predictions, ctx.run_dir)
    plot_regression_predictions(y_test, predictions, ctx.run_dir / "prediction_scatter.png")
    comparisons = _run_third_party_comparisons(ctx, x_train, x_test, y_train, y_test)
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, best_params, comparisons)


def run_regression_1d(ctx: RunContext) -> RunResult:
    x = ctx.bundle.x[[ctx.bundle.info.feature_columns[0]]]
    y = ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state
    )
    model = ctx.spec.estimator_factory(get_settings().random_state)
    model.fit(x_train.iloc[:, 0], y_train)
    predictions = model.predict(x_test.iloc[:, 0])
    metrics = _regression_metrics(y_test, predictions)
    write_frame(
        ctx.run_dir / "predictions.csv", pd.DataFrame({"actual": y_test, "predicted": predictions})
    )
    plot_regression_predictions(y_test, predictions, ctx.run_dir / "prediction_scatter.png")
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, model, metrics, {}, {})


def _write_distribution_predictions(pipeline, x_test, y_test, predictions, run_dir) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "pred_dist"):
        return
    transformed = pipeline.named_steps["preprocess"].transform(x_test)
    try:
        distribution = model.pred_dist(transformed)
    except Exception:
        return
    frame = pd.DataFrame({"actual": y_test, "predicted": predictions})
    params = getattr(distribution, "params", {})
    if isinstance(params, dict):
        for name, values in params.items():
            frame[f"distribution_{name}"] = values
    for quantile, column in [(0.05, "p05"), (0.50, "p50"), (0.95, "p95")]:
        if hasattr(distribution, "ppf"):
            try:
                frame[column] = distribution.ppf(quantile)
            except Exception:
                continue
    write_frame(run_dir / "distribution_predictions.csv", frame)


def _predict_regression(pipeline, x_test):
    try:
        return pipeline.predict(x_test)
    except NotFittedError:
        transformed = pipeline.named_steps["preprocess"].transform(x_test)
        return pipeline.named_steps["model"].predict(transformed)
