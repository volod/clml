"""Shared helpers used by all task runner modules."""

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from clml.config.settings import get_settings
from clml.constants import CV_FOLDS
from clml.data.adapters import write_frame
from clml.data.catalog import DatasetBundle
from clml.features.rules import RuleBasedFeatureEngineer
from clml.methods.registry import MethodSpec
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines._metrics import interpret_metrics
from clml.pipelines.preprocessing import build_model_pipeline
from clml.reporting.plots import plot_named_bars


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _finish(
    spec: MethodSpec,
    bundle: DatasetBundle,
    run_dir: Path,
    fitted_model: Any,
    metrics: dict[str, float | int | str],
    best_params: dict[str, Any],
    comparisons: dict[str, dict[str, float | int | str]],
) -> RunResult:
    from dataclasses import asdict

    result = RunResult(
        method=spec.name,
        dataset=bundle.info.name,
        task=spec.task,
        run_dir=run_dir,
        metrics=metrics,
        best_params=best_params,
        comparisons=comparisons,
    )
    _write_json(
        run_dir / "run.json",
        {
            **asdict(result),
            "run_dir": str(run_dir),
            "method_title": spec.title,
            "guide_section": spec.guide_section,
            "dataset_description": bundle.info.description,
            "metric_interpretation": interpret_metrics(spec.task, metrics),
            "leakage_control": "Preprocessing is inside sklearn Pipeline and optimized through CV.",
        },
    )
    if comparisons:
        _write_json(run_dir / "third_party_comparisons.json", comparisons)
    joblib.dump(fitted_model, run_dir / "model.joblib")
    if isinstance(fitted_model, Pipeline) and mlflow.active_run() is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mlflow.sklearn.log_model(
                    sk_model=fitted_model,
                    artifact_path="registered_model",
                    registered_model_name=spec.name,
                )
            except Exception:
                pass
    return result


def _pipeline(ctx: RunContext, x: pd.DataFrame) -> Pipeline:
    return build_model_pipeline(
        x,
        ctx.spec.estimator_factory(get_settings().random_state),
        nonnegative=ctx.spec.needs_nonnegative,
        feature_engineering=ctx.feature_engineering,
        feature_rules=ctx.feature_rules,
    )


def _optimize_supervised(
    ctx: RunContext,
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    task: str,
) -> dict[str, Any]:
    if ctx.spec.param_space is None or ctx.trials <= 0:
        return {}
    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    cv: Any = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=get_settings().random_state
    )
    if task == "regression":
        cv = CV_FOLDS

    def objective(trial: optuna.Trial) -> float:
        candidate = clone(pipeline)
        candidate.set_params(**ctx.spec.param_space(trial))
        scores = cross_val_score(candidate, x, y, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ctx.trials, show_progress_bar=False)
    return {f"model__{key}": value for key, value in study.best_params.items()}


def _optimize_unsupervised(ctx: RunContext, pipeline: Pipeline, x: pd.DataFrame) -> dict[str, Any]:
    if ctx.spec.param_space is None or ctx.trials <= 0:
        return {}

    def objective(trial: optuna.Trial) -> float:
        candidate = clone(pipeline)
        candidate.set_params(**ctx.spec.param_space(trial))
        labels = _fit_predict(candidate, x)
        transformed = candidate.named_steps["preprocess"].transform(x)
        if len(set(labels)) <= 1 or len(set(labels)) >= len(labels):
            return -1.0
        return float(silhouette_score(transformed, labels))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ctx.trials, show_progress_bar=False)
    return {f"model__{key}": value for key, value in study.best_params.items()}


def _fit_predict(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    model = pipeline.named_steps["model"]
    if hasattr(model, "fit_predict"):
        return pipeline.fit_predict(x)
    pipeline.fit(x)
    if hasattr(pipeline, "predict"):
        return pipeline.predict(x)
    if hasattr(model, "labels_"):
        return model.labels_
    raise ValueError("Estimator does not expose labels or predictions.")


def _run_third_party_comparisons(
    ctx: RunContext,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict[str, float | int | str]]:
    comparisons: dict[str, dict[str, float | int | str]] = {}
    for name, factory in ctx.spec.third_party_factories.items():
        pipeline = build_model_pipeline(
            x_train,
            factory(get_settings().random_state),
            feature_engineering=ctx.feature_engineering,
            feature_rules=ctx.feature_rules,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        if ctx.spec.task == "classification":
            comparisons[name] = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            }
        else:
            comparisons[name] = _regression_metrics(y_test, predictions)
    return comparisons


def _regression_metrics(y_true: Any, predictions: Any) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _anomaly_scores(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "decision_function"):
        return -pipeline.decision_function(x)
    if hasattr(pipeline, "score_samples"):
        return -pipeline.score_samples(x)
    return np.zeros(len(x))


def _feature_engineer(ctx: RunContext, x: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply feature engineering if enabled; return (transformed_x, pipeline_steps)."""
    if not ctx.feature_engineering:
        return x, []
    engineer = RuleBasedFeatureEngineer(ctx.feature_rules)
    return engineer.fit_transform(x), [("features", engineer)]


def _write_statsmodels_artifacts(fitted: Any, run_dir: Path) -> None:
    coefficients = pd.DataFrame({"coef": fitted.params})
    if hasattr(fitted, "bse"):
        coefficients["std_error"] = fitted.bse
    if hasattr(fitted, "pvalues"):
        coefficients["p_value"] = fitted.pvalues
    write_frame(run_dir / "coefficients.csv", coefficients, include_index=True)
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(str(fitted.summary()))
    plot_named_bars(
        coefficients.index.tolist()[:20],
        coefficients["coef"].tolist()[:20],
        run_dir / "coefficients.png",
        title="Model coefficients",
        ylabel="Coefficient",
    )
