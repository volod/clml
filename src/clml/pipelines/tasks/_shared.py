"""Shared helpers used by all task runner modules."""

import json
import logging as stdlib_logging
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import skops.io as skops_io
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from clml.config.log import get_logger
from clml.config.settings import get_settings
from clml.constants import (
    ARTIFACT_CLASSIFICATION_REPORT_JSON,
    ARTIFACT_CONFUSION_MATRIX_PNG,
    ARTIFACT_MODEL_JOBLIB,
    ARTIFACT_MODEL_SKOPS,
    ARTIFACT_PREDICTIONS_CSV,
    ARTIFACT_RUN_JSON,
    ARTIFACT_THIRD_PARTY_JSON,
    CV_FOLDS,
)
from clml.data.adapters import write_frame
from clml.data.catalog import DatasetBundle
from clml.features.rules import RuleBasedFeatureEngineer
from clml.methods.registry import MethodSpec
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines._metrics import interpret_metrics
from clml.pipelines.preprocessing import build_model_pipeline
from clml.reporting.plots import plot_confusion_matrix, plot_named_bars

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _pip_requirements() -> list[str]:
    """Export pinned requirements from the uv lock file once per process."""
    try:
        out = subprocess.check_output(
            ["uv", "export", "--frozen", "--no-hashes", "--quiet"],
            text=True,
        )
        reqs = [
            line.strip()
            for line in out.splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith("-e ")
        ]
        if reqs:
            return reqs
    except Exception:
        pass
    return []


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
        run_dir / ARTIFACT_RUN_JSON,
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
        _write_json(run_dir / ARTIFACT_THIRD_PARTY_JSON, comparisons)
    try:
        skops_io.dump(fitted_model, run_dir / ARTIFACT_MODEL_SKOPS)
    except Exception as exc:
        logger.debug(
            "skops serialization failed for %s (%s) — falling back to joblib",
            spec.name,
            exc,
        )
        joblib.dump(fitted_model, run_dir / ARTIFACT_MODEL_JOBLIB)
    if isinstance(fitted_model, Pipeline) and mlflow.active_run() is not None:
        reqs = _pip_requirements()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mlflow.sklearn.log_model(
                    sk_model=fitted_model,
                    name="registered_model",
                    registered_model_name=spec.name,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
                    pip_requirements=reqs or None,
                )
            except Exception:
                _mlflow_sklearn_log = stdlib_logging.getLogger("mlflow.sklearn")
                _orig_level = _mlflow_sklearn_log.level
                _mlflow_sklearn_log.setLevel(stdlib_logging.ERROR)
                try:
                    mlflow.sklearn.log_model(
                        sk_model=fitted_model,
                        name="registered_model",
                        registered_model_name=spec.name,
                        pip_requirements=reqs or None,
                    )
                except Exception:
                    pass
                finally:
                    _mlflow_sklearn_log.setLevel(_orig_level)
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
    logger.info("optuna tuning: method=%s task=%s trials=%d", ctx.spec.name, task, ctx.trials)
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
    best = {f"model__{key}": value for key, value in study.best_params.items()}
    logger.debug("best params: %s (value=%.4f)", best, study.best_value)
    return best


def _optimize_unsupervised(ctx: RunContext, pipeline: Pipeline, x: pd.DataFrame) -> dict[str, Any]:
    if ctx.spec.param_space is None or ctx.trials <= 0:
        return {}
    logger.info("optuna tuning (unsupervised): method=%s trials=%d", ctx.spec.name, ctx.trials)

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
    best = {f"model__{key}": value for key, value in study.best_params.items()}
    logger.debug("best params: %s (value=%.4f)", best, study.best_value)
    return best


def _optimize_density(ctx: RunContext, pipeline: Pipeline, x: pd.DataFrame) -> dict[str, Any]:
    if ctx.spec.param_space is None or ctx.trials <= 0:
        return {}
    logger.info("optuna tuning (density): method=%s trials=%d", ctx.spec.name, ctx.trials)

    def objective(trial: optuna.Trial) -> float:
        candidate = clone(pipeline)
        candidate.set_params(**ctx.spec.param_space(trial))
        candidate.fit(x)
        return float(np.mean(candidate.score_samples(x)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ctx.trials, show_progress_bar=False)
    best = {f"model__{key}": value for key, value in study.best_params.items()}
    logger.debug("best params: %s (value=%.4f)", best, study.best_value)
    return best


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
        logger.debug("running third-party comparison: %s", name)
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
            comparisons[name] = _classification_metrics(y_test, predictions)
        else:
            comparisons[name] = _regression_metrics(y_test, predictions)
    return comparisons


def _classification_metrics(
    y_true: Any,
    predictions: Any,
    probabilities: Any | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }
    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def _binary_probabilities(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> Any | None:
    if not hasattr(pipeline, "predict_proba") or y.nunique() != 2:
        return None
    return pipeline.predict_proba(x)[:, 1]


def _write_classification_artifacts(
    run_dir: Path,
    y_true: Any,
    predictions: Any,
    probabilities: Any | None = None,
) -> None:
    output = {"actual": y_true, "predicted": predictions}
    if probabilities is not None:
        output["probability"] = probabilities
    write_frame(run_dir / ARTIFACT_PREDICTIONS_CSV, pd.DataFrame(output))
    _write_json(
        run_dir / ARTIFACT_CLASSIFICATION_REPORT_JSON,
        classification_report(y_true, predictions, output_dict=True, zero_division=0),
    )
    plot_confusion_matrix(y_true, predictions, run_dir / ARTIFACT_CONFUSION_MATRIX_PNG)


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
