"""Feature selection task runner."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from clml.config.settings import get_settings
from clml.constants import (
    ARTIFACT_FEATURE_SELECTION_DIAGNOSTICS_CSV,
    ARTIFACT_SELECTED_FEATURES_CSV,
    LOGISTIC_MAX_ITER,
    TEST_SIZE,
)
from clml.data.adapters import write_frame
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.preprocessing import build_preprocessor
from clml.pipelines.tasks._shared import (
    _binary_probabilities,
    _classification_metrics,
    _feature_engineer,
    _finish,
    _write_classification_artifacts,
)
from clml.reporting.plots import plot_named_bars


def run_feature_selection(ctx: RunContext) -> RunResult:
    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state, stratify=y
    )
    x_for_preprocessor, feature_steps = _feature_engineer(ctx, x_train)
    selector = ctx.spec.estimator_factory(get_settings().random_state)
    downstream = LogisticRegression(
        max_iter=LOGISTIC_MAX_ITER,
        random_state=get_settings().random_state,
    )
    pipeline = Pipeline(
        steps=[
            *feature_steps,
            ("preprocess", build_preprocessor(x_for_preprocessor)),
            ("selector", selector),
            ("model", downstream),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = _binary_probabilities(pipeline, x_test, y)
    selected = _selected_features(pipeline)
    metrics: dict[str, float | int] = {
        "n_features_selected": len(selected),
        "downstream_score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        **_classification_metrics(y_test, predictions, probabilities),
    }
    _write_classification_artifacts(ctx.run_dir, y_test, predictions, probabilities)
    write_frame(ctx.run_dir / ARTIFACT_SELECTED_FEATURES_CSV, pd.DataFrame({"feature": selected}))
    _write_feature_diagnostics(pipeline, ctx.run_dir)
    plot_named_bars(
        ["selected", "available"],
        [metrics["n_features_selected"], _available_feature_count(pipeline)],
        ctx.run_dir / "feature_selection_counts.png",
        title="Feature selection count",
        ylabel="Features",
    )
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, {}, {})


def _available_feature_count(pipeline: Pipeline) -> int:
    return int(len(pipeline.named_steps["preprocess"].get_feature_names_out()))


def _selected_features(pipeline: Pipeline) -> list[str]:
    names = np.asarray(pipeline.named_steps["preprocess"].get_feature_names_out())
    selector = pipeline.named_steps["selector"]
    if hasattr(selector, "get_support"):
        return names[selector.get_support()].tolist()
    return names.tolist()


def _write_feature_diagnostics(pipeline: Pipeline, run_dir) -> None:
    names = np.asarray(pipeline.named_steps["preprocess"].get_feature_names_out())
    selector = pipeline.named_steps["selector"]
    frame = pd.DataFrame({"feature": names})
    if hasattr(selector, "get_support"):
        frame["selected"] = selector.get_support()
    if hasattr(selector, "scores_"):
        frame["score"] = selector.scores_
    if hasattr(selector, "ranking_"):
        frame["ranking"] = selector.ranking_
    write_frame(run_dir / ARTIFACT_FEATURE_SELECTION_DIAGNOSTICS_CSV, frame)
