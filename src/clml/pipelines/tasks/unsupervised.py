"""Unsupervised learning task runners: clustering, dimensionality, anomaly, density."""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split

from clml.config.settings import get_settings
from clml.constants import ANOMALY_NORMAL_TEST_SIZE
from clml.data.adapters import write_frame
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks._shared import (
    _anomaly_scores,
    _finish,
    _fit_predict,
    _optimize_unsupervised,
    _pipeline,
)
from clml.reporting.plots import plot_2d_projection


def run_clustering(ctx: RunContext) -> RunResult:
    x, y = ctx.bundle.x, ctx.bundle.y
    pipeline = _pipeline(ctx, x)
    best_params = _optimize_unsupervised(ctx, pipeline, x)
    pipeline.set_params(**best_params)
    labels = _fit_predict(pipeline, x)
    transformed = pipeline.named_steps["preprocess"].transform(x)
    metrics: dict = {"clusters": int(len(set(labels)) - (1 if -1 in labels else 0))}
    if 1 < len(set(labels)) < len(labels):
        metrics["silhouette"] = float(silhouette_score(transformed, labels))
    if y is not None:
        metrics["adjusted_rand"] = float(adjusted_rand_score(y, labels))
    write_frame(ctx.run_dir / "clusters.csv", pd.DataFrame({"cluster": labels}))
    plot_2d_projection(transformed, labels, ctx.run_dir / "clusters.png", ctx.spec.title)
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, best_params, {})


def run_dimensionality(ctx: RunContext) -> RunResult:
    x = ctx.bundle.x
    labels = ctx.bundle.y if ctx.bundle.y is not None else np.zeros(len(x))
    pipeline = _pipeline(ctx, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = pipeline.fit_transform(x)
    metrics: dict = {"components": int(embedding.shape[1])}
    model = pipeline.named_steps["model"]
    if hasattr(model, "explained_variance_ratio_"):
        metrics["explained_variance_ratio_sum"] = float(np.sum(model.explained_variance_ratio_))
    write_frame(
        ctx.run_dir / "embedding.csv",
        pd.DataFrame(embedding[:, :2], columns=["component_0", "component_1"]),
    )
    plot_2d_projection(embedding[:, :2], labels, ctx.run_dir / "embedding.png", ctx.spec.title)
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, {}, {})


def run_anomaly(ctx: RunContext) -> RunResult:
    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    normal_x = x[y == 0]
    x_train_normal, x_test_normal = train_test_split(
        normal_x, test_size=ANOMALY_NORMAL_TEST_SIZE, random_state=get_settings().random_state
    )
    x_test = pd.concat([x_test_normal, x[y == 1]], axis=0)
    y_test = pd.concat(
        [pd.Series(np.zeros(len(x_test_normal))), pd.Series(np.ones(int((y == 1).sum())))], axis=0
    )
    pipeline = _pipeline(ctx, x_train_normal)
    pipeline.fit(x_train_normal)
    raw_predictions = pipeline.predict(x_test)
    predictions = np.where(raw_predictions == -1, 1, 0)
    scores = _anomaly_scores(pipeline, x_test)
    metrics: dict = {
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, scores)),
    }
    write_frame(
        ctx.run_dir / "anomaly_scores.csv",
        pd.DataFrame({"actual": y_test, "predicted": predictions, "score": scores}),
    )
    transformed = pipeline.named_steps["preprocess"].transform(x_test)
    plot_2d_projection(transformed, predictions, ctx.run_dir / "anomalies.png", ctx.spec.title)
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, {}, {})


def run_density(ctx: RunContext) -> RunResult:
    x = ctx.bundle.x
    pipeline = _pipeline(ctx, x)
    best_params = _optimize_unsupervised(ctx, pipeline, x)
    pipeline.set_params(**best_params)
    pipeline.fit(x)
    scores = pipeline.score_samples(x)
    metrics: dict = {
        "mean_log_likelihood": float(np.mean(scores)),
        "std_log_likelihood": float(np.std(scores)),
    }
    write_frame(ctx.run_dir / "density_scores.csv", pd.DataFrame({"log_likelihood": scores}))
    transformed = pipeline.named_steps["preprocess"].transform(x)
    plot_2d_projection(transformed, scores, ctx.run_dir / "density.png", ctx.spec.title)
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, best_params, {})
