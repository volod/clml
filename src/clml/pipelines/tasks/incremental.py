"""Incremental / streaming task runners."""

import pandas as pd

from clml.config.settings import get_settings
from clml.data.adapters import write_frame
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks._shared import _feature_engineer, _finish
from clml.reporting.plots import plot_named_bars


def run_incremental_classification(ctx: RunContext) -> RunResult:
    from river import metrics

    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_stream, _ = _feature_engineer(ctx, x)
    model = ctx.spec.estimator_factory(get_settings().random_state)
    accuracy = metrics.Accuracy()
    f1 = metrics.F1()
    rows: list[dict[str, float | int | str | None]] = []

    for idx, (features, actual) in enumerate(zip(x_stream.to_dict("records"), y, strict=True)):
        actual_value = int(actual)
        predicted = model.predict_one(features)
        predicted_value = int(predicted) if predicted is not None else 0
        probability = _positive_probability(model, features)
        accuracy.update(actual_value, predicted_value)
        f1.update(actual_value, predicted_value)
        rows.append(
            {
                "sample": idx,
                "actual": actual_value,
                "predicted": predicted_value,
                "probability_1": probability,
                "prequential_accuracy": float(accuracy.get()),
                "prequential_f1": float(f1.get()),
            }
        )
        model.learn_one(features, actual_value)

    predictions = pd.DataFrame(rows)
    write_frame(ctx.run_dir / "prequential_predictions.csv", predictions)
    write_frame(
        ctx.run_dir / "prequential_metrics.csv",
        predictions[["sample", "prequential_accuracy", "prequential_f1"]],
    )
    metrics_payload = {
        "samples": len(predictions),
        "accuracy": float(accuracy.get()),
        "f1": float(f1.get()),
    }
    plot_named_bars(
        ["accuracy", "f1"],
        [metrics_payload["accuracy"], metrics_payload["f1"]],
        ctx.run_dir / "incremental_metrics.png",
        title="Final prequential metrics",
        ylabel="Score",
    )
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, model, metrics_payload, {}, {})


def _positive_probability(model, features: dict) -> float | None:
    if not hasattr(model, "predict_proba_one"):
        return None
    probabilities = model.predict_proba_one(features)
    if not probabilities:
        return None
    return float(probabilities.get(1, probabilities.get(True, 0.0)))
