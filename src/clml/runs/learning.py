"""Learning-oriented analysis of run-all results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from clml.constants import (
    AUC_GOOD,
    R2_MODERATE,
    R2_STRONG,
    SILHOUETTE_OVERLAPPING,
    SILHOUETTE_WELL_SEPARATED,
)

MetricValue = float | int | str


def learning_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert run-all rows into practical learning signals."""
    insights = []
    for row in rows:
        if row.get("status") != "ok":
            insights.append(_failed_row(row))
            continue
        metrics = _metrics(row)
        task = str(row.get("task") or _infer_task(metrics, str(row.get("run_dir", ""))))
        primary_metric, primary_score = _primary_score(task, metrics)
        flags = _learning_flags(task, metrics)
        insights.append(
            {
                "method": row.get("method", ""),
                "task": task,
                "dataset": row.get("dataset", ""),
                "guide_section": row.get("guide_section", ""),
                "primary_metric": primary_metric,
                "primary_score": primary_score,
                "signal": _signal_label(task, primary_score),
                "learning_focus": _learning_focus(task, metrics, flags),
                "caution_flags": "; ".join(flags),
                "summary": row.get("summary", ""),
                "run_dir": row.get("run_dir", ""),
            }
        )
    return insights


def learning_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(learning_rows(rows))


def learning_markdown(rows: list[dict[str, Any]]) -> str:
    insights = learning_rows(rows)
    ok = [row for row in insights if row["primary_metric"] != "error"]
    by_task = _best_by_task(ok)
    caution = [row for row in ok if row["caution_flags"]]
    caution = sorted(caution, key=lambda row: (row["signal"], row["method"]))[:20]

    lines = [
        "# Run-All Learning Insights",
        "",
        "This report turns a complete `clml run-all` result into practical study guidance.",
        "Use it to choose high-signal experiments, not to declare universal model rankings.",
        "",
        "## Best Practical Baselines By Task",
        "",
        "| Task | Method | Primary metric | Score | Lesson |",
        "|------|--------|----------------|-------|--------|",
    ]
    for task, task_rows in sorted(by_task.items()):
        for row in task_rows[:3]:
            lines.append(
                "| "
                f"{task} | `{row['method']}` | {row['primary_metric']} | "
                f"{_format_score(row['primary_score'])} | {row['learning_focus']} |"
            )

    lines.extend(
        [
            "",
            "## Cautionary Learning Cases",
            "",
            "| Method | Signal | Caution | What To Learn |",
            "|--------|--------|---------|---------------|",
        ]
    )
    for row in caution:
        lines.append(
            "| "
            f"`{row['method']}` | {row['signal']} | {row['caution_flags']} | "
            f"{row['learning_focus']} |"
        )

    lines.extend(
        [
            "",
            "## Suggested Learning Labs",
            "",
            "- Classification: compare logistic, SGD, calibrated logistic, stacking, and "
            "boosted trees; "
            "watch Macro-F1 versus AUC to separate ranking quality from threshold quality.",
            "- Regression: compare linear, robust, bagged-tree, boosted-tree, statsmodels, and "
            "probabilistic regressors; explain RMSE/MAE gaps before tuning.",
            "- Clustering: compare K-Means, DBSCAN/HDBSCAN/OPTICS, Gaussian mixtures, and "
            "hierarchical methods; check when silhouette and ARI disagree.",
            "- Feature selection: compare selected feature count with downstream Macro-F1; prefer "
            "compact selectors only when the score loss is acceptable.",
            "- Time-aware methods: use forecasting, survival, and streaming runs to learn why "
            "random train/test splits are not enough for temporal problems.",
            "",
        ]
    )
    return "\n".join(lines)


def _metrics(row: dict[str, Any]) -> dict[str, MetricValue]:
    raw = row.get("metrics", {})
    return raw if isinstance(raw, dict) else {}


def _failed_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": row.get("method", ""),
        "task": row.get("task", ""),
        "dataset": row.get("dataset", ""),
        "guide_section": row.get("guide_section", ""),
        "primary_metric": "error",
        "primary_score": "",
        "signal": "failed",
        "learning_focus": "Failure analysis: inspect compatibility, data assumptions, and logs.",
        "caution_flags": row.get("error", ""),
        "summary": row.get("summary", ""),
        "run_dir": row.get("run_dir", ""),
    }


def _infer_task(metrics: dict[str, MetricValue], run_dir: str) -> str:
    if "n_features_selected" in metrics:
        return "feature_selection"
    if "samples" in metrics and "f1" in metrics:
        return "incremental_classification"
    if "concordance_index_test" in metrics:
        return "survival_cox"
    if "logrank_p_value" in metrics:
        return "survival_kaplan_meier"
    if "horizon" in metrics and "mape" in metrics:
        return "timeseries"
    if "roc_auc" in metrics and "accuracy" not in metrics:
        return "anomaly"
    if "sharpe_like_ratio" in metrics:
        return "cvxpy_portfolio"
    if "max_expected_profit" in metrics:
        return "linear_programming"
    if "optimized_expected_sales" in metrics:
        return "nonlinear_optimization"
    if "mean_log_likelihood" in metrics:
        return "density"
    if "f1_macro" in metrics and "roc_auc" in metrics:
        return "classification"
    if {"rmse", "mae", "r2"} <= metrics.keys():
        return "regression"
    if "silhouette" in metrics or "clusters" in metrics:
        return "clustering"
    if "components" in metrics:
        return "dimensionality"
    parts = run_dir.split("/")
    return parts[2] if len(parts) > 2 and parts[0] == ".data" else "other"


def _primary_score(task: str, metrics: dict[str, MetricValue]) -> tuple[str, float | str]:
    if "feature_selection" in task:
        return "downstream_score", _float(metrics.get("downstream_score"))
    if "incremental_classification" in task:
        return "f1", _float(metrics.get("f1"))
    if "classification" in task:
        return "f1_macro", _float(metrics.get("f1_macro", metrics.get("f1")))
    if "regression" in task:
        return "r2", _float(metrics.get("r2"))
    if "clustering" in task:
        return "silhouette", _float(metrics.get("silhouette"))
    if "dimensionality" in task:
        return "explained_variance_ratio_sum", _float(metrics.get("explained_variance_ratio_sum"))
    if "anomaly" in task:
        return "roc_auc", _float(metrics.get("roc_auc"))
    if "density" in task:
        return "mean_log_likelihood", _float(metrics.get("mean_log_likelihood"))
    if "timeseries" in task:
        return "mape", _float(metrics.get("mape"))
    if "survival_cox" in task:
        return "concordance_index_test", _float(metrics.get("concordance_index_test"))
    if "survival_kaplan_meier" in task:
        return "logrank_p_value", _float(metrics.get("logrank_p_value"))
    if "cvxpy" in task:
        return "sharpe_like_ratio", _float(metrics.get("sharpe_like_ratio"))
    if "optimization" in task or "programming" in task:
        return "objective", _first_float(metrics)
    return "score", _first_float(metrics)


def _learning_flags(task: str, metrics: dict[str, MetricValue]) -> list[str]:
    flags: list[str] = []
    if "classification" in task:
        auc = _float(metrics.get("roc_auc"))
        f1 = _float(metrics.get("f1_macro", metrics.get("f1")))
        acc = _float(metrics.get("accuracy"))
        if isinstance(auc, float) and isinstance(f1, float) and auc - f1 > 0.15:
            flags.append("AUC is much higher than Macro-F1; study thresholds and imbalance")
        if isinstance(acc, float) and isinstance(f1, float) and acc - f1 > 0.12:
            flags.append("Accuracy hides weaker per-class performance")
    if "regression" in task:
        r2 = _float(metrics.get("r2"))
        rmse = _float(metrics.get("rmse"))
        mae = _float(metrics.get("mae"))
        if isinstance(r2, float) and r2 < R2_MODERATE:
            flags.append("Weak R2; compare assumptions, features, and target scale")
        if isinstance(rmse, float) and isinstance(mae, float) and mae > 0 and rmse / mae > 1.6:
            flags.append("RMSE is much larger than MAE; inspect outliers")
    if "clustering" in task:
        silhouette = _float(metrics.get("silhouette"))
        ari = _float(metrics.get("adjusted_rand"))
        if isinstance(silhouette, float) and silhouette <= SILHOUETTE_OVERLAPPING:
            flags.append("Low silhouette; geometry does not support compact clusters")
        if isinstance(silhouette, float) and isinstance(ari, float) and ari - silhouette > 0.35:
            flags.append("ARI and silhouette disagree; labels and geometry tell different stories")
    if "dimensionality" in task:
        evr = _float(metrics.get("explained_variance_ratio_sum"))
        if isinstance(evr, float) and evr < 0.35:
            flags.append("Low 2D variance retention; projection is mainly visualization")
    if "incremental_classification" in task:
        acc = _float(metrics.get("accuracy"))
        f1 = _float(metrics.get("f1"))
        if isinstance(acc, float) and isinstance(f1, float) and acc - f1 > 0.25:
            flags.append("Prequential accuracy hides poor positive-class F1")
    return flags


def _learning_focus(task: str, metrics: dict[str, MetricValue], flags: list[str]) -> str:
    if flags:
        return flags[0]
    if "classification" in task:
        return "Study ranking, threshold choice, calibration, and class balance."
    if "regression" in task:
        r2 = _float(metrics.get("r2"))
        if isinstance(r2, float) and r2 >= R2_STRONG:
            return "Strong tabular baseline; inspect residuals and feature effects."
        return "Use as an assumptions check before stronger nonlinear models."
    if "clustering" in task:
        return "Compare cluster geometry against any available labels."
    if "dimensionality" in task:
        return "Use projection artifacts for structure discovery, not final proof."
    if "feature_selection" in task:
        return "Relate compact feature subsets to downstream predictive loss."
    if "anomaly" in task:
        return "Study score ranking separately from the operational alert threshold."
    if "density" in task:
        return "Use log-likelihood as a relative fit signal, not an absolute quality score."
    if "timeseries" in task:
        return "Practice holdout forecasting and trend/seasonality diagnostics."
    if "survival" in task:
        return "Connect censoring, event rates, and ranking quality before prediction claims."
    if "optimization" in task or "programming" in task or "cvxpy" in task:
        return "Inspect objective value together with constraints and slack."
    return "Inspect artifacts and connect metrics to the method assumptions."


def _signal_label(task: str, score: float | str) -> str:
    if not isinstance(score, float):
        return "inspect"
    if "regression" in task:
        return "strong" if score >= R2_STRONG else "moderate" if score >= R2_MODERATE else "weak"
    if "clustering" in task:
        return (
            "strong"
            if score > SILHOUETTE_WELL_SEPARATED
            else "moderate"
            if score > SILHOUETTE_OVERLAPPING
            else "weak"
        )
    if "anomaly" in task:
        return "strong" if score >= AUC_GOOD else "weak"
    if "timeseries" in task:
        return "strong" if score <= 0.05 else "moderate" if score <= 0.15 else "weak"
    return "strong" if score >= 0.8 else "moderate" if score >= 0.6 else "weak"


def _best_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    for task, task_rows in grouped.items():
        reverse = task not in {"timeseries", "survival_kaplan_meier"}
        task_rows.sort(
            key=lambda row: row["primary_score"] if isinstance(row["primary_score"], float) else -1,
            reverse=reverse,
        )
    return grouped


def _float(value: Any) -> float | str:
    if isinstance(value, int | float):
        return float(value)
    return ""


def _first_float(metrics: dict[str, MetricValue]) -> float | str:
    for value in metrics.values():
        score = _float(value)
        if isinstance(score, float):
            return score
    return ""


def _format_score(score: float | str) -> str:
    return f"{score:.3f}" if isinstance(score, float) else ""
