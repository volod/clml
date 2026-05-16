"""Plain-English interpretation of run metrics, keyed by task type."""

from collections.abc import Callable

from clml.constants import (
    AUC_EXCELLENT,
    AUC_GOOD,
    CONCORDANCE_GOOD,
    CONCORDANCE_MODERATE,
    LOGRANK_P_THRESHOLD,
    R2_MODERATE,
    R2_STRONG,
    SILHOUETTE_OVERLAPPING,
    SILHOUETTE_WELL_SEPARATED,
)

# ---------------------------------------------------------------------------
# Per-task interpreter functions
# ---------------------------------------------------------------------------


def _classification(m: dict) -> str:
    acc = float(m.get("accuracy", 0))
    f1 = float(m.get("f1_macro", 0))
    parts = [f"Accuracy {acc:.1%}", f"Macro-F1 {f1:.1%}"]
    if "roc_auc" in m:
        auc = float(m["roc_auc"])
        quality = "excellent" if auc >= AUC_EXCELLENT else "good" if auc >= AUC_GOOD else "poor"
        parts.append(f"AUC {auc:.3f} ({quality})")
    return "; ".join(parts)


def _incremental_classification(m: dict) -> str:
    acc = float(m.get("accuracy", 0))
    f1 = float(m.get("f1", 0))
    samples = int(float(m.get("samples", 0)))
    return f"Prequential accuracy {acc:.1%}; F1 {f1:.1%}; samples {samples}"


def _regression(m: dict) -> str:
    r2 = float(m.get("r2", 0))
    rmse = float(m.get("rmse", 0))
    mae = float(m.get("mae", 0))
    quality = "strong" if r2 >= R2_STRONG else "moderate" if r2 >= R2_MODERATE else "weak"
    return f"R² {r2:.3f} ({quality} fit); RMSE {rmse:.3f}; MAE {mae:.3f}"


def _feature_selection(m: dict) -> str:
    n = int(float(m.get("n_features_selected", 0)))
    score = float(m.get("downstream_score", 0))
    return f"Selected {n} features; downstream macro-F1 {score:.1%}"


def _timeseries(m: dict) -> str:
    horizon = int(float(m.get("horizon", 0)))
    mae = float(m.get("mae", 0))
    rmse = float(m.get("rmse", 0))
    mape = float(m.get("mape", 0))
    return f"{horizon}-step forecast; MAE {mae:.3f}; RMSE {rmse:.3f}; MAPE {mape:.1%}"


def _clustering(m: dict) -> str:
    n = int(float(m.get("clusters", 0)))
    parts = [f"{n} cluster{'s' if n != 1 else ''} found"]
    if "silhouette" in m:
        sil = float(m["silhouette"])
        quality = (
            "well-separated"
            if sil > SILHOUETTE_WELL_SEPARATED
            else "overlapping"
            if sil > SILHOUETTE_OVERLAPPING
            else "poor"
        )
        parts.append(f"silhouette {sil:.3f} ({quality})")
    if "adjusted_rand" in m:
        parts.append(f"ARI {float(m['adjusted_rand']):.3f}")
    return "; ".join(parts)


def _dimensionality(m: dict) -> str:
    n = int(float(m.get("components", 0)))
    if "explained_variance_ratio_sum" in m:
        evr = float(m["explained_variance_ratio_sum"])
        return f"{n} components explain {evr:.1%} of variance"
    return f"Projected to {n} components"


def _anomaly(m: dict) -> str:
    auc = float(m.get("roc_auc", 0))
    f1 = float(m.get("f1_macro", 0))
    quality = "excellent" if auc >= AUC_EXCELLENT else "good" if auc >= AUC_GOOD else "poor"
    return f"AUC {auc:.3f} ({quality}); Macro-F1 {f1:.1%}"


def _density(m: dict) -> str:
    mean_ll = float(m.get("mean_log_likelihood", 0))
    std_ll = float(m.get("std_log_likelihood", 0))
    return f"Mean log-likelihood {mean_ll:.3f} ± {std_ll:.3f}"


def _linear_programming(m: dict) -> str:
    profit = float(m.get("max_expected_profit", 0))
    util = float(m.get("average_resource_utilization", 0))
    return f"Optimal profit {profit:,.0f}; avg resource utilization {util:.1%}"


def _nonlinear_optimization(m: dict) -> str:
    sales = float(m.get("optimized_expected_sales", 0))
    baseline = float(m.get("current_expected_sales", 0))
    lift = (sales - baseline) / max(baseline, 1e-9)
    return f"Optimized sales {sales:,.0f} ({lift:+.1%} lift vs current budget)"


def _cvxpy(m: dict) -> str:
    ret = float(m.get("expected_return", 0))
    vol = float(m.get("volatility", 0))
    sharpe = float(m.get("sharpe_like_ratio", 0))
    return f"Expected return {ret:.1%}; volatility {vol:.1%}; Sharpe-like {sharpe:.2f}"


def _survival_cox(m: dict) -> str:
    ci_test = float(m.get("concordance_index_test", 0))
    ci_train = float(m.get("concordance_index_train", 0))
    quality = (
        "good"
        if ci_test >= CONCORDANCE_GOOD
        else "moderate"
        if ci_test >= CONCORDANCE_MODERATE
        else "weak"
    )
    return f"Cox C-index {ci_test:.3f} test ({quality}), {ci_train:.3f} train"


def _survival_kaplan_meier(m: dict) -> str:
    p = float(m.get("logrank_p_value", 1.0))
    event_frac = float(m.get("event_fraction", 0))
    sig = "significant" if p < LOGRANK_P_THRESHOLD else "not significant"
    return f"Log-rank p={p:.4f} ({sig}); event fraction {event_frac:.1%}"


def _generic(m: dict) -> str:
    numeric = {k: v for k, v in m.items() if isinstance(v, float | int)}
    if numeric:
        return "; ".join(f"{k} {float(v):.3g}" for k, v in list(numeric.items())[:4])
    return "See run artifacts for full metrics."


# ---------------------------------------------------------------------------
# Dispatch table  (task name → interpreter)
# ---------------------------------------------------------------------------

_INTERPRETERS: dict[str, Callable[[dict], str]] = {
    "classification": _classification,
    "incremental_classification": _incremental_classification,
    "imbalanced_classification": _classification,
    "statsmodels_classification": _classification,
    "categorical_encoding": _classification,
    "regression": _regression,
    "statsmodels_regression": _regression,
    "regression_1d": _regression,
    "feature_selection": _feature_selection,
    "timeseries": _timeseries,
    "clustering": _clustering,
    "dimensionality": _dimensionality,
    "anomaly": _anomaly,
    "density": _density,
    "linear_programming": _linear_programming,
    "nonlinear_optimization": _nonlinear_optimization,
    "cvxpy_portfolio": _cvxpy,
    "cvxpy_quadratic": _cvxpy,
    "survival_cox": _survival_cox,
    "survival_kaplan_meier": _survival_kaplan_meier,
}


def interpret_metrics(task: str, metrics: dict[str, float | int | str]) -> str:
    """Return a plain-English one-liner summarising the most important metrics."""
    return _INTERPRETERS.get(task, _generic)(metrics)
