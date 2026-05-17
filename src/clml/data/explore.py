import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from clml.data.adapters import write_frame
from clml.data.catalog import DatasetBundle
from clml.config.log import get_logger
from clml.reporting.plots import plot_correlation_heatmap, plot_feature_distributions

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExplorationReport:
    dataset: str
    rows: int
    columns: int
    task: str
    missing_fraction: dict[str, float]
    numeric_summary: dict[str, dict[str, float]]
    target_summary: dict[str, int | float | str] | None
    suggestions: list[str]


def explore_dataset(bundle: DatasetBundle, output_dir: Path | None = None) -> ExplorationReport:
    logger.debug("exploring dataset: %s (%d rows, %d cols)", bundle.info.name, bundle.info.rows, bundle.info.columns)
    frame = bundle.frame
    numeric = frame[bundle.info.feature_columns].select_dtypes(include="number")
    missing = frame.isna().mean().sort_values(ascending=False)
    summary = numeric.describe().to_dict()
    target_summary = _target_summary(frame, bundle.info.target_column)
    suggestions = _suggestions(bundle, numeric, missing)

    report = ExplorationReport(
        dataset=bundle.info.name,
        rows=bundle.info.rows,
        columns=bundle.info.columns,
        task=bundle.info.task,
        missing_fraction={str(k): float(v) for k, v in missing.items()},
        numeric_summary=_floatify_nested(summary),
        target_summary=target_summary,
        suggestions=suggestions,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "exploration.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)
        write_frame(output_dir / "numeric_summary.csv", numeric.describe().T, include_index=True)
        plot_correlation_heatmap(numeric, output_dir / "correlation.png")
        plot_feature_distributions(
            frame, bundle.info.feature_columns[:8], output_dir / "features.png"
        )
    return report


def _target_summary(
    frame: pd.DataFrame, target_column: str | None
) -> dict[str, int | float | str] | None:
    if target_column is None:
        return None
    target = frame[target_column]
    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
        return {
            "kind": "continuous",
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        }
    counts = target.value_counts().to_dict()
    return {"kind": "categorical", **{str(k): int(v) for k, v in counts.items()}}


def _suggestions(bundle: DatasetBundle, numeric: pd.DataFrame, missing: pd.Series) -> list[str]:
    suggestions: list[str] = []
    categorical_columns = [
        col
        for col in bundle.info.feature_columns
        if col in bundle.frame.columns and not pd.api.types.is_numeric_dtype(bundle.frame[col])
    ]
    if missing.max() > 0:
        suggestions.append("Add imputation in the preprocessing pipeline before model fitting.")
    if categorical_columns:
        suggestions.append(
            "Use one-hot encoding for categorical features; keep it inside Pipeline."
        )
    if numeric.shape[1] > 25:
        suggestions.append(
            "Use feature selection, PCA, or regularized linear models to reduce variance."
        )
    if not numeric.empty and numeric.skew(numeric_only=True).abs().max() > 1.0:
        suggestions.append("Consider log/quantile transforms for skewed numeric features.")
    if _has_high_correlation(numeric):
        suggestions.append(
            "Check correlated predictors; regularization or tree models may be more stable."
        )
    if bundle.info.task == "classification":
        suggestions.append(
            "Start with LogisticRegression, RandomForestClassifier, SVC, and gradient boosting."
        )
        target = bundle.y
        if target is not None and target.value_counts(normalize=True).min() < 0.2:
            suggestions.append(
                "Use stratified validation and class_weight or resampling for imbalance."
            )
    elif bundle.info.task == "regression":
        suggestions.append(
            "Start with Ridge/Lasso, RandomForestRegressor, SVR, and gradient boosting."
        )
        suggestions.append("Inspect residuals and compare RMSE with MAE for outlier sensitivity.")
    elif bundle.info.task == "clustering":
        suggestions.append(
            "Scale features, inspect PCA projections, then compare KMeans, DBSCAN, and GMM."
        )
    elif bundle.info.task == "anomaly":
        suggestions.append(
            "Scale features and compare IsolationForest, LocalOutlierFactor, and OneClassSVM."
        )
    if not suggestions:
        suggestions.append(
            "Use scaling, cross-validation, and permutation importance as a baseline workflow."
        )
    return suggestions


def _has_high_correlation(numeric: pd.DataFrame) -> bool:
    if numeric.shape[1] < 2:
        return False
    corr = numeric.corr(numeric_only=True).abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    upper = corr.where(mask)
    return bool(upper.max().max() > 0.85)


def _floatify_nested(raw: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        str(col): {str(stat): float(value) for stat, value in stats.items()}
        for col, stats in raw.items()
    }
