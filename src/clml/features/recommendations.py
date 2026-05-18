from dataclasses import dataclass
from typing import Any

import pandas as pd

from clml.constants import SKEW_THRESHOLD
from clml.data.catalog import DatasetBundle
from clml.methods.registry import MethodSpec


@dataclass(frozen=True)
class FeatureEngineeringAdvice:
    recommendations: list[str]
    candidate_rules: list[dict[str, Any]]


def recommend_feature_engineering(
    bundle: DatasetBundle,
    spec: MethodSpec,
) -> FeatureEngineeringAdvice:
    x = bundle.x
    numeric = x.select_dtypes(include="number")
    categorical = x.select_dtypes(exclude="number")
    datetime_columns = _datetime_columns(x)
    recommendations: list[str] = []
    rules: list[dict[str, Any]] = []

    for column in datetime_columns:
        recommendations.append(f"Extract calendar parts from `{column}` before modeling.")
        rules.append(
            {
                "type": "datetime_parts",
                "column": column,
                "parts": ["day_of_week", "month", "quarter", "is_weekend"],
            }
        )

    if "age" in numeric.columns:
        recommendations.append(
            "Create age bands when interpretability or nonlinear effects matter."
        )
        rules.append(
            {
                "type": "numeric_bins",
                "column": "age",
                "bins": [0, 25, 30, 35, 40, 50, 65, 120],
                "labels": ["<25", "25-29", "30-34", "35-39", "40-49", "50-64", "65+"],
                "output_column": "age_group",
            }
        )

    if {"loan_amount", "annual_income"}.issubset(x.columns):
        recommendations.append("Add loan-to-income ratio; it is often stronger than raw amounts.")
        rules.append(
            {
                "type": "ratio",
                "numerator": "loan_amount",
                "denominator": "annual_income",
                "output_column": "loan_to_income",
            }
        )

    for column, skew in numeric.skew(numeric_only=True).abs().items():
        if skew > SKEW_THRESHOLD:
            recommendations.append(
                f"Consider `log1p` transform for skewed numeric feature `{column}`."
            )
            rules.append({"type": "log1p", "column": column})

    if not categorical.empty:
        recommendations.append(
            "Keep categorical encoding inside the training pipeline to avoid leakage."
        )

    method_name = spec.name.lower()
    guide = spec.guide_section.lower()
    if any(token in method_name for token in ["linear", "logistic", "sgd", "svm", "svc", "svr"]):
        recommendations.append(
            "Linear/kernel methods benefit from scaling, skew reduction, and explicit interactions."
        )
    tree_tokens = ["forest", "tree", "boost", "xgboost", "lightgbm", "catboost"]
    if any(token in method_name for token in tree_tokens):
        recommendations.append(
            "Tree methods usually need less scaling; focus on ratios and date parts."
        )
    if "nearest" in guide or "knn" in method_name:
        recommendations.append(
            "Distance methods are sensitive to scale and one-hot dimensionality."
        )
    if "naive bayes" in guide:
        recommendations.append(
            "Naive Bayes often prefers nonnegative, discretized, or count-like features."
        )
    if "clustering" in spec.task or spec.task in {"dimensionality", "anomaly"}:
        recommendations.append(
            "Unsupervised methods need scaling and careful handling of sparse dummies."
        )
    if "statsmodels" in method_name:
        recommendations.append(
            "Statsmodels is best with interpretable, low-collinearity engineered features."
        )
    if "imbalanced" in method_name:
        recommendations.append(
            "Apply feature engineering before resampling, and resample only inside CV/train."
        )

    if not recommendations:
        recommendations.append(
            "Start with leakage-safe ratios, bins, date parts, and skew transforms."
        )

    return FeatureEngineeringAdvice(recommendations=recommendations, candidate_rules=rules)


def _datetime_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            columns.append(column)
        elif "date" in column.lower() or column.lower().endswith("_at"):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().mean() > 0.9:
                columns.append(column)
    return columns
