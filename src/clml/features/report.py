import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("CLML_DATA_DIR", ".data")) / "service" / "matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from clml.constants import (
    ARTIFACT_FE_CANDIDATE_RULES_JSON,
    ARTIFACT_FE_RECOMMENDATIONS_JSON,
    CATALOG_SEED,
    PLOT_DPI,
    PLOT_FIGSIZE_MUTUAL_INFO,
    PLOT_FIGSIZE_SKEW,
    PLOT_TOP_FEATURES_COUNT,
)
from clml.data.catalog import DatasetBundle
from clml.features.recommendations import recommend_feature_engineering
from clml.methods.registry import MethodSpec


def write_feature_engineering_report(
    bundle: DatasetBundle,
    spec: MethodSpec,
    output_dir: Path,
    *,
    enabled: bool,
    rules: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    advice = recommend_feature_engineering(bundle, spec)
    payload = {
        **asdict(advice),
        "enabled": enabled,
        "active_rules": rules,
        "method": spec.name,
        "dataset": bundle.info.name,
    }
    with (output_dir / ARTIFACT_FE_RECOMMENDATIONS_JSON).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    if advice.candidate_rules:
        with (output_dir / ARTIFACT_FE_CANDIDATE_RULES_JSON).open("w", encoding="utf-8") as fh:
            json.dump({"rules": advice.candidate_rules}, fh, indent=2)
    _plot_skew(bundle.x, output_dir / "numeric_skew.png")
    _plot_mutual_information(bundle, output_dir / "mutual_information.png")


def _plot_skew(x: pd.DataFrame, path: Path) -> None:
    numeric = x.select_dtypes(include="number")
    if numeric.empty:
        return
    skew = (
        numeric.skew(numeric_only=True)
        .abs()
        .sort_values(ascending=False)
        .head(PLOT_TOP_FEATURES_COUNT)
    )
    if skew.empty:
        return
    plt.figure(figsize=PLOT_FIGSIZE_SKEW)
    sns.barplot(x=skew.values, y=skew.index, hue=skew.index, palette="rocket", legend=False)
    plt.xlabel("Absolute skew")
    plt.ylabel("Feature")
    plt.title("Numeric skew candidates for transformation")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def _plot_mutual_information(bundle: DatasetBundle, path: Path) -> None:
    y = bundle.y
    if y is None or bundle.info.task not in {"classification", "regression"}:
        return
    x = pd.get_dummies(bundle.x, dummy_na=True)
    x = x.replace([float("inf"), float("-inf")], pd.NA)
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    if x.empty:
        return
    if bundle.info.task == "classification":
        scores = mutual_info_classif(x, y, random_state=CATALOG_SEED)
    else:
        scores = mutual_info_regression(x, y, random_state=CATALOG_SEED)
    series = (
        pd.Series(scores, index=x.columns)
        .sort_values(ascending=False)
        .head(PLOT_TOP_FEATURES_COUNT)
    )
    if series.empty:
        return
    plt.figure(figsize=PLOT_FIGSIZE_MUTUAL_INFO)
    sns.barplot(x=series.values, y=series.index, hue=series.index, palette="mako", legend=False)
    plt.xlabel("Mutual information")
    plt.ylabel("Feature")
    plt.title("Feature signal candidates")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
