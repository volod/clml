import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".data") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay


def plot_correlation_heatmap(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    plt.figure(figsize=(10, 8))
    corr = frame.corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, square=False)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_feature_distributions(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    if not columns:
        return
    cols = columns[: min(len(columns), 8)]
    axes = frame[cols].hist(figsize=(12, 8), bins=30)
    for axis_row in axes:
        for axis in axis_row:
            axis.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_regression_predictions(y_true, y_pred, path: Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=35)
    low = min(float(min(y_true)), float(min(y_pred)))
    high = max(float(max(y_true)), float(max(y_pred)))
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_timeseries_forecast(
    series: pd.Series,
    forecast: pd.Series,
    path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values, label="observed", linewidth=1.8)
    plt.plot(forecast.index, forecast.values, label="forecast", linewidth=2.0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_2d_projection(x, labels, path: Path, title: str) -> None:
    if getattr(x, "shape", [0, 0])[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(x)
    else:
        coords = x
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=28, legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_named_bars(
    labels: list[str],
    values: list[float],
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(x=labels, y=values, hue=labels, palette="crest", legend=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_marketing_response_curves(
    frame: pd.DataFrame, optimized_spend: np.ndarray, path: Path
) -> None:
    plt.figure(figsize=(9, 6))
    for idx, row in frame.iterrows():
        spend_grid = np.linspace(0, row["max_spend"], 80)
        response = row["baseline_sales"] + (row["saturation_sales"] - row["baseline_sales"]) * (
            1 - np.exp(-spend_grid / row["response_scale"])
        )
        plt.plot(spend_grid, response, label=row["channel"])
        plt.scatter([optimized_spend[idx]], [response_at_spend(row, optimized_spend[idx])], s=35)
    plt.xlabel("Spend")
    plt.ylabel("Expected sales")
    plt.title("Marketing response curves with optimized spend")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def response_at_spend(row: pd.Series, spend: float) -> float:
    return float(
        row["baseline_sales"]
        + (row["saturation_sales"] - row["baseline_sales"])
        * (1 - np.exp(-spend / row["response_scale"]))
    )
