import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("CLML_DATA_DIR", ".data")) / "service" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

from clml.constants import (
    CATALOG_SEED,
    PLOT_BARS_LABEL_ROTATION,
    PLOT_DISTRIBUTION_BINS,
    PLOT_DISTRIBUTION_MAX_COLUMNS,
    PLOT_DPI,
    PLOT_FIGSIZE_BARS,
    PLOT_FIGSIZE_DISTRIBUTIONS,
    PLOT_FIGSIZE_HEATMAP,
    PLOT_FIGSIZE_PROJECTION,
    PLOT_FIGSIZE_RESPONSE_CURVES,
    PLOT_FIGSIZE_SCATTER,
    PLOT_FIGSIZE_TIMESERIES,
    PLOT_PROJECTION_MARKER_SIZE,
    PLOT_PROJECTION_PALETTE,
    PLOT_RESPONSE_GRID_POINTS,
    PLOT_RESPONSE_SCATTER_SIZE,
    PLOT_SCATTER_LINE_WIDTH,
    PLOT_SCATTER_MARKER_SIZE,
    PLOT_TIMESERIES_FORECAST_LINE_WIDTH,
    PLOT_TIMESERIES_OBSERVED_LINE_WIDTH,
)


def plot_correlation_heatmap(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    plt.figure(figsize=PLOT_FIGSIZE_HEATMAP)
    corr = frame.corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, square=False)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_feature_distributions(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    if not columns:
        return
    cols = columns[: min(len(columns), PLOT_DISTRIBUTION_MAX_COLUMNS)]
    axes = frame[cols].hist(figsize=PLOT_FIGSIZE_DISTRIBUTIONS, bins=PLOT_DISTRIBUTION_BINS)
    for axis_row in axes:
        for axis in axis_row:
            axis.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_regression_predictions(y_true, y_pred, path: Path) -> None:
    plt.figure(figsize=PLOT_FIGSIZE_SCATTER)
    sns.scatterplot(x=y_true, y=y_pred, s=PLOT_SCATTER_MARKER_SIZE)
    low = min(float(min(y_true)), float(min(y_pred)))
    high = max(float(max(y_true)), float(max(y_pred)))
    plt.plot([low, high], [low, high], color="black", linewidth=PLOT_SCATTER_LINE_WIDTH)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_timeseries_forecast(
    series: pd.Series,
    forecast: pd.Series,
    path: Path,
    title: str,
) -> None:
    plt.figure(figsize=PLOT_FIGSIZE_TIMESERIES)
    plt.plot(
        series.index,
        series.values,
        label="observed",
        linewidth=PLOT_TIMESERIES_OBSERVED_LINE_WIDTH,
    )
    plt.plot(
        forecast.index,
        forecast.values,
        label="forecast",
        linewidth=PLOT_TIMESERIES_FORECAST_LINE_WIDTH,
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_2d_projection(x, labels, path: Path, title: str) -> None:
    if getattr(x, "shape", [0, 0])[1] > 2:
        coords = PCA(n_components=2, random_state=CATALOG_SEED).fit_transform(x)
    else:
        coords = x
    plt.figure(figsize=PLOT_FIGSIZE_PROJECTION)
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=labels,
        palette=PLOT_PROJECTION_PALETTE,
        s=PLOT_PROJECTION_MARKER_SIZE,
        legend=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_named_bars(
    labels: list[str],
    values: list[float],
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=PLOT_FIGSIZE_BARS)
    sns.barplot(x=labels, y=values, hue=labels, palette="crest", legend=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=PLOT_BARS_LABEL_ROTATION, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_marketing_response_curves(
    frame: pd.DataFrame, optimized_spend: np.ndarray, path: Path
) -> None:
    plt.figure(figsize=PLOT_FIGSIZE_RESPONSE_CURVES)
    for idx, row in frame.iterrows():
        spend_grid = np.linspace(0, row["max_spend"], PLOT_RESPONSE_GRID_POINTS)
        response = row["baseline_sales"] + (row["saturation_sales"] - row["baseline_sales"]) * (
            1 - np.exp(-spend_grid / row["response_scale"])
        )
        plt.plot(spend_grid, response, label=row["channel"])
        plt.scatter(
            [optimized_spend[idx]],
            [response_at_spend(row, optimized_spend[idx])],
            s=PLOT_RESPONSE_SCATTER_SIZE,
        )
    plt.xlabel("Spend")
    plt.ylabel("Expected sales")
    plt.title("Marketing response curves with optimized spend")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def response_at_spend(row: pd.Series, spend: float) -> float:
    return float(
        row["baseline_sales"]
        + (row["saturation_sales"] - row["baseline_sales"])
        * (1 - np.exp(-spend / row["response_scale"]))
    )
