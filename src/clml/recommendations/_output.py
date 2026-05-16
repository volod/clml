import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from clml.data.catalog import DatasetBundle
from clml.data.explore import explore_dataset
from clml.recommendations._models import DatasetMethodAdvice
from clml.reporting.plots import (
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_named_bars,
)


def _write_advice(
    advice: DatasetMethodAdvice,
    bundle: DatasetBundle,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    explore_dataset(bundle, output_dir / "exploration")
    with (output_dir / "method_advice.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(advice), fh, indent=2)
    _write_markdown(advice, output_dir / "method_advice.md")
    numeric = frame.select_dtypes(include="number")
    plot_correlation_heatmap(numeric, output_dir / "correlation.png")
    plot_feature_distributions(
        frame,
        frame.columns[:8].tolist(),
        output_dir / "feature_distributions.png",
    )
    if advice.recommendations:
        counts = pd.Series([rec.priority for rec in advice.recommendations]).value_counts()
        plot_named_bars(
            counts.index.tolist(),
            counts.values.astype(float).tolist(),
            output_dir / "recommendation_priorities.png",
            title="Recommended method priorities",
            ylabel="Count",
        )


def _write_markdown(advice: DatasetMethodAdvice, path: Path) -> None:
    lines = [
        f"# Method Advice For `{advice.dataset_name}`",
        "",
        f"- Path: `{advice.path}`",
        f"- Shape: {advice.rows} rows x {advice.columns} columns",
        f"- Inferred task: `{advice.inferred_task}`",
        f"- Target column: `{advice.target_column}`",
        f"- Time column: `{advice.time_column}`",
        "",
        "## Dataset Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in advice.notes)
    lines.extend(["", "## Recommended Methods", ""])
    for rec in advice.recommendations:
        lines.extend(
            [
                f"### `{rec.method}` ({rec.priority})",
                "",
                f"- Why suggested: {rec.why_suggested}",
                f"- Use case: {rec.use_case}",
                f"- Expected result: {rec.expected_result}",
                f"- Interpretation: {rec.interpretation}",
                f"- Command: `{rec.command}`",
                "",
            ]
        )
    lines.extend(["", "## Why Other Implemented Methods Were Not Suggested", ""])
    for item in advice.not_recommended:
        methods = ", ".join(f"`{method}`" for method in item.methods)
        lines.extend(
            [
                f"### {item.category}",
                "",
                f"- Reason: {item.reason}",
                f"- Methods: {methods}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
