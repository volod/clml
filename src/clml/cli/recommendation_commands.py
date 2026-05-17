from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from clml.cli.common import console
from clml.config.settings import get_settings
from clml.data.catalog import load_dataset
from clml.features.recommendations import recommend_feature_engineering
from clml.features.report import write_feature_engineering_report
from clml.methods.registry import get_method
from clml.recommendations.methods import advise_methods_for_file


def feature_advice_command(
    method: Annotated[
        str,
        typer.Option(help="Method name from `clml list-methods`."),
    ] = "random_forest_classifier",
    dataset: Annotated[
        str | None,
        typer.Option(help="Override the method default dataset."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional output directory for plots and rule suggestions."),
    ] = None,
) -> None:
    settings = get_settings()
    spec = get_method(method)
    bundle = load_dataset(dataset or spec.dataset)
    if output_dir is None:
        output_dir = settings.data_dir / "feature_advice" / method
    advice = recommend_feature_engineering(bundle, spec)
    write_feature_engineering_report(
        bundle,
        spec,
        output_dir,
        enabled=False,
        rules=[],
    )
    console.print(f"Feature engineering advice written to {output_dir}")
    for recommendation in advice.recommendations:
        console.print(f"- {recommendation}")


def suggest_methods_command(
    data: Annotated[
        Path,
        typer.Option("--data", help="Tabular data file path to analyze."),
    ],
    target_column: Annotated[
        str | None,
        typer.Option(help="Optional target column. If omitted, a target is inferred."),
    ] = None,
    time_column: Annotated[
        str | None,
        typer.Option(help="Optional time/date column. If omitted, one is inferred."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for advice JSON, Markdown, and plots."),
    ] = None,
    max_recommendations: Annotated[
        int,
        typer.Option(help="Maximum number of method recommendations."),
    ] = 12,
) -> None:
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.data_dir / "method_advice" / data.stem
    advice = advise_methods_for_file(
        data,
        target_column=target_column,
        time_column=time_column,
        output_dir=output_dir,
        max_recommendations=max_recommendations,
    )
    console.print(f"Method advice written to {output_dir}")
    console.print(f"Inferred task: {advice.inferred_task}")
    console.print(f"Target column: {advice.target_column}")
    console.print(f"Time column: {advice.time_column}")
    table = Table(title="Recommended implemented methods")
    table.add_column("priority")
    table.add_column("method")
    table.add_column("why suggested")
    table.add_column("use case")
    table.add_column("interpretation")
    for rec in advice.recommendations:
        table.add_row(
            rec.priority,
            rec.method,
            rec.why_suggested,
            rec.use_case,
            rec.interpretation,
        )
    console.print(table)
    if advice.not_recommended:
        console.print("[bold]Why other implemented methods were not suggested[/bold]")
        for item in advice.not_recommended:
            console.print(f"- {item.category}: {item.reason}")
