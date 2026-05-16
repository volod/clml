from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from clml.cli.common import console
from clml.config.settings import get_settings
from clml.data.catalog import available_dataset_names, load_dataset, prepare_all_datasets
from clml.data.explore import explore_dataset
from clml.methods.registry import USER_GUIDE_COVERAGE, list_methods


def list_methods_command() -> None:
    table = Table(title="Available method pipelines")
    table.add_column("name")
    table.add_column("task")
    table.add_column("dataset")
    table.add_column("guide section")
    for spec in list_methods():
        table.add_row(spec.name, spec.task, spec.dataset, spec.guide_section)
    console.print(table)


def coverage_command() -> None:
    for group, entries in USER_GUIDE_COVERAGE.items():
        console.print(f"[bold]{group}[/bold]")
        for entry in entries:
            console.print(f"  - {entry}")


def datasets_command() -> None:
    for name in available_dataset_names():
        bundle = load_dataset(name)
        console.print(
            f"{name}: {bundle.info.rows} rows, {bundle.info.columns} columns, "
            f"task={bundle.info.task}, source={bundle.info.source}"
        )


def prepare_data_command() -> None:
    infos = prepare_all_datasets()
    for info in infos:
        console.print(f"cached {info.name}: {info.rows} rows, {info.columns} columns")


def dataexplore_command(
    dataset: Annotated[
        str,
        typer.Option(help="Dataset name from `clml datasets`."),
    ] = "credit_risk",
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional output directory."),
    ] = None,
) -> None:
    settings = get_settings()
    bundle = load_dataset(dataset)
    if output_dir is None:
        output_dir = settings.data_dir / "dataexplore" / dataset
    report = explore_dataset(bundle, output_dir)
    console.print(f"Exploration written to {output_dir}")
    for suggestion in report.suggestions:
        console.print(f"- {suggestion}")
