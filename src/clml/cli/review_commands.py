from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from clml.cli.common import console
from clml.data.adapters import write_frame
from clml.runs.review import last_run, runs_frame


def runs_list_command(
    method: Annotated[str | None, typer.Option(help="Filter by method name.")] = None,
    limit: Annotated[int, typer.Option(help="Maximum number of rows to show.")] = 20,
) -> None:
    frame = runs_frame(method).head(limit)
    if frame.empty:
        console.print("No runs found.")
        return
    table = Table(title="Run artifacts")
    for column in ["method", "dataset", "task", "timestamp", "run_dir"]:
        table.add_column(column)
    for _, row in frame.iterrows():
        table.add_row(
            str(row.get("method", "")),
            str(row.get("dataset", "")),
            str(row.get("task", "")),
            str(row.get("timestamp", "")),
            str(row.get("run_dir", "")),
        )
    console.print(table)


def runs_last_command(
    method: Annotated[str | None, typer.Option(help="Method name to inspect.")] = None,
) -> None:
    record = last_run(method)
    if record is None:
        console.print("No runs found.")
        raise typer.Exit(code=1)
    console.print(f"Run: {record.run_dir}")
    console.print(f"Method: {record.method}")
    console.print(f"Dataset: {record.dataset}")
    console.print(f"Task: {record.task}")
    console.print("Metrics:")
    console.print(record.metrics)
    if record.best_params:
        console.print("Best params:")
        console.print(record.best_params)
    if record.comparisons:
        console.print("Comparisons:")
        console.print(record.comparisons)


def runs_export_command(
    method: Annotated[str | None, typer.Option(help="Filter by method name.")] = None,
    output: Annotated[Path | None, typer.Option(help="Tabular output path.")] = None,
) -> None:
    from clml.config.settings import get_settings

    resolved = output or (get_settings().reports_dir / "runs_summary.csv")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame = runs_frame(method)
    write_frame(resolved, frame)
    console.print(f"Exported {len(frame)} rows to {resolved}")
