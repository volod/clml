from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from clml.cli.common import console
from clml.config.settings import get_settings
from clml.methods.registry import list_methods
from clml.pipelines._metrics import interpret_metrics
from clml.pipelines.runner import run_data_file_method, run_method
from clml.runs.review import write_run_all_summary


def run_command(
    method: Annotated[
        str,
        typer.Option(help="Method name from `clml list-methods`."),
    ] = "random_forest_classifier",
    dataset: Annotated[
        str | None,
        typer.Option(help="Override the method default dataset."),
    ] = None,
    trials: Annotated[
        int | None,
        typer.Option(help="Override CLML_OPTUNA_TRIALS."),
    ] = None,
    feature_engineering: Annotated[
        bool | None,
        typer.Option(help="Enable rule-based feature engineering for this run."),
    ] = None,
    feature_rules: Annotated[
        Path | None,
        typer.Option(help="JSON file with feature engineering rules."),
    ] = None,
) -> None:
    result = run_method(
        method,
        dataset,
        trials=trials,
        feature_engineering=feature_engineering,
        feature_rules_path=feature_rules,
    )
    console.print(f"Run written to {result.run_dir}")
    console.print(result.metrics)
    if result.comparisons:
        console.print("Third-party comparisons:")
        console.print(result.comparisons)


def run_data_command(
    method: Annotated[
        str,
        typer.Option(help="Method name from `clml list-methods`."),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", help="Tabular data file path."),
    ],
    target_column: Annotated[
        str | None,
        typer.Option(help="Target column for supervised data-file runs."),
    ] = None,
    trials: Annotated[
        int | None,
        typer.Option(help="Override CLML_OPTUNA_TRIALS."),
    ] = None,
    feature_engineering: Annotated[
        bool | None,
        typer.Option(help="Enable rule-based feature engineering for this run."),
    ] = None,
    feature_rules: Annotated[
        Path | None,
        typer.Option(help="JSON file with feature engineering rules."),
    ] = None,
) -> None:
    result = run_data_file_method(
        method,
        data,
        target_column=target_column,
        trials=trials,
        feature_engineering=feature_engineering,
        feature_rules_path=feature_rules,
    )
    console.print(f"Run written to {result.run_dir}")
    console.print(result.metrics)


def run_all_command(
    trials: Annotated[
        int | None,
        typer.Option(help="Override CLML_OPTUNA_TRIALS for every tunable method."),
    ] = 0,
    continue_on_error: Annotated[
        bool,
        typer.Option(help="Continue running remaining methods if one method fails."),
    ] = True,
    feature_engineering: Annotated[
        bool | None,
        typer.Option(help="Enable rule-based feature engineering for every compatible method."),
    ] = None,
    feature_rules: Annotated[
        Path | None,
        typer.Option(help="JSON file with feature engineering rules."),
    ] = None,
) -> None:
    settings = get_settings()
    summary_dir = settings.reports_dir / "run_all"
    all_specs = list(list_methods())
    total = len(all_specs)
    rows = []
    try:
        for idx, spec in enumerate(all_specs, start=1):
            console.print(f"Running {spec.name} [dim]{idx}/{total}[/dim]")
            try:
                result = run_method(
                    spec.name,
                    trials=trials,
                    feature_engineering=feature_engineering,
                    feature_rules_path=feature_rules,
                )
                interpretation = interpret_metrics(result.task, result.metrics)
                rows.append(
                    {
                        "method": spec.name,
                        "task": spec.task,
                        "dataset": result.dataset,
                        "guide_section": spec.guide_section,
                        "status": "ok",
                        "run_dir": str(result.run_dir),
                        "metrics": result.metrics,
                        "summary": interpretation,
                    }
                )
                console.print(f"  [green]ok[/green]  {interpretation}")
            except Exception as exc:
                rows.append(
                    {
                        "method": spec.name,
                        "task": spec.task,
                        "dataset": spec.dataset,
                        "guide_section": spec.guide_section,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                console.print(f"  failed: {exc}")
                if not continue_on_error:
                    write_run_all_summary(rows, summary_dir)
                    raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\nInterrupted — writing partial summary.")
        write_run_all_summary(rows, summary_dir)
        failures = [row for row in rows if row["status"] == "failed"]
        _print_run_all_table(rows)
        console.print(f"\nRun-all summary written to {summary_dir}")
        completed = len(rows) - len(failures)
        console.print(
            f"Completed [bold]{completed} / {len(rows)}[/bold] methods before interrupt"
        )
        raise typer.Exit(code=130) from None
    write_run_all_summary(rows, summary_dir)
    failures = [row for row in rows if row["status"] == "failed"]
    _print_run_all_table(rows)
    console.print(f"\nRun-all summary written to {summary_dir}")
    console.print(f"Completed [bold]{len(rows) - len(failures)} / {len(rows)}[/bold] methods")
    if failures:
        raise typer.Exit(code=1)


def _print_run_all_table(rows: list[dict]) -> None:
    table = Table(title="Run-All Results", show_lines=False, highlight=True)
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", width=7)
    table.add_column("Result")
    for row in rows:
        if row["status"] == "ok":
            status = "[green]ok[/green]"
            result = row.get("summary", "")
        else:
            status = "[red]fail[/red]"
            result = f"[red]{row.get('error', '')}[/red]"
        table.add_row(row["method"], status, result)
    console.print(table)
