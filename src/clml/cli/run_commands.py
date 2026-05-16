from pathlib import Path
from typing import Annotated

import typer

from clml.cli.common import console
from clml.config.settings import get_settings
from clml.methods.registry import list_methods
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
    summary_dir = settings.data_dir / "run_all"
    rows = []
    for spec in list_methods():
        console.print(f"Running {spec.name}")
        try:
            result = run_method(
                spec.name,
                trials=trials,
                feature_engineering=feature_engineering,
                feature_rules_path=feature_rules,
            )
            rows.append(
                {
                    "method": spec.name,
                    "status": "ok",
                    "run_dir": str(result.run_dir),
                    "metrics": result.metrics,
                }
            )
            console.print(f"  ok: {result.run_dir}")
        except Exception as exc:
            rows.append({"method": spec.name, "status": "failed", "error": str(exc)})
            console.print(f"  failed: {exc}")
            if not continue_on_error:
                write_run_all_summary(rows, summary_dir)
                raise typer.Exit(code=1) from exc
    write_run_all_summary(rows, summary_dir)
    failures = [row for row in rows if row["status"] == "failed"]
    console.print(f"Run-all summary written to {summary_dir}")
    console.print(f"Completed {len(rows) - len(failures)} / {len(rows)} methods")
    if failures:
        raise typer.Exit(code=1)
