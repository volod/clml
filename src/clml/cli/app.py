import typer

from clml.cli.catalog_commands import (
    coverage_command,
    dataexplore_command,
    datasets_command,
    list_methods_command,
    prepare_data_command,
)
from clml.cli.recommendation_commands import feature_advice_command, suggest_methods_command
from clml.cli.review_commands import runs_export_command, runs_last_command, runs_list_command
from clml.cli.run_commands import run_all_command, run_command, run_data_command

app = typer.Typer(help="Classical ML playground.")
data_app = typer.Typer(help="Dataset cache commands.")
runs_app = typer.Typer(help="Run artifact review commands.")
app.add_typer(data_app, name="data")
app.add_typer(runs_app, name="runs")

app.command("list-methods")(list_methods_command)
app.command("coverage")(coverage_command)
app.command("datasets")(datasets_command)
app.command("dataexplore")(dataexplore_command)
app.command("feature-advice")(feature_advice_command)
app.command("suggest-methods")(suggest_methods_command)
app.command("run")(run_command)
app.command("run-data")(run_data_command)
app.command("run-all")(run_all_command)
data_app.command("prepare")(prepare_data_command)
runs_app.command("list")(runs_list_command)
runs_app.command("last")(runs_last_command)
runs_app.command("export")(runs_export_command)
