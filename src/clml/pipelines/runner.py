"""Pipeline orchestration: builds RunContext and dispatches to task runners."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from clml.config.log import get_logger
from clml.config.settings import get_settings
from clml.data.adapters import infer_data_format, read_frame
from clml.data.catalog import DatasetBundle, DatasetInfo, load_dataset
from clml.data.explore import explore_dataset
from clml.features.report import write_feature_engineering_report
from clml.features.rules import load_rules
from clml.methods.registry import MethodSpec, get_method
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines._metrics import interpret_metrics
from clml.pipelines.tasks import TASK_RUNNERS
from clml.recommendations.methods import _infer_target_column, _infer_time_column

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_method(
    method_name: str,
    dataset_name: str | None = None,
    *,
    trials: int | None = None,
    feature_engineering: bool | None = None,
    feature_rules_path: Path | None = None,
) -> RunResult:
    settings = get_settings()
    spec = get_method(method_name)
    bundle = load_dataset(dataset_name or spec.dataset)
    logger.info("starting run: method=%s dataset=%s", method_name, bundle.info.name)
    ctx = _make_context(settings, spec, bundle, trials, feature_engineering, feature_rules_path)
    return _execute_run(ctx, extra_params={})


def run_data_file_method(
    method_name: str,
    data_path: Path,
    *,
    target_column: str | None = None,
    trials: int | None = None,
    feature_engineering: bool | None = None,
    feature_rules_path: Path | None = None,
) -> RunResult:
    logger.info("starting run: method=%s data=%s", method_name, data_path)
    settings = get_settings()
    spec = get_method(method_name)
    frame = read_frame(data_path)
    if target_column is not None and target_column not in frame.columns:
        columns = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Target column `{target_column}` was not found in `{data_path}`. "
            f"Available columns: {columns}"
        )
    supervised_tasks = {
        "classification",
        "feature_selection",
        "incremental_classification",
        "regression",
        "statsmodels_classification",
        "statsmodels_regression",
        "categorical_encoding",
        "imbalanced_classification",
        "timeseries",
    }
    inferred_target = target_column
    if inferred_target is None and spec.task in supervised_tasks:
        inferred_target = _infer_target_column(frame, _infer_time_column(frame))
    if spec.task in supervised_tasks and inferred_target is None:
        raise ValueError(f"Method `{method_name}` requires --target-column for tabular data.")
    feature_columns = [col for col in frame.columns if col != inferred_target]
    bundle = DatasetBundle(
        info=DatasetInfo(
            name=data_path.stem,
            task=spec.task,
            description=f"User-provided tabular dataset from {data_path}",
            feature_columns=feature_columns,
            target_column=inferred_target,
            source=str(data_path),
            rows=len(frame),
            columns=len(frame.columns),
        ),
        frame=frame,
    )
    ctx = _make_context(settings, spec, bundle, trials, feature_engineering, feature_rules_path)
    return _execute_run(
        ctx,
        extra_params={
            "data_path": str(data_path),
            "data_format": infer_data_format(data_path),
            "target_column": inferred_target or "",
        },
    )


# ---------------------------------------------------------------------------
# Internal orchestration
# ---------------------------------------------------------------------------


def _make_context(
    settings: Any,
    spec: MethodSpec,
    bundle: DatasetBundle,
    trials: int | None,
    feature_engineering: bool | None,
    feature_rules_path: Path | None,
) -> RunContext:
    use_fe = settings.feature_engineering if feature_engineering is None else feature_engineering
    rules_path = feature_rules_path or settings.feature_rules
    feature_rules = load_rules(rules_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.data_dir / spec.name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    n_trials = settings.optuna_trials if trials is None else trials
    logger.debug(
        "run context: dir=%s trials=%d feature_engineering=%s rules=%d",
        run_dir,
        n_trials,
        use_fe,
        len(feature_rules),
    )
    return RunContext(
        spec=spec,
        bundle=bundle,
        run_dir=run_dir,
        trials=n_trials,
        feature_engineering=use_fe,
        feature_rules=feature_rules,
    )


def _execute_run(ctx: RunContext, extra_params: dict[str, str]) -> RunResult:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(ctx.spec.name)
    with mlflow.start_run(run_name=ctx.run_dir.name):
        mlflow.log_params(
            {
                "method": ctx.spec.name,
                "dataset": ctx.bundle.info.name,
                "task": ctx.spec.task,
                "guide_section": ctx.spec.guide_section,
                "feature_engineering": ctx.feature_engineering,
                "feature_rules_count": len(ctx.feature_rules),
                **extra_params,
            }
        )
        description = ctx.spec.notes or f"{ctx.spec.title} — {ctx.spec.guide_section}"
        mlflow.set_tags(
            {
                "method_title": ctx.spec.title,
                "description": description,
                "dataset_description": ctx.bundle.info.description,
            }
        )
        explore_dataset(ctx.bundle, ctx.run_dir / "exploration")
        write_feature_engineering_report(
            ctx.bundle,
            ctx.spec,
            ctx.run_dir / "feature_engineering",
            enabled=ctx.feature_engineering,
            rules=ctx.feature_rules,
        )
        result = _dispatch_run(ctx)
        numeric_metrics = {
            k: float(v) for k, v in result.metrics.items() if isinstance(v, int | float)
        }
        mlflow.log_metrics(numeric_metrics)
        interpretation = interpret_metrics(ctx.spec.task, result.metrics)
        mlflow.set_tag("metric_interpretation", interpretation)
        mlflow.log_artifacts(str(ctx.run_dir))
        logger.info("run complete: %s → %s", ctx.spec.name, interpretation)
    return result


def _dispatch_run(ctx: RunContext) -> RunResult:
    runner = TASK_RUNNERS.get(ctx.spec.task)
    if runner is None:
        raise ValueError(f"Unsupported task type: {ctx.spec.task!r}")
    logger.debug("dispatching task=%s method=%s", ctx.spec.task, ctx.spec.name)
    return runner(ctx)
