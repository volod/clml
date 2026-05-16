from pathlib import Path

from clml.data.adapters import read_frame
from clml.data.catalog import DatasetBundle, DatasetInfo
from clml.recommendations._exclusions import _not_recommended_methods
from clml.recommendations._inference import (
    _bundle_task,
    _infer_target_column,
    _infer_task,
    _infer_time_column,
    _profile_notes,
)
from clml.recommendations._models import (
    DatasetMethodAdvice,
)
from clml.recommendations._output import _write_advice
from clml.recommendations._rules import _recommend_methods


def advise_methods_for_file(
    data_path: Path,
    *,
    target_column: str | None = None,
    time_column: str | None = None,
    output_dir: Path | None = None,
    max_recommendations: int = 12,
) -> DatasetMethodAdvice:
    frame = read_frame(data_path)
    if target_column is not None and target_column not in frame.columns:
        columns = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Target column `{target_column}` was not found in `{data_path}`. "
            f"Available columns: {columns}"
        )
    dataset_name = data_path.stem
    time_column = time_column or _infer_time_column(frame)
    target_column = target_column or _infer_target_column(frame, time_column)
    inferred_task = _infer_task(frame, target_column, time_column)
    feature_columns = [col for col in frame.columns if col != target_column]
    bundle = DatasetBundle(
        info=DatasetInfo(
            name=dataset_name,
            task=_bundle_task(inferred_task),
            description=f"User-provided tabular dataset from {data_path}",
            feature_columns=feature_columns,
            target_column=target_column,
            source=str(data_path),
            rows=len(frame),
            columns=len(frame.columns),
        ),
        frame=frame,
    )
    notes = _profile_notes(frame, target_column, time_column)
    recommendations = _recommend_methods(
        frame,
        target_column=target_column,
        time_column=time_column,
        inferred_task=inferred_task,
    )[:max_recommendations]
    not_recommended = _not_recommended_methods(inferred_task, recommendations)
    advice = DatasetMethodAdvice(
        dataset_name=dataset_name,
        path=str(data_path),
        rows=len(frame),
        columns=len(frame.columns),
        inferred_task=inferred_task,
        target_column=target_column,
        time_column=time_column,
        notes=notes,
        recommendations=recommendations,
        not_recommended=not_recommended,
    )
    if output_dir is not None:
        _write_advice(advice, bundle, frame, output_dir)
    return advice
