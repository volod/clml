from collections.abc import Callable

import pandas as pd
from sklearn import datasets

from clml import constants
from clml.data.types import DatasetBundle, DatasetInfo


def _sklearn_bunch_frame(
    name: str,
    task: str,
    description: str,
    loader: Callable[..., object],
    source: str,
) -> DatasetBundle:
    bunch = loader(as_frame=True)
    frame = bunch.frame.copy()
    target_column = "target"
    if target_column not in frame.columns:
        target_column = bunch.target.name if hasattr(bunch.target, "name") else "target"
        frame[target_column] = bunch.target
    feature_columns = [str(col) for col in bunch.data.columns]
    frame.columns = [str(col) for col in frame.columns]
    info = DatasetInfo(
        name=name,
        task=task,
        description=description,
        feature_columns=feature_columns,
        target_column=target_column,
        source=source,
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _digits() -> DatasetBundle:
    bunch = datasets.load_digits()
    feature_columns = [f"pixel_{idx}" for idx in range(bunch.data.shape[1])]
    frame = pd.DataFrame(bunch.data, columns=feature_columns)
    frame["target"] = bunch.target
    info = DatasetInfo(
        name="digits",
        task="classification",
        description="8x8 handwritten digit images flattened to numeric features.",
        feature_columns=feature_columns,
        target_column="target",
        source="sklearn.datasets.load_digits",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _moons() -> DatasetBundle:
    x, y = datasets.make_moons(
        n_samples=constants.MOONS_N_SAMPLES,
        noise=constants.MOONS_NOISE,
        random_state=constants.CATALOG_SEED,
    )
    frame = pd.DataFrame(x, columns=["x0", "x1"])
    frame["target"] = y
    info = DatasetInfo(
        name="moons",
        task="classification",
        description="Generated non-linear two-class dataset for kernels and manifolds.",
        feature_columns=["x0", "x1"],
        target_column="target",
        source="sklearn.datasets.make_moons",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _blobs() -> DatasetBundle:
    x, y = datasets.make_blobs(
        n_samples=constants.BLOBS_N_SAMPLES,
        n_features=constants.BLOBS_N_FEATURES,
        centers=constants.BLOBS_CENTERS,
        cluster_std=constants.BLOBS_CLUSTER_STD,
        random_state=constants.CATALOG_SEED,
    )
    feature_columns = [f"feature_{idx}" for idx in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=feature_columns)
    frame["target"] = y
    info = DatasetInfo(
        name="blobs",
        task="clustering",
        description="Generated Gaussian blobs with labels kept only for external validation.",
        feature_columns=feature_columns,
        target_column="target",
        source="sklearn.datasets.make_blobs",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)
