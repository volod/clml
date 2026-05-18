import json
from collections.abc import Callable

from sklearn import datasets

from clml.config.log import get_logger
from clml.config.settings import get_settings
from clml.constants import ARTIFACT_DATASET_DATA_CSV, ARTIFACT_DATASET_METADATA_JSON
from clml.data.adapters import read_frame, write_frame
from clml.data.sklearn_datasets import _blobs, _digits, _moons, _sklearn_bunch_frame
from clml.data.specialized_datasets import (
    _airline_passengers,
    _anomaly,
    _customer_survival,
    _marketing_mix,
    _portfolio_assets,
    _production_planning,
    _streaming_churn,
)
from clml.data.supervised_datasets import _credit_risk, _housing_prices, _monotone_1d
from clml.data.types import DatasetBundle, DatasetInfo

logger = get_logger(__name__)


def available_dataset_names() -> list[str]:
    return sorted(_BUILDERS)


def load_dataset(name: str, *, prepare: bool = True) -> DatasetBundle:
    if name not in _BUILDERS:
        names = ", ".join(available_dataset_names())
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {names}")
    if prepare:
        return prepare_dataset(name)
    logger.debug("building dataset without cache: %s", name)
    return _BUILDERS[name]()


def prepare_dataset(name: str) -> DatasetBundle:
    settings = get_settings()
    dataset_dir = settings.datasets_dir / name
    data_path = dataset_dir / ARTIFACT_DATASET_DATA_CSV
    metadata_path = dataset_dir / ARTIFACT_DATASET_METADATA_JSON
    if data_path.exists() and metadata_path.exists():
        frame = read_frame(data_path)
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = DatasetBundle(
            info=DatasetInfo(**raw),
            frame=frame,
        )
        if cached.info.rows == len(frame):
            logger.debug("loaded dataset from cache: %s (%d rows)", name, cached.info.rows)
            return cached
    logger.info("building and caching dataset: %s", name)
    bundle = _BUILDERS[name]()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_frame(data_path, bundle.frame)
    metadata_path.write_text(json.dumps(bundle.info.__dict__, indent=2), encoding="utf-8")
    return bundle


def prepare_all_datasets() -> list[DatasetInfo]:
    return [prepare_dataset(name).info for name in available_dataset_names()]


_BUILDERS: dict[str, Callable[[], DatasetBundle]] = {
    "airline_passengers": _airline_passengers,
    "anomaly": _anomaly,
    "blobs": _blobs,
    "breast_cancer": lambda: _sklearn_bunch_frame(
        "breast_cancer",
        "classification",
        "Breast cancer diagnostic measurements for binary classification.",
        datasets.load_breast_cancer,
        "sklearn.datasets.load_breast_cancer",
    ),
    "credit_risk": _credit_risk,
    "customer_survival": _customer_survival,
    "diabetes": lambda: _sklearn_bunch_frame(
        "diabetes",
        "regression",
        "Diabetes disease progression benchmark for regression.",
        datasets.load_diabetes,
        "sklearn.datasets.load_diabetes",
    ),
    "digits": _digits,
    "housing_prices": _housing_prices,
    "iris": lambda: _sklearn_bunch_frame(
        "iris",
        "classification",
        "Iris flower measurements for multiclass classification.",
        datasets.load_iris,
        "sklearn.datasets.load_iris",
    ),
    "marketing_mix": _marketing_mix,
    "monotone_1d": _monotone_1d,
    "moons": _moons,
    "portfolio_assets": _portfolio_assets,
    "production_planning": _production_planning,
    "streaming_churn": _streaming_churn,
    "wine": lambda: _sklearn_bunch_frame(
        "wine",
        "classification",
        "Wine chemistry measurements for multiclass classification.",
        datasets.load_wine,
        "sklearn.datasets.load_wine",
    ),
}
