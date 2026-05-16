from clml.methods._coverage import USER_GUIDE_COVERAGE
from clml.methods._types import MethodSpec
from clml.methods.specs.classification_ensembles import METHODS as CLASSIFICATION_ENSEMBLES_METHODS
from clml.methods.specs.classification_linear import METHODS as CLASSIFICATION_LINEAR_METHODS
from clml.methods.specs.classification_other import METHODS as CLASSIFICATION_OTHER_METHODS
from clml.methods.specs.clustering_anomaly import METHODS as CLUSTERING_ANOMALY_METHODS
from clml.methods.specs.dimensionality import METHODS as DIMENSIONALITY_METHODS
from clml.methods.specs.extensions import METHODS as EXTENSIONS_METHODS
from clml.methods.specs.regression_ensembles import METHODS as REGRESSION_ENSEMBLES_METHODS
from clml.methods.specs.regression_linear import METHODS as REGRESSION_LINEAR_METHODS
from clml.methods.specs.regression_other import METHODS as REGRESSION_OTHER_METHODS

__all__ = ["METHODS", "USER_GUIDE_COVERAGE", "MethodSpec", "by_task", "get_method", "list_methods"]

METHOD_GROUPS = (
    CLASSIFICATION_LINEAR_METHODS,
    CLASSIFICATION_ENSEMBLES_METHODS,
    CLASSIFICATION_OTHER_METHODS,
    REGRESSION_LINEAR_METHODS,
    REGRESSION_ENSEMBLES_METHODS,
    REGRESSION_OTHER_METHODS,
    DIMENSIONALITY_METHODS,
    CLUSTERING_ANOMALY_METHODS,
    EXTENSIONS_METHODS,
)

METHODS: dict[str, MethodSpec] = {}
for group in METHOD_GROUPS:
    METHODS.update(group)


def get_method(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        names = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method '{name}'. Available methods: {names}") from exc


def list_methods() -> list[MethodSpec]:
    return [METHODS[name] for name in sorted(METHODS)]


def by_task(task: str) -> list[MethodSpec]:
    return [spec for spec in list_methods() if spec.task == task]
