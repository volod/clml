from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

EstimatorFactory = Callable[[int], Any]
ParamSpace = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    title: str
    guide_section: str
    task: str
    dataset: str
    estimator_factory: EstimatorFactory
    param_space: ParamSpace | None = None
    needs_nonnegative: bool = False
    third_party_factories: dict[str, EstimatorFactory] = field(default_factory=dict)
    notes: str = ""
