from dataclasses import dataclass


@dataclass(frozen=True)
class MethodRecommendation:
    method: str
    why_suggested: str
    use_case: str
    expected_result: str
    interpretation: str
    command: str
    priority: str


@dataclass(frozen=True)
class NonRecommendation:
    category: str
    methods: list[str]
    reason: str


@dataclass(frozen=True)
class DatasetMethodAdvice:
    dataset_name: str
    path: str
    rows: int
    columns: int
    inferred_task: str
    target_column: str | None
    time_column: str | None
    notes: list[str]
    recommendations: list[MethodRecommendation]
    not_recommended: list[NonRecommendation]
