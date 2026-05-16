from clml.methods.registry import list_methods
from clml.recommendations._models import MethodRecommendation, NonRecommendation


def _not_recommended_methods(
    inferred_task: str,
    recommendations: list[MethodRecommendation],
) -> list[NonRecommendation]:
    recommended = {item.method for item in recommendations}
    all_methods = {spec.name: spec for spec in list_methods()}

    def names_for(tasks: set[str] | None = None, keywords: tuple[str, ...] = ()) -> list[str]:
        names = []
        for name, spec in all_methods.items():
            if name in recommended:
                continue
            if tasks is not None and spec.task in tasks:
                names.append(name)
            elif keywords and any(keyword in name for keyword in keywords):
                names.append(name)
        return sorted(set(names))

    groups: list[NonRecommendation] = []
    if inferred_task == "time_series_cumulative_regression":
        groups.extend(
            [
                NonRecommendation(
                    "Classification methods",
                    names_for(
                        {
                            "classification",
                            "feature_selection",
                            "incremental_classification",
                            "statsmodels_classification",
                            "categorical_encoding",
                            "imbalanced_classification",
                        }
                    ),
                    "No categorical target was detected; the dataset contains cumulative "
                    "numeric counters.",
                ),
                NonRecommendation(
                    "Anomaly methods",
                    names_for({"anomaly"}),
                    "Useful only after converting cumulative counters to daily deltas or rates.",
                ),
                NonRecommendation(
                    "Density and mixture models",
                    names_for({"density"}) + names_for(keywords=("mixture",)),
                    "They model distribution shape, but do not directly answer trend or "
                    "target questions.",
                ),
                NonRecommendation(
                    "Survival and optimization methods",
                    names_for(
                        {
                            "survival_cox",
                            "survival_kaplan_meier",
                            "linear_programming",
                            "nonlinear_optimization",
                            "cvxpy_portfolio",
                            "cvxpy_quadratic",
                            "timeseries",
                        }
                    ),
                    "The data file is observational time/counter data, not event-duration or "
                    "constraint data.",
                ),
                NonRecommendation(
                    "Small-data probabilistic classifiers",
                    names_for(keywords=("gaussian_process_classifier", "naive", "qda", "lda")),
                    "These expect supervised class labels and are not aligned with "
                    "cumulative regression.",
                ),
            ]
        )
    elif inferred_task == "regression":
        groups.extend(
            [
                NonRecommendation(
                    "Classification methods",
                    names_for(
                        {
                            "classification",
                            "feature_selection",
                            "incremental_classification",
                            "statsmodels_classification",
                        }
                    ),
                    "The selected target is continuous, not categorical.",
                ),
                NonRecommendation(
                    "Optimization and survival methods",
                    names_for(
                        {
                            "linear_programming",
                            "nonlinear_optimization",
                            "cvxpy_portfolio",
                            "cvxpy_quadratic",
                            "survival_cox",
                            "survival_kaplan_meier",
                            "timeseries",
                        }
                    ),
                    "The data file lacks explicit constraints or duration/event columns.",
                ),
            ]
        )
    elif inferred_task == "timeseries":
        groups.extend(
            [
                NonRecommendation(
                    "Classification methods",
                    names_for(
                        {
                            "classification",
                            "feature_selection",
                            "incremental_classification",
                            "statsmodels_classification",
                            "categorical_encoding",
                            "imbalanced_classification",
                        }
                    ),
                    "The selected target is a continuous time-ordered series, not categorical.",
                ),
                NonRecommendation(
                    "Optimization and survival methods",
                    names_for(
                        {
                            "linear_programming",
                            "nonlinear_optimization",
                            "cvxpy_portfolio",
                            "cvxpy_quadratic",
                            "survival_cox",
                            "survival_kaplan_meier",
                        }
                    ),
                    "The data file looks like an ordered forecasting problem, not constraints or "
                    "duration/event data.",
                ),
            ]
        )
    elif inferred_task == "classification":
        groups.extend(
            [
                NonRecommendation(
                    "Regression methods",
                    names_for(
                        {"regression", "regression_1d", "statsmodels_regression", "timeseries"}
                    ),
                    "The selected target is categorical.",
                ),
                NonRecommendation(
                    "Optimization and survival methods",
                    names_for(
                        {
                            "linear_programming",
                            "nonlinear_optimization",
                            "cvxpy_portfolio",
                            "cvxpy_quadratic",
                            "survival_cox",
                            "survival_kaplan_meier",
                        }
                    ),
                    "The data file lacks explicit constraints or duration/event columns.",
                ),
            ]
        )
    else:
        groups.extend(
            [
                NonRecommendation(
                    "Supervised methods",
                    names_for(
                        {
                            "classification",
                            "feature_selection",
                            "incremental_classification",
                            "regression",
                            "timeseries",
                        }
                    ),
                    "No reliable target column was detected.",
                ),
                NonRecommendation(
                    "Task-specific extensions",
                    names_for(
                        {
                            "linear_programming",
                            "nonlinear_optimization",
                            "cvxpy_portfolio",
                            "cvxpy_quadratic",
                            "survival_cox",
                            "survival_kaplan_meier",
                        }
                    ),
                    "The data file does not match required constraint or survival schemas.",
                ),
            ]
        )
    groups = [group for group in groups if group.methods]
    covered = recommended | {method for group in groups for method in group.methods}
    missing = sorted(name for name in all_methods if name not in covered)
    if missing:
        groups.append(
            NonRecommendation(
                "Lower-priority implemented methods",
                missing,
                "These implemented methods were assessed but are not first-choice matches for "
                "the inferred dataset profile and recommendation limit.",
            )
        )
    return groups
