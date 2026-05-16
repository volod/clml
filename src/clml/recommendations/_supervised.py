import pandas as pd

from clml.recommendations._common import (
    _has_outliers,
    _is_imbalanced,
    _is_nonnegative,
    _is_right_skewed,
    _is_strictly_positive,
    _rec,
)
from clml.recommendations._models import MethodRecommendation


def recommend_regression(
    frame: pd.DataFrame, target_column: str | None
) -> list[MethodRecommendation]:
    target = frame[target_column] if target_column else None
    recommendations = [
        _rec(
            "linear_regression",
            "The target appears continuous, so a simple regression baseline is appropriate.",
            "Fast regression baseline.",
            "RMSE/MAE/R2.",
            "Baseline for comparison.",
            target_column,
            "high",
        ),
        _rec(
            "ridge_regression",
            "Ridge is a stable linear baseline when numeric predictors are correlated.",
            "Regularized regression baseline.",
            "RMSE/MAE/R2 with less coefficient instability than OLS.",
            "Compare against OLS to judge collinearity sensitivity.",
            target_column,
            "high",
        ),
        _rec(
            "random_forest_regressor",
            "Tree ensembles are robust nonlinear baselines for mixed tabular features.",
            "Nonlinear tabular regression.",
            "Regression metrics and model artifact.",
            "Captures interactions.",
            target_column,
            "high",
        ),
        _rec(
            "extra_trees_regressor",
            "Extra Trees is a randomized tree ensemble that often trains quickly and "
            "reduces variance.",
            "Alternative nonlinear ensemble regression.",
            "Regression metrics and model artifact.",
            "Useful when Random Forest works but randomized thresholds may generalize better.",
            target_column,
            "high",
        ),
        _rec(
            "hist_gradient_boosting_regressor",
            "Histogram gradient boosting is a strong sklearn tabular regressor for "
            "nonlinear patterns.",
            "Strong boosted-tree regression without third-party dependencies.",
            "Regression metrics and prediction plot.",
            "Compare to forests to see whether additive boosting helps.",
            target_column,
            "medium",
        ),
        _rec(
            "quantile_regression",
            "Quantile regression estimates conditional medians or tails instead of only the mean.",
            "Uncertainty- or tail-aware linear regression.",
            "Regression metrics for a selected quantile fit.",
            "Use when asymmetric error or upper/lower target behavior matters.",
            target_column,
            "medium",
        ),
        _rec(
            "bagging_regressor",
            "Bagging demonstrates variance reduction through bootstrap aggregation.",
            "Variance-reduced regression ensemble.",
            "Regression metrics and averaged predictions.",
            "Useful as a simple ensemble baseline around unstable base learners.",
            target_column,
            "medium",
        ),
    ]
    if target is not None and _is_nonnegative(target):
        recommendations.append(
            _rec(
                "poisson_regression",
                "The target is non-negative, so a count/exposure GLM may be appropriate.",
                "Poisson GLM for count-like numeric targets.",
                "Regression metrics with positive predictions.",
                "Use for counts/rates; if the target is continuous money, compare cautiously.",
                target_column,
                "medium",
            )
        )
    if target is not None and _is_strictly_positive(target) and _is_right_skewed(target):
        recommendations.append(
            _rec(
                "gamma_regression",
                "The target is strictly positive and right-skewed, which matches Gamma GLM "
                "assumptions.",
                "Positive skewed-target GLM.",
                "Regression metrics with positive predictions.",
                "Useful for costs, severities, durations, and similar positive outcomes.",
                target_column,
                "medium",
            )
        )
    if target is not None and _has_outliers(target):
        recommendations.extend(
            [
                _rec(
                    "ransac_regression",
                    "The target distribution has potential outliers; RANSAC fits a "
                    "consensus inlier model.",
                    "Robust regression under gross outliers.",
                    "Regression metrics and robust fitted model.",
                    "Large gaps versus OLS suggest outliers are influencing the mean fit.",
                    target_column,
                    "medium",
                ),
                _rec(
                    "theil_sen_regression",
                    "Theil-Sen provides another robust linear baseline using median slopes.",
                    "Robust linear regression for small to medium datasets.",
                    "Regression metrics from a median-slope fit.",
                    "Useful when a few high-leverage points should not dominate.",
                    target_column,
                    "medium",
                ),
            ]
        )
    else:
        recommendations.append(
            _rec(
                "ard_regression",
                "ARD Regression adds Bayesian sparsity through per-feature regularization.",
                "Sparse Bayesian linear regression.",
                "Regression metrics and a fitted Bayesian linear model.",
                "Features with strong shrinkage are likely less relevant under a linear model.",
                target_column,
                "medium",
            )
        )
    if len(frame) <= 10_000:
        recommendations.append(
            _rec(
                "ngboost_regressor",
                "NGBoost adds predictive uncertainty by fitting a full conditional distribution.",
                "Probabilistic regression with uncertainty estimates.",
                "Regression metrics plus distribution prediction artifacts.",
                "Use intervals and distribution parameters to inspect uncertainty, not only RMSE.",
                target_column,
                "medium",
            )
        )
    return recommendations


def recommend_classification(
    frame: pd.DataFrame, target_column: str | None, time_column: str | None
) -> list[MethodRecommendation]:
    target = frame[target_column] if target_column else None
    is_imbalanced = target is not None and _is_imbalanced(target)
    recommendations = [
        _rec(
            "logistic_regression",
            "The target appears categorical, so an interpretable classifier baseline is useful.",
            "Interpretable classification baseline.",
            "Accuracy/F1/ROC-AUC.",
            "Check class separability.",
            target_column,
            "high",
        ),
        _rec(
            "linear_svc_classifier",
            "LinearSVC is a fast max-margin baseline for larger or high-dimensional "
            "classification data.",
            "Linear margin classifier.",
            "Accuracy/F1 and confusion matrix.",
            "Compare with logistic regression to separate probability calibration from margin fit.",
            target_column,
            "high",
        ),
        _rec(
            "random_forest_classifier",
            "Tree ensembles handle nonlinear categorical/numeric interactions well.",
            "Nonlinear tabular classifier.",
            "Confusion matrix and classification metrics.",
            "Captures interactions.",
            target_column,
            "high",
        ),
        _rec(
            "extra_trees_classifier",
            "Extra Trees gives a randomized-tree ensemble comparison to Random Forest.",
            "Fast nonlinear classification ensemble.",
            "Confusion matrix and classification metrics.",
            "Useful when forest-style nonlinear interactions are likely.",
            target_column,
            "medium",
        ),
        _rec(
            "hist_gradient_boosting_classifier",
            "Histogram gradient boosting is a strong sklearn tabular classifier for "
            "nonlinear patterns.",
            "Boosted-tree classification.",
            "Accuracy/F1/ROC-AUC when available.",
            "Compare to forests to see whether boosting improves difficult cases.",
            target_column,
            "medium",
        ),
        _rec(
            "feature_select_k_best_f_classif",
            "Univariate f-test selection is a fast way to identify individually strong "
            "classification features.",
            "Filter feature selection before a downstream classifier.",
            "Selected feature list and downstream macro-F1.",
            "Use as a simple baseline; it ignores feature interactions.",
            target_column,
            "medium",
        ),
        _rec(
            "feature_select_from_l1_linear_svc",
            "An L1 linear SVM provides embedded sparse feature selection.",
            "Model-based feature selection before downstream classification.",
            "Selected feature list and downstream classification score.",
            "Selected features survived regularized model pressure.",
            target_column,
            "medium",
        ),
    ]
    if not is_imbalanced and len(frame) <= 5_000:
        recommendations.append(
            _rec(
                "nu_svc_classifier",
                "NuSVC gives the SVM margin model a different regularization framing.",
                "Kernel SVM comparison on smaller classification datasets.",
                "Classification metrics and confusion matrix.",
                "Useful when boundary shape matters and the dataset is not too large.",
                target_column,
                "medium",
            )
        )
    if is_imbalanced:
        recommendations.append(
            _rec(
                "imbalanced_smote_logistic",
                "The class distribution appears imbalanced, so SMOTE can test "
                "minority-class sensitivity.",
                "Classification when classes are imbalanced.",
                "F1/ROC-AUC with resampling.",
                "Compare minority-class behavior against non-resampled baselines.",
                target_column,
                "medium",
            )
        )
    if time_column or len(frame) >= 1_000:
        recommendations.append(
            _rec(
                "river_logistic_regression",
                "The data is ordered or large enough to make online prequential learning "
                "a useful comparison.",
                "Streaming binary classification, learning one sample at a time.",
                "Prequential accuracy/F1 and per-sample prediction history.",
                "Read metrics as online performance before each sample is learned.",
                target_column,
                "medium",
            )
        )
    return recommendations
