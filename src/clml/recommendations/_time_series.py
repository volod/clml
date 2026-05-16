import pandas as pd

from clml.recommendations._common import _largest_numeric_column, _rec
from clml.recommendations._models import MethodRecommendation


def recommend_cumulative_series(
    frame: pd.DataFrame, target_column: str | None
) -> list[MethodRecommendation]:
    target = target_column or _largest_numeric_column(frame)
    return [
        _rec(
            "linear_regression",
            "The dataset has a date column and cumulative numeric target, so a linear "
            "trend is the first baseline.",
            f"Trend baseline for cumulative `{target}` over time/day features.",
            "A simple fitted trend and prediction scatter/residual behavior.",
            "Interpret slope-like effects as average cumulative growth.",
            target,
            priority="high",
        ),
        _rec(
            "ridge_regression",
            "Cumulative equipment counters are highly correlated; Ridge stabilizes coefficients.",
            "Regularized trend model when cumulative equipment columns are collinear.",
            "Stable coefficients and regression metrics such as RMSE/MAE/R2.",
            "If Ridge is close to OLS, the trend is simple; if better, collinearity matters.",
            target,
            priority="high",
        ),
        _rec(
            "random_forest_regressor",
            "Tree ensembles can capture phase changes and nonlinear interactions between counters.",
            "Nonlinear regression for cumulative totals or daily increments.",
            "Prediction quality plus feature-importance-friendly fitted model artifact.",
            "Strong results may indicate nonlinear phases; avoid far extrapolation.",
            target,
            priority="high",
        ),
        _rec(
            "extra_trees_regressor",
            "Extremely randomized trees give a second nonlinear ensemble view with "
            "less split optimization than Random Forest.",
            "Fast nonlinear regression for cumulative counters and phase changes.",
            "Regression metrics plus a fitted ensemble artifact.",
            "Compare against Random Forest to see whether randomized split thresholds help.",
            target,
            priority="high",
        ),
        _rec(
            "hist_gradient_boosting_regressor",
            "Boosting is a strong tabular method for nonlinear numeric regression patterns.",
            "Boosted-tree regression for nonlinear time phases and interactions.",
            "Usually stronger tabular regression metrics than linear baselines.",
            "Interpret as predictive pattern capture, not causal explanation.",
            target,
            priority="high",
        ),
        _rec(
            "poisson_regression",
            "Cumulative count-like non-negative targets fit the Poisson GLM framing better "
            "than plain squared-error regression.",
            "Count/exposure-style GLM baseline for non-negative counters.",
            "Regression metrics and positive fitted predictions.",
            "Use when the target is a count or rate-like measure; compare to Tweedie.",
            target,
            priority="medium",
        ),
        _rec(
            "quantile_regression",
            "Cumulative data can have asymmetric residuals; median regression checks "
            "robust central tendency.",
            "Median or conditional-quantile trend model.",
            "Regression metrics for the selected quantile fit.",
            "Useful when upper/lower behavior matters more than the mean.",
            target,
            priority="medium",
        ),
        _rec(
            "pca",
            "Many loss counters move together; PCA summarizes shared variation into components.",
            "Reduce many equipment-loss counters to dominant directions.",
            "2D embedding and explained-variance summary.",
            "Nearby days have similar profiles; components often represent intensity.",
            None,
            priority="medium",
        ),
        _rec(
            "tsne",
            "t-SNE can reveal local neighborhoods or regime changes in many correlated counters.",
            "Visual exploration of similar dates or profiles.",
            "2D neighborhood embedding.",
            "Interpret neighborhoods qualitatively; t-SNE axes are not direct factors.",
            None,
            priority="medium",
        ),
        _rec(
            "umap",
            "UMAP is a faster modern manifold view that often preserves more global "
            "structure than t-SNE.",
            "High-dimensional profile visualization.",
            "2D embedding plot.",
            "Use UMAP for neighborhood and broad structure exploration, not causal claims.",
            None,
            priority="medium",
        ),
        _rec(
            "kmeans",
            "Rows can be segmented into periods with similar equipment-loss profiles.",
            "Cluster days into regimes by equipment-loss profile.",
            "Cluster labels, silhouette score, and PCA-style cluster plot.",
            "Interpret clusters as phases; validate against dates/events externally.",
            None,
            priority="medium",
        ),
        _rec(
            "kernel_pca",
            "If phases are curved or nonlinear, Kernel PCA can reveal structure missed by PCA.",
            "Nonlinear 2D projection for cumulative equipment-loss trajectories.",
            "Curved embedding that can reveal nonlinear phases.",
            "Use visual neighborhoods as exploratory structure, not labels.",
            None,
            priority="medium",
        ),
        _rec(
            "statsmodels_ols",
            "Statsmodels adds coefficient tables and p-values for interpretable trend analysis.",
            "Inference-oriented trend model for interpretable coefficients.",
            "Coefficient table, p-values, summary text, and regression metrics.",
            "Use coefficients for associations; autocorrelation still matters.",
            target,
            priority="medium",
        ),
    ]


def recommend_timeseries(
    frame: pd.DataFrame, target_column: str | None
) -> list[MethodRecommendation]:
    target = target_column or _largest_numeric_column(frame)
    return [
        _rec(
            "statsmodels_exponential_smoothing",
            "A date-like column and continuous numeric target suggest forecasting; "
            "Holt-Winters is the first seasonal baseline.",
            "Trend and seasonal forecasting for one ordered target series.",
            "Holdout MAE/RMSE/MAPE and forecast plot.",
            "Use as a compact seasonal baseline before adding ARIMA complexity.",
            target,
            priority="high",
        ),
        _rec(
            "statsmodels_sarimax",
            "SARIMAX models autocorrelation, differencing, and seasonal structure in one "
            "state-space model.",
            "Seasonal ARIMA-style forecasting.",
            "Holdout forecast metrics plus forecast artifact.",
            "Useful when residual autocorrelation remains after simpler seasonal baselines.",
            target,
            priority="high",
        ),
        _rec(
            "statsmodels_autoreg",
            "AutoReg tests how much predictive signal is present in lagged target values.",
            "Autoregressive forecasting baseline.",
            "Holdout MAE/RMSE/MAPE and forecast plot.",
            "Strong AutoReg results indicate target lags carry most of the signal.",
            target,
            priority="medium",
        ),
        _rec(
            "pca",
            "Projection can still help inspect feature structure around the time series.",
            "Exploratory view of row-level feature structure.",
            "2D embedding plot.",
            "Use only for exploratory structure, not forecasting.",
            None,
            priority="medium",
        ),
    ]
