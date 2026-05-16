# Extended Method Candidates

These are practical classical-ML and optimization extensions beyond core scikit-learn.
Core scikit-learn methods, including the zero-new-dependency Tier 1 additions, are documented in
[methods.md](methods.md).

| Area | Library | Methods | Status | Why add it |
| --- | --- | --- | --- | --- |
| Gradient boosting | XGBoost | `xgboost_classifier`, `xgboost_regressor` | Implemented | Strong tabular baseline with regularized boosted trees. |
| Gradient boosting | LightGBM | `lightgbm_classifier`, `lightgbm_regressor` | Implemented | Efficient histogram boosting, often strong on medium/large tabular data. |
| Gradient boosting | CatBoost | `catboost_classifier`, `catboost_regressor` | Implemented | Strong tabular booster, especially useful when categorical handling is important. |
| Linear programming | SciPy | `scipy_linear_programming` | Implemented | Operations-research baseline for constrained product-mix optimization. |
| Nonlinear optimization | SciPy | `scipy_nonlinear_optimization` | Implemented | Constrained continuous optimization with nonlinear response curves. |
| Statistical modeling | statsmodels | `statsmodels_ols`, `statsmodels_glm_regression`, `statsmodels_logit`, `statsmodels_robust_regression` | Implemented | Adds coefficient inference, confidence intervals, p-values, and model diagnostics. |
| Time series forecasting | statsmodels | `statsmodels_exponential_smoothing`, `statsmodels_sarimax`, `statsmodels_autoreg` | Implemented | ETS, seasonal ARIMA, and autoregressive baselines for ordered seasonal series. |
| Imbalanced learning | imbalanced-learn | `imbalanced_smote_logistic`, `imbalanced_balanced_random_forest`, `imbalanced_random_undersampling_logistic` | Implemented | Useful for realistic imbalanced classification workflows. |
| Categorical encoding | category-encoders | `category_target_encoder_logistic`, `category_leave_one_out_encoder_logistic`, `category_hashing_encoder_sgd` | Implemented | Adds practical encoders beyond one-hot for high-cardinality categorical data. |
| General convex optimization | CVXPY | `cvxpy_quadratic_programming`, `cvxpy_portfolio_optimization` | Implemented | Clear modeling layer for convex objectives and constraints. |
| Survival analysis | lifelines | `lifelines_cox_model`, `lifelines_kaplan_meier` | Implemented | Classical time-to-event modeling not covered by sklearn. |
| Rule and interpretable models | interpret-core | `ebm_classifier`, `ebm_regressor` | Implemented | Strong interpretable generalized additive model for tabular data. |
| Manifold learning | umap-learn | `umap` | Implemented | Modern high-dimensional visualization with faster runtime and stronger global-structure preservation than t-SNE. |
| Probabilistic boosting | NGBoost | `ngboost_regressor` | Implemented | Gradient boosting that predicts full conditional distributions for uncertainty-aware regression. |
| Online learning | River | `river_logistic_regression` | Implemented | Streaming classifier with prequential evaluation and one-sample-at-a-time updates. |

## Implemented Extended Pipelines

- `xgboost_classifier` and `xgboost_regressor` use the standard supervised pipeline: dataset load, exploration, preprocessing, optional Optuna tuning, fit, metrics, predictions, plots, and run metadata.
- `lightgbm_classifier` and `lightgbm_regressor` follow the same supervised path.
- `catboost_classifier` and `catboost_regressor` follow the same supervised path with fixed compact defaults.
- `scipy_linear_programming` loads `production_planning`, solves a constrained product-mix LP with `scipy.optimize.linprog`, and writes allocation, resource utilization, profit metrics, and bar plots.
- `scipy_nonlinear_optimization` loads `marketing_mix`, solves bounded budget allocation with `scipy.optimize.minimize`, and writes spend allocation, expected sales lift, response curves, and bar plots.
- `statsmodels_*` methods write coefficient tables, summaries, metrics, predictions, and coefficient plots.
- `statsmodels_exponential_smoothing`, `statsmodels_sarimax`, and `statsmodels_autoreg` run on `airline_passengers`, hold out the final horizon, and write forecast metrics plus forecast plots.
- `imbalanced_*` methods use imbalanced-learn samplers or balanced ensembles inside the training pipeline.
- `category_*` methods compare target, leave-one-out, and hashing encoders for categorical tabular features.
- `cvxpy_*` methods solve convex allocation problems and write portfolio weights, covariance, and risk/return analytics.
- `lifelines_*` methods write Cox coefficients, concordance metrics, Kaplan-Meier curves, and survival probabilities.
- `ebm_*` methods use interpret-core Explainable Boosting Machines through the standard supervised pipeline.
- `umap` uses umap-learn's sklearn-compatible transformer inside the dimensionality pipeline and writes a 2D embedding.
- `ngboost_regressor` follows the standard regression path and writes `distribution_predictions.csv` with distribution parameters/quantiles when available.
- `river_logistic_regression` uses a dedicated incremental runner on `streaming_churn`, writing prequential predictions and online metrics.

## Useful Commands

```bash
uv run clml run --method xgboost_classifier --trials 0
uv run clml run --method lightgbm_regressor --trials 0
uv run clml run --method catboost_classifier --trials 0
uv run clml run --method scipy_linear_programming
uv run clml run --method scipy_nonlinear_optimization
uv run clml run --method statsmodels_logit
uv run clml run --method statsmodels_exponential_smoothing
uv run clml run --method statsmodels_sarimax
uv run clml run --method statsmodels_autoreg
uv run clml run --method imbalanced_smote_logistic
uv run clml run --method category_target_encoder_logistic
uv run clml run --method cvxpy_portfolio_optimization
uv run clml run --method lifelines_cox_model
uv run clml run --method ebm_classifier --trials 0
uv run clml run --method umap --trials 0
uv run clml run --method ngboost_regressor --trials 0
uv run clml run --method river_logistic_regression
```
