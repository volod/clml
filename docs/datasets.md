# Datasets

Datasets are prepared by `make data` and cached under `.data/datasets`. The catalog favors
reasonable local examples that show practical tradeoffs without requiring large downloads.

| Dataset | Task | Size | Description | Learning value |
| --- | --- | ---: | --- | --- |
| `airline_passengers` | time series | 144 x 4 | Synthetic monthly airline-passenger series with linear trend, annual seasonality, and noise. | Useful for ETS, SARIMAX, AutoReg, holdout forecasting metrics, and seasonal forecast plots. |
| `anomaly` | anomaly detection | 745 x 6 | Generated tabular data with compact normal population and shifted outliers. | Useful for Isolation Forest, LOF, One-Class SVM, contamination handling, and ROC/F1 tradeoffs. |
| `blobs` | clustering | 900 x 7 | Generated Gaussian clusters with labels kept for external validation only. | Useful for comparing centroid, hierarchy, density, mixture, and spectral clustering assumptions. |
| `breast_cancer` | classification | 569 x 31 | Numeric clinical binary classification dataset from sklearn. | Useful for fast classifiers that benefit from clean numeric features and limited runtime. |
| `credit_risk` | classification | 3500 x 13 | Synthetic-realistic credit-risk data with mixed feature types, application dates, class imbalance, correlated financial predictors, and nonlinear risk. | Main classification learning dataset; useful for preprocessing, leakage control, feature engineering, calibration, imbalance, and tree-vs-linear comparisons. |
| `customer_survival` | survival | 1400 x 8 | Synthetic customer churn time-to-event data with censoring, contracts, support load, spend, and observed churn events. | Useful for Cox models, Kaplan-Meier curves, censoring, hazard interpretation, and group survival comparisons. |
| `diabetes` | regression | 442 x 11 | Small numeric medical regression dataset from sklearn. | Useful for fast regression demos and expensive estimators such as Gaussian processes or kernel methods. |
| `digits` | classification / dimensionality | 1797 x 65 | 8x8 handwritten digit images flattened into pixel features. | Useful for Naive Bayes count-like models, PCA/NMF/SVD/ICA, manifold projections, and multiclass behavior. |
| `housing_prices` | regression | 3200 x 11 | Synthetic-realistic housing data with listing dates, skewed prices, categorical location effects, nonlinear luxury uplift, and outliers. | Main regression learning dataset; useful for robust error analysis, feature engineering, nonlinear models, and residual plots. |
| `iris` | classification | 150 x 5 | Classic small multiclass flower classification dataset from sklearn. | Useful where very fast multiclass behavior matters more than realism. |
| `moons` | classification | 900 x 3 | Generated nonlinear two-class dataset. | Useful for kernels, semi-supervised learning, and methods where decision-boundary shape is the main lesson. |
| `marketing_mix` | nonlinear optimization | 6 x 6 | Marketing channel table with current spend, spend bounds, and saturating response curves. | Useful for constrained nonlinear budget allocation and marginal-return analysis. |
| `production_planning` | linear programming | 5 x 6 | Product table with profit, demand bounds, and per-unit resource consumption. | Useful for linear programming, binding constraints, slack, and resource-utilization analysis. |
| `portfolio_assets` | convex optimization | 6 x 6 | Asset allocation table with return, volatility, liquidity, sector, and maximum weight constraints. | Useful for quadratic programming, mean-variance portfolio optimization, and risk/return tradeoffs. |
| `streaming_churn` | incremental classification | 1200 x 8 | Ordered synthetic churn stream with mild concept drift and numeric behavioral features. | Useful for River online learning, prequential evaluation, and one-sample-at-a-time updates. |
| `wine` | classification | 178 x 14 | Numeric multiclass wine cultivar dataset from sklearn. | Useful for compact multiclass experiments with feature scaling. |

## Notes

- Generated datasets are deterministic and local, so `make data` works without network access.
- Targets are stored in the `target` column.
- Mixed-type datasets exercise the project preprocessing path: median imputation, scaling, and one-hot encoding inside sklearn `Pipeline`.
- Some methods intentionally use smaller datasets because their computational cost would obscure the learning goal on larger data.
