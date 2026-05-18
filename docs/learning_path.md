# Learning Path For Data Scientists With Strong Math And Physics Background

This path turns `clml` into a structured laboratory for classical machine learning. It assumes you
are comfortable with linear algebra, calculus, probability, optimization, and physical modeling, but
want a practical route from theory to reliable data-science work.

## How To Use This Repository While Learning

Work in short experiment cycles:

1. Inspect the dataset with `clml dataexplore --dataset <name>`.
2. Read method candidates in [methods.md](methods.md) and metrics in [metrics.md](metrics.md).
3. Run an interpretable baseline with `--trials 0`.
4. Compare a nonlinear method and a regularized method.
5. Tune only after the baseline is understood.
6. Review artifacts with `clml runs last --method <method>` and MLflow.
7. Write down the failure mode: bias, variance, leakage, wrong metric, bad target, bad validation, or
   weak features.

Prefer this command pattern:

```bash
clml dataexplore --dataset credit_risk
clml run --method logistic_regression --dataset credit_risk --trials 0
clml run --method random_forest_classifier --dataset credit_risk --trials 0
clml run --method hist_gradient_boosting_classifier --dataset credit_risk --trials 20
clml runs last --method hist_gradient_boosting_classifier
```

After a full sweep, inspect the generated learning report:

```bash
clml run-all --trials 1
```

This writes `.data/reports/run_all/learning_insights.md` and
`.data/reports/run_all/learning_insights.csv`. The Markdown report highlights practical baselines,
cautionary cases, and learning labs derived from the actual run outcomes.

## Stage 1: Statistical Foundations In Code

Goal: connect familiar math to empirical behavior.

Core ideas:

- Random variables, conditional expectation, loss minimization, likelihood, bias and variance.
- Train/test split, cross-validation, leakage, calibration, and uncertainty.
- Metrics as estimators of operational utility, not abstract scores.

Recommended experiments:

| Question | Methods | Dataset | What to inspect |
|----------|---------|---------|-----------------|
| What does a linear decision boundary miss? | `logistic_regression`, `lda_classifier`, `qda_classifier` | `credit_risk` | Accuracy vs Macro-F1 vs AUC |
| How much does regularization stabilize coefficients? | `linear_regression`, `ridge_regression`, `lasso_regression`, `elastic_net_regression` | `diabetes` | R2, coefficient artifacts |
| When does a robust loss help? | `huber_regression`, `ransac_regression`, `theil_sen_regression` | `housing_prices` | RMSE vs MAE gap |
| How does probability calibration matter? | `logistic_regression`, `calibrated_logistic_regression` | `credit_risk` | AUC and predicted probabilities |

Practice standard: every conclusion must mention the validation design and the metric weakness.

## Stage 2: Geometry, Kernels, And High-Dimensional Structure

Goal: reason about feature spaces the way you would reason about state spaces.

Core ideas:

- Scaling and distance concentration.
- Inner products, margins, kernels, manifolds, projections.
- Identifiability: a pretty projection is not proof of separability.

Recommended experiments:

| Question | Methods | Dataset | What to inspect |
|----------|---------|---------|-----------------|
| How does margin geometry classify samples? | `svc_classifier`, `linear_svc_classifier`, `nu_svc_classifier` | `breast_cancer` | Macro-F1, support-vector pressure |
| What variance is retained by linear projection? | `pca`, `truncated_svd`, `factor_analysis` | `digits` | Explained variance and embedding plots |
| What nonlinear neighborhoods appear? | `isomap`, `kernel_pca`, `tsne`, `umap` | `digits` | 2D projection artifacts |
| When do local methods fail? | `knn_classifier`, `knn_regressor` | `credit_risk`, `housing_prices` | Sensitivity to dimensionality |

Physics analogy: treat projection methods like coordinate transforms. They can reveal structure, but
they can also hide conserved quantities, constraints, and sampling artifacts.

## Stage 3: Trees, Ensembles, And Nonlinear Tabular Modeling

Goal: build strong tabular baselines without losing interpretability discipline.

Core ideas:

- Decision trees partition feature space into piecewise-constant regions.
- Bagging reduces variance; boosting reduces bias by fitting residual structure.
- Feature importance is model-dependent and can be misleading under correlation.

Recommended experiments:

| Question | Methods | Dataset | What to inspect |
|----------|---------|---------|-----------------|
| What does a single tree overfit? | `decision_tree_classifier`, `decision_tree_regressor` | `credit_risk`, `housing_prices` | Tree behavior vs ensemble metrics |
| How much variance does bagging remove? | `bagging_classifier`, `random_forest_classifier`, `extra_trees_classifier` | `credit_risk` | Macro-F1 and AUC |
| Which booster wins on this data? | `gradient_boosting_classifier`, `hist_gradient_boosting_classifier`, `xgboost_classifier`, `lightgbm_classifier`, `catboost_classifier` | `credit_risk` | `third_party_comparisons.json` |
| What does uncertainty add? | `ngboost_regressor` | `housing_prices` | Point metrics and distribution artifacts |

Practice standard: choose the simplest model that meets the metric and interpretability requirement.

## Stage 4: Feature Engineering And Data Quality

Goal: make feature work auditable and reproducible.

Core ideas:

- Put transformations inside pipelines to prevent leakage.
- Treat feature rules as versioned hypotheses.
- Diagnose missingness, skew, cardinality, imbalance, and correlated counters before modeling.

Commands:

```bash
clml feature-advice --method random_forest_classifier --dataset credit_risk
clml run --method random_forest_classifier \
  --dataset credit_risk \
  --feature-engineering \
  --feature-rules examples/feature_rules/credit_risk.json
```

Experiments:

| Question | Methods | Dataset | What to inspect |
|----------|---------|---------|-----------------|
| Does feature selection preserve signal? | `feature_select_k_best_f_classif`, `feature_rfe_logistic`, `feature_rfecv_logistic` | `breast_cancer` | Selected feature diagnostics |
| Does target encoding help categories? | `category_target_encoder_logistic`, `category_leave_one_out_encoder_logistic`, `category_hashing_encoder_sgd` | `credit_risk` | Leakage risk and AUC |
| Does class balancing help minority cases? | `imbalanced_smote_logistic`, `imbalanced_random_undersampling_logistic`, `imbalanced_balanced_random_forest` | `credit_risk` | Macro-F1 and minority fraction |

## Stage 5: Unsupervised Learning, Anomaly Detection, And Density

Goal: use unsupervised methods as instruments, not as automatic truth machines.

Core ideas:

- Clusters are model assumptions about geometry and density.
- Anomaly score direction and threshold must match the operational definition of abnormal.
- Density models are sensitive to scale, dimension, and kernel bandwidth.

Recommended experiments:

| Question | Methods | Dataset | What to inspect |
|----------|---------|---------|-----------------|
| Are groups compact or density-shaped? | `kmeans`, `dbscan`, `hdbscan`, `optics`, `spectral_clustering` | `blobs` | Silhouette, cluster plots |
| Is anomaly ranking useful? | `isolation_forest`, `one_class_svm`, `local_outlier_factor`, `elliptic_envelope` | `anomaly` | AUC vs Macro-F1 |
| Does a generative view fit? | `gaussian_mixture`, `bayesian_gaussian_mixture`, `kernel_density` | `blobs` | Log-likelihood and plots |

Practice standard: cluster labels are hypotheses. Validate them against domain constraints or
downstream usefulness before naming them.

## Stage 6: Time, Survival, Streaming, And Optimization

Goal: move beyond independent rows.

Core ideas:

- Time order changes validation: random splits can leak future information.
- Survival data separates event occurrence from observation censoring.
- Streaming models evaluate before training on each sample.
- Optimization models require objective functions and constraints, not just predictors.

Recommended experiments:

| Area | Methods | Dataset | What to inspect |
|------|---------|---------|-----------------|
| Forecasting | `statsmodels_exponential_smoothing`, `statsmodels_sarimax`, `statsmodels_autoreg` | `airline_passengers` | Holdout MAE, RMSE, MAPE |
| Survival | `lifelines_cox_model`, `lifelines_kaplan_meier` | `customer_survival` | C-index, log-rank p-value |
| Online learning | `river_logistic_regression` | `streaming_churn` | Prequential accuracy and F1 |
| Convex optimization | `cvxpy_portfolio_optimization`, `cvxpy_quadratic_programming` | `portfolio_assets` | Return, volatility, constraints |
| Mathematical programming | `scipy_linear_programming`, `scipy_nonlinear_optimization` | `production_planning`, `marketing_mix` | Objective value and utilization |

## Capstone Workflow

Use your own CSV and produce an evidence-backed model note.

```bash
clml suggest-methods --data path/to/data.csv --target-column target
clml run-data --method ridge_regression --data path/to/data.csv --target-column target --trials 0
clml run-data --method random_forest_regressor --data path/to/data.csv --target-column target --trials 0
clml run-data --method hist_gradient_boosting_regressor --data path/to/data.csv --target-column target --trials 20
```

Your final note should include:

- Problem framing: prediction, explanation, segmentation, anomaly ranking, forecast, or optimization.
- Target definition and why it is learnable from the available features.
- Validation design and leakage controls.
- Baseline result, strongest result, and why the improvement is credible.
- Error analysis by segment, time, or feature regime.
- Metric interpretation from [metrics.md](metrics.md), including what the metric does not measure.
- Operational next step: collect better labels, change decision threshold, add features, simplify model,
  or deploy a monitored baseline.

## Reading Order Inside This Repo

1. [quickstart.md](quickstart.md): commands and run artifacts.
2. [datasets.md](datasets.md): built-in experiments.
3. [methods.md](methods.md): implemented method catalog.
4. [metrics.md](metrics.md): interpretation rules and thresholds.
5. [feature_engineering.md](feature_engineering.md): auditable feature rules.
6. [method_recommendations.md](method_recommendations.md): using `suggest-methods` on new data.
7. [mlflow.md](mlflow.md): comparing runs and registered models.
