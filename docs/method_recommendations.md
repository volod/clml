# Using `suggest-methods`

`suggest-methods` analyzes an arbitrary tabular data file and recommends implemented `clml` methods that
fit the dataset shape, target type, time columns, and practical interpretation goal. It does not
train a model by itself; it creates an advice report and commands you can run next.

## Basic Usage

```bash
clml suggest-methods --data path/to/dataset.csv
```

Use explicit columns when inference is not what you want:

```bash
clml suggest-methods \
  --data path/to/dataset.csv \
  --target-column target \
  --time-column date
```

Choose where reports are written:

```bash
clml suggest-methods \
  --data path/to/dataset.csv \
  --output-dir .data/method_advice/my_dataset
```

Limit recommendation count:

```bash
clml suggest-methods --data path/to/dataset.csv --max-recommendations 6
```

## What The Command Infers

The command loads the data file through the data adapter and profiles:

- Row and column count.
- Numeric versus categorical/text column count.
- Missing-value concentration.
- Candidate time/date column.
- Candidate target column.
- Target type: classification, regression, unsupervised, or cumulative time-series regression.
- High numeric correlation, which often appears in cumulative counters or redundant telemetry.

Inference rules are intentionally simple and auditable:

- Columns named `target`, `label`, `y`, `class`, or `event` are treated as explicit targets.
- Date-like columns are detected from names such as `date`, `*_at`, `day`, `month`, or `year`.
- Continuous numeric targets imply regression.
- Low-cardinality targets imply classification.
- Date-like data with many monotonic numeric counters is treated as cumulative time-series regression.
- If no target is reliable, unsupervised exploration is recommended.

## Output Files

The output directory contains:

- `method_advice.json`: complete machine-readable advice, including suggested and not-suggested methods.
- `method_advice.md`: human-readable explanation.
- `exploration/exploration.json`: standard dataset profile.
- `exploration/numeric_summary.csv`: numeric descriptive statistics.
- `exploration/correlation.png`: numeric correlation heatmap.
- `exploration/features.png`: basic feature distributions.
- `correlation.png`: top-level correlation heatmap.
- `feature_distributions.png`: top-level distribution plot.
- `recommendation_priorities.png`: count of high/medium recommendations.

## Understanding Suggested Methods

Each recommendation includes:

- `why_suggested`: why this method fits the observed dataset profile.
- `use_case`: what question this method helps answer.
- `expected_result`: what artifacts or metrics the run will produce.
- `interpretation`: how to read the output safely.
- `command`: a runnable `clml run-data` command template.
- `priority`: high or medium.

The rule set now prefers core scikit-learn methods before third-party extensions when there is a
good built-in fit. For example:

- Continuous targets can surface Ridge, Random Forest, Extra Trees, Histogram Gradient Boosting,
  Quantile Regression, Bagging, Poisson/Gamma GLMs, RANSAC, Theil-Sen, or ARD depending on target
  shape and outliers.
- Medium-size continuous targets can also surface `ngboost_regressor` when uncertainty-aware
  probabilistic predictions are useful.
- Date-like continuous targets can surface `statsmodels_exponential_smoothing`,
  `statsmodels_sarimax`, and `statsmodels_autoreg` for holdout time-series forecasting.
- Categorical targets can surface Logistic Regression, LinearSVC, Random Forest, Extra Trees,
  Histogram Gradient Boosting, NuSVC for balanced smaller datasets, and imbalance-specific methods
  when class skew is detected.
- Classification targets can surface sklearn feature-selection methods such as SelectKBest,
  RFE/RFECV, and SelectFromModel when feature parsimony is useful.
- Ordered or larger classification data can surface `river_logistic_regression` for online
  prequential evaluation.
- Unsupervised data can surface PCA, t-SNE, UMAP, Locally Linear Embedding, MDS, K-Means, and
  anomaly detection for structure-first exploration.

For supervised recommendations, `run-data` needs a target:

```bash
clml run-data \
  --method ridge_regression \
  --data path/to/dataset.csv \
  --target-column target \
  --trials 0
```

For unsupervised recommendations, target is optional and usually omitted:

```bash
clml run-data --method pca --data path/to/dataset.csv --trials 0
clml run-data --method kmeans --data path/to/dataset.csv --trials 0
```

## Why Some Methods Are Not Suggested

The report has a separate section: `Why Other Implemented Methods Were Not Suggested`.
Together, `Recommended Methods` and this section cover every implemented method, so the advice
output is an exhaustive assessment rather than a partial shortlist.

This section groups implemented methods by reason, for example:

- Classification methods are skipped when no categorical target exists.
- Regression methods are skipped when the selected target is categorical.
- Survival methods are skipped when no duration/event schema exists.
- Optimization methods are skipped when no objective, constraints, or allocation schema exists.
- Anomaly methods may be deferred when cumulative counters should first be transformed to daily deltas.
- Density or mixture models may be skipped when the main question is target prediction or trend analysis.

Not suggested does not mean useless. It means the method is not the first fit for the current
dataset profile. You can still run any compatible method with `clml run-data`.

## War Equipment Losses Example

In this workspace the available file is:

```bash
.data/datasets/war/russia_losses_equipment.csv
```

The originally requested path, `.data/datasets/war/russians_losses_equipment.csv`, is not present in
this workspace.

Run advice:

```bash
clml suggest-methods \
  --data .data/datasets/war/russia_losses_equipment.csv \
  --output-dir .data/method_advice/war_equipment
```

Observed profile:

- Shape: 1536 rows x 20 columns.
- Time column: `date`.
- Inferred default target: `drone`.
- Inferred task: `time_series_cumulative_regression`.
- Many numeric columns are cumulative counters and highly correlated.
- Some newer equipment columns have substantial missingness.

Suggested methods and why:

- `linear_regression`: baseline cumulative trend model.
- `ridge_regression`: handles collinearity between cumulative counters better than plain OLS.
- `random_forest_regressor`: captures nonlinear phases and interactions between equipment categories.
- `extra_trees_regressor`: provides a randomized-tree ensemble comparison for nonlinear phases.
- `hist_gradient_boosting_regressor`: strong boosted-tree baseline for nonlinear tabular regression.
- `poisson_regression`: GLM framing for non-negative count-like cumulative targets.
- `quantile_regression`: median/quantile trend model for asymmetric residual behavior.
- `pca`: summarizes shared movement among many cumulative counters into dominant components.
- `tsne`: visualizes local neighborhoods or regime changes in correlated counters.
- `umap`: faster modern manifold visualization for local and broader structure.
- `kmeans`: segments dates into similar loss-profile regimes.
- `kernel_pca`: visualizes nonlinear phase structure.
- `statsmodels_ols`: provides coefficient table, p-values, and inference-oriented trend summary.

Example run:

```bash
clml run-data \
  --method ridge_regression \
  --data .data/datasets/war/russia_losses_equipment.csv \
  --target-column drone \
  --trials 0
```

Unsupervised exploration:

```bash
clml run-data \
  --method pca \
  --data .data/datasets/war/russia_losses_equipment.csv \
  --trials 0
```

Interpretation caveat: cumulative war-loss counters are strongly autocorrelated and strongly tied
to time. Treat model outputs as exploratory pattern analysis, not causal inference or reliable
forecasting without time-series validation, differencing/rate features, and domain review.
