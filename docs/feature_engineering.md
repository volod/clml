# Feature Engineering

Feature engineering is available as an optional, leakage-safe pipeline step. It runs inside the
model pipeline before preprocessing, so generated columns are included in cross-validation and final
fit consistently.

## Commands

Get method-specific recommendations and visual diagnostics:

```bash
clml feature-advice --method random_forest_classifier --dataset credit_risk
clml feature-advice --method linear_regression --dataset housing_prices
```

Run a method with explicit feature rules:

```bash
clml run \
  --method random_forest_classifier \
  --trials 0 \
  --feature-engineering \
  --feature-rules examples/feature_rules/credit_risk.json
```

Use feature engineering through `.env`:

```bash
CLML_FEATURE_ENGINEERING=true
CLML_FEATURE_RULES=examples/feature_rules/credit_risk.json
```

## Rule Types

Rules are JSON objects under a top-level `rules` list.

| Rule | Purpose | Example |
| --- | --- | --- |
| `datetime_parts` | Convert dates to day-of-week, month, quarter, year, or weekend flags. | `application_date` to `application_date_day_of_week` |
| `numeric_bins` | Convert continuous values to interpretable groups. | `age` to `age_group` |
| `ratio` | Create domain ratios. | `loan_amount / annual_income` |
| `product` | Create interaction products. | `income * utilization` |
| `log1p` | Reduce skew in nonnegative numeric features. | `log1p_annual_income` |
| `clip` | Cap outliers before modeling. | clip utilization to `[0, 1]` |
| `text_length` | Convert text fields to length features. | comment to `comment_length` |

Example rule files:

- [credit_risk.json](../examples/feature_rules/credit_risk.json)
- [housing_prices.json](../examples/feature_rules/housing_prices.json)

## Method-Specific Guidance

- Linear, logistic, SGD, SVM, KNN, and neural methods benefit from scaling, skew reduction, bins, and explicit interactions.
- Tree and boosting methods usually need less scaling; useful engineered features are ratios, date parts, leakage-safe aggregates, and meaningful categorical encodings.
- Naive Bayes often prefers nonnegative, discretized, or count-like features.
- Clustering, dimensionality reduction, and anomaly methods are sensitive to scale and sparse dummy expansion.
- Statsmodels methods benefit from interpretable bins, low collinearity, and small feature sets.
- Imbalanced-learning methods should apply feature engineering before sampling, and sampling must remain inside train/CV only.

## Visual Diagnostics

Every run writes a feature-engineering folder:

```text
<DATA_DIR>/runs/<task-group>/<method>/<timestamp>/feature_engineering/
```

Typical files:

- `recommendations.json`: method-aware recommendations and candidate rules.
- `candidate_rules.json`: ready-to-edit rule suggestions.
- `numeric_skew.png`: skewed numeric features worth transforming.
- `mutual_information.png`: supervised feature-signal ranking when a target exists.

Implemented visual tooling uses `matplotlib`, `seaborn`, and sklearn mutual information. Suitable
future optional libraries are `feature-engine` for additional sklearn-compatible transformers,
`featuretools` for automated relational feature synthesis, `tsfresh` for time-series features, and
`ydata-profiling` for heavier profile reports.
