# Quick Start

## Installation

```bash
git clone <repo>
cd clml
make venv       # creates .venv and installs all dependencies
make setenv     # copies .env.example → .env
make data       # caches all built-in datasets to <DATA_DIR>/datasets/
source .venv/bin/activate
```

Or use `uv run clml ...` without activating the environment.

## First run

```bash
clml run --method random_forest_classifier
```

This will:
- Load the `credit_risk` dataset
- Explore it and write plots to `<DATA_DIR>/runs/classification/random_forest_classifier/<timestamp>/exploration/`
- Generate feature engineering advice in `feature_engineering/`
- Train with no Optuna tuning (default `CLML_OPTUNA_TRIALS=0`)
- Log the run to MLflow (SQLite backend at `<DATA_DIR>/service/mlflow.db`)
- Register the fitted sklearn Pipeline in the **MLflow Model Registry**
- Print metrics to the terminal

## Explore a dataset

```bash
clml dataexplore --dataset credit_risk
```

Writes correlation heatmap, feature distributions, and preprocessing suggestions
to `<DATA_DIR>/runs/classification/dataexplore/credit_risk/`.

## Review results

Terminal:
```bash
clml runs last --method random_forest_classifier
```

MLflow UI (compare all runs):
```bash
make mlflow
# Open http://127.0.0.1:5000
```

Each run shows:
- **Parameters** — method, dataset, task, feature engineering settings
- **Metrics** — accuracy / R² / silhouette / AUC depending on task
- **`metric_interpretation` tag** — plain-English summary, e.g.:
  `Accuracy 91.4%; Macro-F1 87.2%; AUC 0.953 (excellent)`
- **Artifacts** — model, predictions, plots, exploration reports
- **Models tab** — registered sklearn Pipelines (one version per run)

## Tune a method

```bash
clml run --method gradient_boosting_classifier --trials 20
```

Optuna searches the method's param space for 20 trials using 3-fold CV.
Best params are stored in `run.json` and logged to MLflow.

## Run all methods

```bash
clml run-all --trials 0
```

Smoke-tests every method. Summary files are written to `<DATA_DIR>/reports/run_all/`.
Use `learning_insights.md` for practical lessons and `summary.csv` for raw metrics.

## Feature engineering

```bash
# Get advice for a specific method/dataset combination
clml feature-advice --method random_forest_classifier --dataset credit_risk

# Apply rules during training
clml run --method random_forest_classifier \
         --feature-engineering \
         --feature-rules examples/feature_rules/credit_risk.json
```

## Analyze your own data file

```bash
clml suggest-methods --data mydata.csv --target-column churn
clml run-data --method gradient_boosting_classifier --data mydata.csv --target-column churn
```

## Human learning path

If you are using this project to learn applied ML rather than only run experiments, follow
[learning_path.md](learning_path.md). It sequences the implemented methods from statistical
baselines through geometry, ensembles, feature engineering, unsupervised learning, time-aware
methods, survival analysis, streaming, and optimization.

## Key environment variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CLML_DATA_DIR` | `.data` | Root directory for all outputs (datasets, runs, service, reports) |
| `CLML_OPTUNA_TRIALS` | `8` | Hyperparameter trials per run |
| `CLML_RANDOM_STATE` | `42` | Global random seed |
| `CLML_MLFLOW_TRACKING_URI` | `sqlite:///<DATA_DIR>/service/mlflow.db` | MLflow backend (auto-derived from DATA_DIR when empty) |
| `CLML_FEATURE_ENGINEERING` | `false` | Apply feature rules by default |
