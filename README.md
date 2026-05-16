# clml

Classical machine learning playground: 121 method families with unified
exploration, preprocessing, Optuna tuning, MLflow logging, and visualization.

## Setup

```bash
make venv && make setenv && make data
source .venv/bin/activate
```

## Quick Start

```bash
clml list-methods                                 # browse all 121 methods
clml run --method random_forest_classifier        # train and log to MLflow
clml runs last --method random_forest_classifier  # review results
make mlflow                                       # open http://127.0.0.1:5000
```

## Common Commands

```bash
# Discovery
clml datasets
clml dataexplore --dataset credit_risk
clml feature-advice --method random_forest_classifier --dataset credit_risk
clml suggest-methods --data mydata.csv --target-column label

# Training
clml run --method gradient_boosting_classifier --trials 8
clml run --method random_forest_classifier --feature-engineering \
         --feature-rules examples/feature_rules/credit_risk.json
clml run-all --trials 0                           # smoke-test every method

# Review
clml runs list
clml runs last
clml runs export --output .data/runs_summary.csv
```

Without activating `.venv`, prefix commands with `uv run`.

## Workflow

1. `clml list-methods` → pick candidates (see [docs/methods.md](docs/methods.md))
2. `clml dataexplore` → understand the dataset
3. `clml feature-advice` → get preprocessing hints
4. `clml run --trials 0` → smoke-test
5. `clml run --trials 20` → tune the best candidates
6. `make mlflow` → compare runs, inspect registered models

## Docs

| File | Contents |
|------|----------|
| [docs/quickstart.md](docs/quickstart.md) | Step-by-step getting started |
| [docs/methods.md](docs/methods.md) | All 121 methods with notes |
| [docs/datasets.md](docs/datasets.md) | Built-in dataset catalog |
| [docs/mlflow.md](docs/mlflow.md) | MLflow integration and Model Registry |
| [docs/feature_engineering.md](docs/feature_engineering.md) | Feature rules reference |
| [docs/extended_methods.md](docs/extended_methods.md) | Third-party library methods |
| [docs/method_recommendations.md](docs/method_recommendations.md) | `suggest-methods` guide |

## Project Layout

```
src/clml/       Python package
docs/           Reference documentation
examples/       Feature rule examples
scripts/        Make targets shell scripts
.data/          Run outputs and dataset cache (gitignored)
```
