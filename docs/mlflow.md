# MLflow Integration

Every `clml run` automatically logs to MLflow. The local SQLite backend is configured
in `.env` as `CLML_MLFLOW_TRACKING_URI=sqlite:///.data/mlflow.db`.

## Start the UI

```bash
make mlflow
# or: uv run mlflow ui --backend-store-uri sqlite:///.data/mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`.

## What gets logged

### Per run

| Item | MLflow location | Description |
|------|----------------|-------------|
| Params | Parameters tab | method, dataset, task, guide_section, feature_engineering, feature_rules_count |
| Metrics | Metrics tab | Task-specific numerics (accuracy, R², silhouette, AUC, …) |
| `metric_interpretation` tag | Tags tab | Plain-English metric summary |
| `description` tag | Tags tab | Method description from notes or guide section |
| `method_title` / `dataset_description` | Tags tab | Human-readable context |
| All output files | Artifacts tab | Plots, predictions, coefficients, model.joblib |
| Registered model | Models tab | sklearn Pipeline (one version per run) |

### Model Registry

All sklearn-compatible Pipelines are registered under the method name:

- Navigate to **Models** in the MLflow UI
- Each method has one registered model (e.g., `random_forest_classifier`)
- Each `clml run` adds a new version of that model
- Optimization/survival/statsmodels methods log artifacts but are not registered

**Load a registered model:**

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/random_forest_classifier/latest")
predictions = model.predict(x_test)
```

**Load by run URI:**

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///.data/mlflow.db")
model = mlflow.sklearn.load_model("runs:/<run_id>/registered_model")
```

## Understanding metric_interpretation

Every run includes a `metric_interpretation` tag with a plain-English summary:

| Task | Example tag value |
|------|-------------------|
| classification | `Accuracy 92.3%; Macro-F1 88.1%; AUC 0.961 (excellent)` |
| regression | `R² 0.847 (strong fit); RMSE 18.3; MAE 12.1` |
| feature_selection | `Selected 12 features; downstream macro-F1 94.7%` |
| timeseries | `24-step forecast; MAE 4.98; RMSE 6.03; MAPE 1.1%` |
| clustering | `4 clusters found; silhouette 0.54 (well-separated); ARI 0.712` |
| dimensionality | `8 components explain 93.4% of variance` |
| anomaly | `AUC 0.912 (excellent); Macro-F1 82.4%` |
| density | `Mean log-likelihood −1.23 ± 0.41` |
| linear_programming | `Optimal profit 48,200; avg resource utilization 91.3%` |
| nonlinear_optimization | `Optimized sales 2,140,000 (+18.5% lift vs current budget)` |
| cvxpy_portfolio | `Expected return 8.2%; volatility 12.1%; Sharpe-like 0.68` |
| survival_cox | `Cox C-index 0.731 test (good), 0.774 train` |
| survival_kaplan_meier | `Log-rank p=0.0031 (significant); event fraction 28.4%` |

AUC quality thresholds: ≥ 0.90 = excellent, ≥ 0.75 = good, < 0.75 = poor.
R² quality thresholds: ≥ 0.80 = strong, ≥ 0.50 = moderate, < 0.50 = weak.
Silhouette quality: > 0.5 = well-separated, > 0.2 = overlapping, ≤ 0.2 = poor.

## Comparing runs across methods

1. Open the **Experiments** view and select experiments for multiple methods
2. Click **Compare** → parallel coordinates or scatter plots of metrics
3. Use `clml runs export` to get a flat CSV for external analysis:

```bash
clml runs export --output .data/runs_summary.csv
```

## Querying MLflow programmatically

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///.data/mlflow.db")
client = mlflow.tracking.MlflowClient()

# List registered models
for m in client.search_registered_models():
    print(m.name, "—", len(m.latest_versions), "version(s)")

# Get all runs for a method
runs = mlflow.search_runs(experiment_names=["random_forest_classifier"])
print(runs[["metrics.accuracy", "metrics.roc_auc", "tags.metric_interpretation"]])
```
