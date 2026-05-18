import json

import pandas as pd

from clml.constants import ARTIFACT_RUN_ALL_LEARNING_CSV, ARTIFACT_RUN_ALL_LEARNING_MD
from clml.runs.learning import learning_rows
from clml.runs.review import write_run_all_summary


def test_learning_rows_flags_auc_macro_f1_gap():
    rows = [
        {
            "method": "random_forest_classifier",
            "task": "classification",
            "dataset": "credit_risk",
            "status": "ok",
            "metrics": {"accuracy": 0.71, "f1_macro": 0.60, "roc_auc": 0.81},
            "summary": "Accuracy 71.0%; Macro-F1 60.0%; AUC 0.810 (good)",
        }
    ]

    [row] = learning_rows(rows)

    assert row["primary_metric"] == "f1_macro"
    assert row["signal"] == "moderate"
    assert "AUC is much higher than Macro-F1" in row["caution_flags"]


def test_write_run_all_summary_adds_learning_artifacts(tmp_path):
    rows = [
        {
            "method": "hist_gradient_boosting_regressor",
            "task": "regression",
            "dataset": "housing_prices",
            "guide_section": "Ensembles",
            "status": "ok",
            "run_dir": "runs/regression/hist_gradient_boosting_regressor/run",
            "metrics": {"r2": 0.87, "rmse": 57_000.0, "mae": 42_000.0},
            "summary": "R2 0.870",
        },
        {
            "method": "affinity_propagation",
            "task": "clustering",
            "dataset": "blobs",
            "guide_section": "Clustering",
            "status": "ok",
            "run_dir": "runs/clustering/affinity_propagation/run",
            "metrics": {"clusters": 10, "silhouette": 0.19, "adjusted_rand": 0.52},
            "summary": "10 clusters found",
        },
    ]

    write_run_all_summary(rows, tmp_path)

    assert (tmp_path / "summary.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text())[0]["method"]
    markdown = (tmp_path / ARTIFACT_RUN_ALL_LEARNING_MD).read_text()
    assert "Best Practical Baselines By Task" in markdown
    frame = pd.read_csv(tmp_path / ARTIFACT_RUN_ALL_LEARNING_CSV)
    assert set(frame["method"]) == {"hist_gradient_boosting_regressor", "affinity_propagation"}

