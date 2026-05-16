"""Specialized classification runners: imbalanced learning and categorical encoding."""

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from clml.config.settings import get_settings
from clml.constants import (
    BALANCED_RF_N_ESTIMATORS,
    HASHING_ENCODER_COMPONENTS,
    LOGISTIC_MAX_ITER,
    LOO_ENCODER_SIGMA,
    TARGET_ENCODER_SMOOTHING,
    TEST_SIZE,
)
from clml.data.adapters import write_frame
from clml.features.rules import RuleBasedFeatureEngineer
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.preprocessing import build_preprocessor
from clml.pipelines.tasks._shared import _finish, _write_json
from clml.reporting.plots import plot_confusion_matrix

# ---------------------------------------------------------------------------
# Imbalanced classification
# ---------------------------------------------------------------------------


def _imbalanced_pipeline_steps(
    spec_name: str, seed: int, feature_steps: list, preprocessor
) -> list:
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    match spec_name:
        case "imbalanced_smote_logistic":
            return [
                *feature_steps,
                ("preprocess", preprocessor),
                ("sampler", SMOTE(random_state=seed)),
                ("model", LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=seed)),
            ]
        case "imbalanced_random_undersampling_logistic":
            return [
                *feature_steps,
                ("preprocess", preprocessor),
                ("sampler", RandomUnderSampler(random_state=seed)),
                ("model", LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=seed)),
            ]
        case "imbalanced_balanced_random_forest":
            return [
                *feature_steps,
                ("preprocess", preprocessor),
                (
                    "model",
                    BalancedRandomForestClassifier(
                        n_estimators=BALANCED_RF_N_ESTIMATORS,
                        random_state=seed,
                        sampling_strategy="all",
                        replacement=True,
                        bootstrap=False,
                    ),
                ),
            ]
        case _:
            raise ValueError(f"Unsupported imbalanced method: {spec_name}")


def run_imbalanced_classification(ctx: RunContext) -> RunResult:
    from imblearn.pipeline import Pipeline as ImbalancedPipeline

    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state, stratify=y
    )
    seed = get_settings().random_state
    feature_steps: list = []
    x_for_preprocessor = x_train
    if ctx.feature_engineering:
        engineer = RuleBasedFeatureEngineer(ctx.feature_rules)
        x_for_preprocessor = engineer.fit_transform(x_train)
        feature_steps.append(("features", engineer))
    steps = _imbalanced_pipeline_steps(
        ctx.spec.name, seed, feature_steps, build_preprocessor(x_for_preprocessor)
    )
    pipeline = ImbalancedPipeline(steps=steps)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics: dict = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "minority_fraction_train": float(y_train.value_counts(normalize=True).min()),
    }
    write_frame(
        ctx.run_dir / "predictions.csv",
        pd.DataFrame({"actual": y_test, "predicted": predictions, "probability": probabilities}),
    )
    _write_json(
        ctx.run_dir / "classification_report.json",
        classification_report(y_test, predictions, output_dict=True),
    )
    plot_confusion_matrix(y_test, predictions, ctx.run_dir / "confusion_matrix.png")
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, {}, {})


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def _categorical_encoder_and_model(spec_name: str, categorical_columns: list, seed: int):
    from category_encoders import HashingEncoder, LeaveOneOutEncoder, TargetEncoder

    _factories = {
        "category_target_encoder_logistic": lambda: (
            TargetEncoder(cols=categorical_columns, smoothing=TARGET_ENCODER_SMOOTHING),
            LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=seed),
        ),
        "category_leave_one_out_encoder_logistic": lambda: (
            LeaveOneOutEncoder(
                cols=categorical_columns, sigma=LOO_ENCODER_SIGMA, random_state=seed
            ),
            LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=seed),
        ),
        "category_hashing_encoder_sgd": lambda: (
            HashingEncoder(cols=categorical_columns, n_components=HASHING_ENCODER_COMPONENTS),
            SGDClassifier(loss="log_loss", random_state=seed),
        ),
    }
    factory = _factories.get(spec_name)
    if factory is None:
        raise ValueError(f"Unsupported categorical encoding method: {spec_name}")
    return factory()


def run_categorical_encoding(ctx: RunContext) -> RunResult:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    x, y = ctx.bundle.x, ctx.bundle.y
    assert y is not None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=get_settings().random_state, stratify=y
    )
    seed = get_settings().random_state
    feature_steps: list = []
    x_for_encoder = x_train
    if ctx.feature_engineering:
        engineer = RuleBasedFeatureEngineer(ctx.feature_rules)
        x_for_encoder = engineer.fit_transform(x_train)
        feature_steps.append(("features", engineer))
    categorical_columns = x_for_encoder.select_dtypes(exclude="number").columns.tolist()
    encoder, model = _categorical_encoder_and_model(ctx.spec.name, categorical_columns, seed)
    pipeline = Pipeline(
        steps=[
            *feature_steps,
            ("encoder", encoder),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics: dict = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        output = {"actual": y_test, "predicted": predictions, "probability": probabilities}
    else:
        output = {"actual": y_test, "predicted": predictions}
    write_frame(ctx.run_dir / "predictions.csv", pd.DataFrame(output))
    _write_json(
        ctx.run_dir / "classification_report.json",
        classification_report(y_test, predictions, output_dict=True),
    )
    plot_confusion_matrix(y_test, predictions, ctx.run_dir / "confusion_matrix.png")
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, pipeline, metrics, {}, {})
