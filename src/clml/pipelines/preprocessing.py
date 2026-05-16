import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from clml.features.rules import RuleBasedFeatureEngineer


def build_preprocessor(x: pd.DataFrame, *, nonnegative: bool = False) -> ColumnTransformer:
    numeric_columns = x.select_dtypes(include="number").columns.tolist()
    categorical_columns = [col for col in x.columns if col not in numeric_columns]
    scaler = MinMaxScaler() if nonnegative else StandardScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def build_model_pipeline(
    x: pd.DataFrame,
    model,
    *,
    nonnegative: bool = False,
    feature_engineering: bool = False,
    feature_rules: list[dict] | None = None,
) -> Pipeline:
    steps = []
    if feature_engineering:
        engineer = RuleBasedFeatureEngineer(feature_rules or [])
        x_for_preprocessor = engineer.fit_transform(x)
        steps.append(("features", engineer))
    else:
        x_for_preprocessor = x
    steps.extend(
        [
            ("preprocess", build_preprocessor(x_for_preprocessor, nonnegative=nonnegative)),
            ("model", model),
        ]
    )
    return Pipeline(steps=steps)
