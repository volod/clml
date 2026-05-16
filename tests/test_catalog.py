from clml.data.catalog import available_dataset_names, load_dataset


def test_realistic_datasets_are_available():
    names = available_dataset_names()
    assert "credit_risk" in names
    assert "housing_prices" in names
    assert "airline_passengers" in names


def test_credit_risk_has_mixed_features():
    bundle = load_dataset("credit_risk", prepare=False)
    assert bundle.info.rows >= 3000
    assert "home_ownership" in bundle.info.feature_columns
    assert bundle.frame["target"].nunique() == 2


def test_housing_prices_has_regression_target():
    bundle = load_dataset("housing_prices", prepare=False)
    assert bundle.info.rows >= 3000
    assert bundle.frame["target"].nunique() > 100


def test_extended_datasets_are_available():
    names = available_dataset_names()
    assert "customer_survival" in names
    assert "portfolio_assets" in names
    assert "production_planning" in names
    assert "marketing_mix" in names
    assert "streaming_churn" in names
