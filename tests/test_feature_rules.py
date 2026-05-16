import pandas as pd

from clml.features.rules import RuleBasedFeatureEngineer


def test_rule_based_feature_engineer_adds_common_features():
    frame = pd.DataFrame(
        {
            "event_date": ["2024-01-01", "2024-01-06"],
            "age": [29, 42],
            "loan_amount": [100.0, 200.0],
            "annual_income": [1000.0, 4000.0],
        }
    )
    transformer = RuleBasedFeatureEngineer(
        [
            {"type": "datetime_parts", "column": "event_date", "parts": ["day_of_week"]},
            {
                "type": "numeric_bins",
                "column": "age",
                "bins": [0, 30, 120],
                "labels": ["young", "adult"],
                "output_column": "age_group",
            },
            {
                "type": "ratio",
                "numerator": "loan_amount",
                "denominator": "annual_income",
                "output_column": "loan_to_income",
            },
            {"type": "log1p", "column": "annual_income"},
        ]
    )

    result = transformer.fit_transform(frame)

    assert "event_date_day_of_week" in result.columns
    assert "age_group" in result.columns
    assert "loan_to_income" in result.columns
    assert "log1p_annual_income" in result.columns
    assert result["age_group"].tolist() == ["young", "adult"]
