import pandas as pd

from clml.data.adapters import write_frame
from clml.methods.registry import list_methods
from clml.recommendations.methods import advise_methods_for_file


def _covered_methods(advice):
    recommended = {item.method for item in advice.recommendations}
    not_recommended = {method for group in advice.not_recommended for method in group.methods}
    return recommended | not_recommended


def test_method_advice_detects_time_series_cumulative_data(tmp_path):
    path = tmp_path / "losses.csv"
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "day": [1, 2, 3],
            "tank": [10, 12, 15],
            "drone": [100, 150, 210],
        }
    )
    write_frame(path, frame)

    advice = advise_methods_for_file(path)

    assert advice.inferred_task == "time_series_cumulative_regression"
    assert advice.time_column == "date"
    assert advice.target_column == "drone"
    ridge = next(item for item in advice.recommendations if item.method == "ridge_regression")
    assert "correlated" in ridge.why_suggested
    assert any(group.category == "Classification methods" for group in advice.not_recommended)


def test_method_advice_assesses_every_implemented_method_for_classification(tmp_path):
    path = tmp_path / "classification.csv"
    frame = pd.DataFrame(
        {
            "age": [22, 45, 31, 55, 29, 48, 37, 61, 33, 41, 26, 58] * 3,
            "income": [35, 82, 44, 91, 38, 75, 57, 102, 49, 64, 41, 97] * 3,
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
        }
    )
    write_frame(path, frame)

    advice = advise_methods_for_file(path, target_column="target", max_recommendations=8)

    all_methods = {spec.name for spec in list_methods()}
    assert _covered_methods(advice) == all_methods
    assert "feature_select_k_best_f_classif" in {item.method for item in advice.recommendations}


def test_method_advice_recommends_timeseries_and_assesses_all_methods(tmp_path):
    path = tmp_path / "series.csv"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=48, freq="MS").astype(str),
            "month_index": range(48),
            "target": [100 + idx * 2 + (idx % 12) * 3 for idx in range(48)],
        }
    )
    write_frame(path, frame)

    advice = advise_methods_for_file(path, target_column="target", max_recommendations=6)

    assert advice.inferred_task == "timeseries"
    assert "statsmodels_exponential_smoothing" in {item.method for item in advice.recommendations}
    assert _covered_methods(advice) == {spec.name for spec in list_methods()}
