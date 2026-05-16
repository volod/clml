from clml.data.adapters import CsvDictReader, CsvDictWriter, read_frame


def test_csv_dict_adapter_round_trips_typed_frame(tmp_path):
    path = tmp_path / "records.csv"
    CsvDictWriter().write(
        path,
        [
            {"feature": 1.5, "label": 0, "group": "a"},
            {"feature": 2.0, "label": 1, "group": "b"},
        ],
    )

    rows = CsvDictReader().read(path)
    frame = read_frame(path)

    assert rows == [
        {"feature": "1.5", "label": "0", "group": "a"},
        {"feature": "2.0", "label": "1", "group": "b"},
    ]
    assert frame["feature"].tolist() == [1.5, 2.0]
    assert frame["label"].tolist() == [0, 1]
    assert frame["group"].tolist() == ["a", "b"]
