import pytest

from src.ml import data as data_module
from src.ml.data import URGENCY_LEVELS, URGENCY_MAP


def test_urgency_map_covers_all_condition_labels():
    assert set(URGENCY_MAP.keys()) == {1, 2, 3, 4, 5}
    assert set(URGENCY_MAP.values()) <= set(URGENCY_LEVELS)


@pytest.mark.parametrize(
    "condition_label, expected_urgency",
    [(4, "urgente"), (1, "atenção"), (3, "atenção"), (2, "normal"), (5, "normal")],
)
def test_urgency_mapping_matches_documented_decision(condition_label, expected_urgency):
    assert URGENCY_MAP[condition_label] == expected_urgency


def test_load_csv_maps_urgency_and_keeps_expected_columns(tmp_path, monkeypatch):
    csv_path = tmp_path / "medical_tc_train.csv"
    csv_path.write_text(
        "condition_label,medical_abstract\n"
        '4,"acute cardiac event"\n'
        '2,"routine digestive checkup"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)

    df = data_module.load_train_df()

    assert list(df.columns) == ["text", "condition_label", "urgency"]
    assert len(df) == 2
    assert df.iloc[0]["urgency"] == "urgente"
    assert df.iloc[1]["urgency"] == "normal"


def test_load_csv_raises_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        data_module.load_train_df()
