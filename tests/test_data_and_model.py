"""Tests for the C-MAPSS loader, feature builder, and prediction contract.

Run:
    .venv/bin/python -m pytest tests/test_data_and_model.py -q
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_cmapss import (  # noqa: E402
    COLUMNS,
    RUL_CAP,
    add_training_rul,
    last_cycle_per_unit,
    load_raw,
    load_test_rul,
)
from src.features.build_features import (  # noqa: E402
    DROPPED_SENSORS,
    INFORMATIVE_SENSORS,
    build_features,
    feature_columns,
)

PROCESSED = ROOT / "data" / "processed"
PRED_CONTRACT_COLS = ["unit_id", "last_cycle", "true_rul", "pred_rul", "risk_band"]
FI_CONTRACT_COLS = ["feature", "importance"]


# --- Loader ------------------------------------------------------------------


def test_load_raw_shape_and_columns():
    df = load_raw("FD001", "train")
    assert list(df.columns) == COLUMNS
    assert len(COLUMNS) == 26
    assert df["unit"].nunique() == 100  # FD001 has 100 training units
    assert df["unit"].dtype.kind == "i"
    assert df["cycle"].dtype.kind == "i"
    assert not df.isna().any().any()


def test_load_test_and_rul_align():
    test = load_raw("FD001", "test")
    rul = load_test_rul("FD001")
    assert test["unit"].nunique() == 100
    assert len(rul) == 100
    assert rul.index.min() == 1 and rul.index.max() == 100


# --- RUL computation on a toy frame -----------------------------------------


def test_training_rul_toy_frame():
    # Unit 1 runs 5 cycles, unit 2 runs 3 cycles.
    toy = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
        }
    )
    out = add_training_rul(toy, cap=None)
    # RUL = max_cycle - cycle, per unit.
    assert out.loc[out["unit"] == 1, "rul"].tolist() == [4, 3, 2, 1, 0]
    assert out.loc[out["unit"] == 2, "rul"].tolist() == [2, 1, 0]
    # Input frame not mutated.
    assert "rul" not in toy.columns


def test_training_rul_cap_applied():
    toy = pd.DataFrame({"unit": [1] * 5, "cycle": [1, 2, 3, 4, 5]})
    out = add_training_rul(toy, cap=2)
    assert out["rul"].tolist() == [2, 2, 2, 1, 0]  # clipped at 2


def test_default_cap_is_125():
    toy = pd.DataFrame({"unit": [1, 1], "cycle": [1, 500]})
    out = add_training_rul(toy)  # default cap
    assert out.loc[out["cycle"] == 1, "rul"].iloc[0] == RUL_CAP


def test_last_cycle_per_unit():
    toy = pd.DataFrame(
        {"unit": [1, 1, 2, 2, 2], "cycle": [1, 9, 3, 4, 8], "sensor_2": [0, 1, 2, 3, 4]}
    )
    last = last_cycle_per_unit(toy)
    assert last["unit"].tolist() == [1, 2]
    assert last["cycle"].tolist() == [9, 8]
    assert last["sensor_2"].tolist() == [1, 4]


# --- Feature builder ---------------------------------------------------------


def test_dropped_and_informative_sensors_disjoint():
    assert set(DROPPED_SENSORS).isdisjoint(INFORMATIVE_SENSORS)
    assert len(INFORMATIVE_SENSORS) == 14


def test_build_features_columns_and_no_nan():
    df = load_raw("FD001", "train")
    feat, cols = build_features(df)
    # 14 sensors * (value + roll_mean + roll_std) = 42 features.
    assert cols == feature_columns()
    assert len(cols) == 42
    assert feat[cols].isna().sum().sum() == 0
    # Rolling features never leak across units: unit boundary keeps counts intact.
    assert len(feat) == len(df)


# --- Prediction contract (requires train_baseline.py to have run) ------------


@pytest.mark.parametrize(
    "path,cols",
    [
        (PROCESSED / "test_predictions.csv", PRED_CONTRACT_COLS),
        (PROCESSED / "feature_importances.csv", FI_CONTRACT_COLS),
    ],
)
def test_output_files_exist_with_contract_columns(path, cols):
    assert path.exists(), f"{path} missing — run src/models/train_baseline.py first"
    df = pd.read_csv(path)
    assert list(df.columns) == cols


def test_predictions_contract_content():
    df = pd.read_csv(PROCESSED / "test_predictions.csv")
    assert len(df) == 100  # one row per FD001 test unit
    assert df["unit_id"].is_unique
    assert set(df["risk_band"].unique()) <= {"high", "medium", "low"}
    # risk_band must match the pred_rul thresholds exactly.
    def band(p):
        return "high" if p <= 30 else ("medium" if p <= 80 else "low")

    assert (df["risk_band"] == df["pred_rul"].map(band)).all()
    assert (df["pred_rul"] >= 0).all()
