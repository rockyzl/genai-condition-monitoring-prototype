"""Diagnostic evidence layer.

Reads the Data Scientist contract artifacts
(``data/processed/test_predictions.csv`` and
``data/processed/feature_importances.csv``) plus the raw C-MAPSS test file, and
writes one structured JSON evidence record per test unit under
``data/processed/evidence/unit_<id>.json``.

The evidence record is the hand-off to the RAG diagnostic assistant: it captures
what the model predicted, which signals drove the prediction, and a compact
last-window summary of the raw sensors, together with an explicit, honest
uncertainty note. ``true_rul`` is carried only under a key that marks it as
ground truth for evaluation, never as a model input.

This module is intentionally dependency-light (pandas + numpy) and defines the
shared path/column constants reused by the app and the eval harness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
PRED_PATH = DATA_PROCESSED / "test_predictions.csv"
FI_PATH = DATA_PROCESSED / "feature_importances.csv"
RAW_TEST_PATH = ROOT / "data" / "raw" / "CMAPSSData" / "test_FD001.txt"
EVIDENCE_DIR = DATA_PROCESSED / "evidence"
KB_DIR = ROOT / "docs" / "knowledge_base"

# --- C-MAPSS raw column layout (26 columns) -------------------------------
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS

# --- Tunables -------------------------------------------------------------
LAST_WINDOW = 30      # cycles of history summarized per unit
TOP_K_SENSORS = 6     # sensors summarized (from importance ranking)
TOP_K_SIGNALS = 6     # feature-importance rows reported verbatim
MODEL_NAME = "random_forest_baseline"

UNCERTAINTY_NOTE = (
    "This remaining-useful-life value is a single point estimate from a "
    "baseline model and carries no calibrated confidence interval. The training "
    "RUL target is capped with a piecewise-linear ceiling (commonly 125 cycles), "
    "so predictions for healthy early-life units are compressed toward that cap "
    "and must not be read as precise cycle counts. Prediction error is largest "
    "near end-of-life, exactly where it matters most. Use this output as "
    "decision support to prioritize inspection, not as an authoritative "
    "time-to-failure. Human review is required."
)


def load_raw_test(path: Path = RAW_TEST_PATH) -> pd.DataFrame:
    """Load a C-MAPSS ``test_FD00X.txt`` into a named 26-column frame.

    The raw files are whitespace-separated with trailing spaces; we read all
    whitespace-delimited fields, keep the first 26, and apply the standard
    column names.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(RAW_COLS)]
    df.columns = RAW_COLS
    return df


def load_predictions(path: Path = PRED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"unit_id", "last_cycle", "true_rul", "pred_rul", "risk_band"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df


def load_feature_importances(path: Path = FI_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"feature", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def rank_sensors_from_importances(
    fi: pd.DataFrame, k: int = TOP_K_SENSORS
) -> list[str]:
    """Return up to ``k`` raw sensor columns, ordered by feature importance.

    Feature names may be raw (``sensor_4``) or derived (``sensor_4_roll_mean``);
    both resolve to the underlying ``sensor_4`` column. The first time a sensor
    appears (highest importance) fixes its rank.
    """
    ordered: list[str] = []
    for feat in fi["feature"]:
        m = re.search(r"sensor[_ ]?(\d{1,2})", str(feat), flags=re.IGNORECASE)
        if not m:
            continue
        col = f"sensor_{int(m.group(1))}"
        if col in SENSOR_COLS and col not in ordered:
            ordered.append(col)
        if len(ordered) >= k:
            break
    return ordered


def _trend_direction(values: np.ndarray) -> str:
    """Classify a short series as increasing / decreasing / flat via slope."""
    n = len(values)
    if n < 2:
        return "flat"
    std = float(np.std(values))
    slope = float(np.polyfit(np.arange(n), values, 1)[0])
    total_change = slope * (n - 1)
    if std == 0 or abs(total_change) < 0.5 * std:
        return "flat"
    return "increasing" if total_change > 0 else "decreasing"


def sensor_summary_for_unit(
    unit_df: pd.DataFrame, sensors: list[str], window: int = LAST_WINDOW
) -> dict:
    """Last-window mean/std/trend for the given sensors of one unit."""
    tail = unit_df.sort_values("cycle").tail(window)
    summary: dict[str, dict] = {}
    for col in sensors:
        vals = tail[col].to_numpy(dtype=float)
        summary[col] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "trend": _trend_direction(vals),
            "window_cycles": int(len(vals)),
        }
    return summary


def build_unit_evidence(
    pred_row: pd.Series,
    unit_df: pd.DataFrame,
    top_sensors: list[str],
    top_signals: list[dict],
    window: int = LAST_WINDOW,
) -> dict:
    """Assemble one evidence record. ``true_rul`` is eval-only, never a signal."""
    return {
        "asset_id": int(pred_row["unit_id"]),
        "last_cycle": int(pred_row["last_cycle"]),
        "predicted_rul": round(float(pred_row["pred_rul"]), 2),
        "true_rul_eval_only": float(pred_row["true_rul"]),
        "true_rul_note": "ground truth held out for evaluation only; not a model input",
        "risk_band": str(pred_row["risk_band"]),
        "sensor_summary": sensor_summary_for_unit(unit_df, top_sensors, window),
        "top_contributing_signals": top_signals,
        "uncertainty_note": UNCERTAINTY_NOTE,
        "model": MODEL_NAME,
    }


def run(
    pred_path: Path = PRED_PATH,
    fi_path: Path = FI_PATH,
    raw_path: Path = RAW_TEST_PATH,
    out_dir: Path = EVIDENCE_DIR,
    window: int = LAST_WINDOW,
) -> list[Path]:
    """Build evidence for every unit in ``pred_path``; return written paths."""
    preds = load_predictions(pred_path)
    fi = load_feature_importances(fi_path)
    raw = load_raw_test(raw_path)

    top_sensors = rank_sensors_from_importances(fi, TOP_K_SENSORS)
    if not top_sensors:
        # Fallback: importance file exposed no sensor-derived features. Pick the
        # highest-variance raw sensors so the summary is still informative.
        variances = raw[SENSOR_COLS].var().sort_values(ascending=False)
        top_sensors = list(variances.index[:TOP_K_SENSORS])

    top_signals = [
        {"feature": str(r.feature), "importance": round(float(r.importance), 6)}
        for r in fi.head(TOP_K_SIGNALS).itertuples()
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for _, pred_row in preds.iterrows():
        uid = int(pred_row["unit_id"])
        unit_df = raw[raw["unit"] == uid]
        if unit_df.empty:
            print(
                f"[build_evidence] WARNING: unit {uid} absent from raw file "
                f"{raw_path.name}; skipping.",
                file=sys.stderr,
            )
            continue
        evidence = build_unit_evidence(
            pred_row, unit_df, top_sensors, top_signals, window
        )
        out_path = out_dir / f"unit_{uid}.json"
        out_path.write_text(json.dumps(evidence, indent=2))
        written.append(out_path)
    return written


def _check_inputs() -> str | None:
    """Return a human-readable error if a required input is missing, else None."""
    for label, p in [
        ("predictions (test_predictions.csv)", PRED_PATH),
        ("feature importances (feature_importances.csv)", FI_PATH),
        ("raw test data (test_FD001.txt)", RAW_TEST_PATH),
    ]:
        if not p.exists():
            return (
                f"Required input not found: {label} at {p}.\n"
                "The diagnostic evidence layer depends on the Data Scientist "
                "artifacts. Run the data/model pipeline (Phases 2-3) first, then "
                "re-run build_evidence."
            )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, default=PRED_PATH)
    parser.add_argument("--fi", type=Path, default=FI_PATH)
    parser.add_argument("--raw", type=Path, default=RAW_TEST_PATH)
    parser.add_argument("--out", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--window", type=int, default=LAST_WINDOW)
    args = parser.parse_args()

    # Only enforce the default-path preflight when using defaults; custom paths
    # (e.g. tests) get direct, explicit errors from the loaders.
    if (args.pred, args.fi, args.raw) == (PRED_PATH, FI_PATH, RAW_TEST_PATH):
        err = _check_inputs()
        if err:
            print(err, file=sys.stderr)
            return 1

    written = run(args.pred, args.fi, args.raw, args.out, args.window)
    print(f"[build_evidence] wrote {len(written)} evidence records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
