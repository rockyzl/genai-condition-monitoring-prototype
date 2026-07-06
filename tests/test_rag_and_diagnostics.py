"""Tests for the retriever, diagnostic assistant, and evidence builder.

Self-contained: a tiny fixture knowledge base and synthetic model artifacts are
built in ``tmp_path``, so these run green independently of the Data Scientist's
timing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# --- make `src` importable regardless of invocation dir -------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import build_evidence as be  # noqa: E402
from src.rag.assistant import diagnose, normalize_ws  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "failure_modes.md").write_text(
        "# Failure Modes\n\n"
        "## HPC Degradation\n\n"
        "As compressor efficiency falls, outlet temperature and turbine gas "
        "temperature rise together in a monotonic degradation trend.\n\n"
        "## Bearing Wear\n\n"
        "Bearing wear is a mechanical degradation mode that appears as rising "
        "vibration amplitude at characteristic frequencies.\n"
    )
    (kb / "maintenance_review_checklist.md").write_text(
        "# Maintenance Review Checklist\n\n"
        "## Verify Data Quality\n\n"
        "Before trusting any flag, verify data quality and check for frozen or "
        "railed sensor values as the first inspection step.\n\n"
        "## Escalate When Uncertain\n\n"
        "If the model and the sensor evidence disagree, escalate to a senior "
        "reviewer for human review rather than acting on the score alone.\n"
    )
    return kb


@pytest.fixture
def synthetic_evidence() -> dict:
    return {
        "asset_id": 7,
        "last_cycle": 120,
        "predicted_rul": 42.5,
        "true_rul_eval_only": 40.0,
        "risk_band": "medium",
        "sensor_summary": {
            "sensor_4": {"mean": 1401.2, "std": 3.1, "trend": "increasing",
                         "window_cycles": 30},
            "sensor_11": {"mean": 47.3, "std": 0.4, "trend": "decreasing",
                          "window_cycles": 30},
        },
        "top_contributing_signals": [
            {"feature": "sensor_4_roll_mean", "importance": 0.128},
            {"feature": "sensor_11_roll_mean", "importance": 0.089},
        ],
        "uncertainty_note": "Point estimate; no calibrated interval. Human review required.",
        "model": "random_forest_baseline",
    }


# =========================================================================
# Retriever
# =========================================================================
def test_retriever_returns_relevant_chunk(kb_dir: Path):
    r = Retriever(kb_dir)
    assert len(r) >= 4  # four sections across two files
    results = r.retrieve("compressor efficiency temperature degradation trend", k=4)
    assert results, "expected at least one retrieved chunk"
    top = results[0]
    assert top["source_file"] == "failure_modes.md"
    assert top["section"] == "HPC Degradation"
    assert top["score"] > 0.0


def test_retriever_empty_kb_returns_nothing(tmp_path: Path):
    empty = tmp_path / "empty_kb"
    empty.mkdir()
    r = Retriever(empty)
    assert len(r) == 0
    assert r.retrieve("anything at all", k=4) == []


# =========================================================================
# Assistant
# =========================================================================
def test_assistant_output_contract(kb_dir: Path, synthetic_evidence: dict):
    r = Retriever(kb_dir)
    report = diagnose(synthetic_evidence, r)

    # Governance guarantees
    assert report["human_review_required"] is True
    assert report["citations"], "report must cite retrieved evidence"
    assert normalize_ws(report["uncertainty"])
    assert report["safety_note"]

    # Summary must reference the actual predicted RUL number
    assert str(synthetic_evidence["predicted_rul"]) in report["summary"]

    # Supporting evidence references the sensor facts
    joined = " ".join(report["supporting_evidence"])
    assert "sensor_4" in joined

    # Every *sourced* failure-mode / next-step claim traces to a KB chunk
    kb_texts = [c["text"] for c in r.chunks]
    for claim in report["possible_failure_modes"] + report["recommended_next_steps"]:
        if not claim.get("source_file"):
            continue
        excerpt = normalize_ws(claim.get("evidence") or claim.get("detail") or "")
        assert any(excerpt in normalize_ws(t) for t in kb_texts), (
            f"claim not grounded in KB: {excerpt!r}"
        )


def test_assistant_says_so_when_nothing_retrieved(tmp_path: Path, synthetic_evidence: dict):
    empty = tmp_path / "empty_kb"
    empty.mkdir()
    r = Retriever(empty)
    report = diagnose(synthetic_evidence, r)
    # No invented causes: the explicit "none retrieved" marker, no citations.
    assert report["citations"] == []
    assert report["possible_failure_modes"][0]["source_file"] is None
    assert "none retrieved" in report["possible_failure_modes"][0]["failure_mode"]
    assert report["human_review_required"] is True


# =========================================================================
# Evidence builder
# =========================================================================
def _write_raw(path: Path) -> None:
    rows = []
    for unit in (1, 2):
        for cycle in range(1, 6):
            vals = [unit, cycle, 0.0, 0.0, 100.0] + [
                500.0 + cycle + s for s in range(21)
            ]
            rows.append(vals)
    df = pd.DataFrame(rows, columns=be.RAW_COLS)
    df.to_csv(path, sep=" ", header=False, index=False)


def test_build_evidence_on_synthetic(tmp_path: Path):
    raw = tmp_path / "test_raw.txt"
    _write_raw(raw)

    pred = tmp_path / "test_predictions.csv"
    pd.DataFrame(
        {
            "unit_id": [1, 2],
            "last_cycle": [5, 5],
            "true_rul": [40, 90],
            "pred_rul": [38.2, 105.4],
            "risk_band": ["medium", "low"],
        }
    ).to_csv(pred, index=False)

    fi = tmp_path / "feature_importances.csv"
    pd.DataFrame(
        {
            "feature": [
                "sensor_4_roll_mean", "sensor_11_roll_mean", "sensor_2",
                "sensor_15_roll_mean", "sensor_7", "sensor_12_roll_mean",
            ],
            "importance": [0.13, 0.09, 0.07, 0.06, 0.05, 0.04],
        }
    ).to_csv(fi, index=False)

    out = tmp_path / "evidence"
    written = be.run(pred, fi, raw, out, window=3)
    assert len(written) == 2

    ev = json.loads((out / "unit_1.json").read_text())
    assert ev["asset_id"] == 1
    assert ev["model"] == "random_forest_baseline"
    assert ev["risk_band"] == "medium"
    assert "true_rul_eval_only" in ev and "true_rul" not in ev  # eval-only label
    assert isinstance(ev["predicted_rul"], (int, float))
    assert ev["sensor_summary"], "sensor summary must not be empty"
    # top sensors resolved from feature importances (sensor_4 was top)
    assert "sensor_4" in ev["sensor_summary"]
    assert ev["top_contributing_signals"][0]["feature"] == "sensor_4_roll_mean"
    assert ev["uncertainty_note"]
