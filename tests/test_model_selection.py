"""Tests for the Phase-B model-selection bake-off.

Covers the leakage-safe protocol (GroupKFold-by-unit with zero unit crossing and
identical folds shared by every candidate), the selection JSON schema, the
champion-beats-the-Ridge-floor HALT, and deterministic reruns (same champion and
same CV numbers under the same seed).

The full three-candidate bake-off is expensive (it cross-validates a
RandomForest), so it is run once behind a module-scoped fixture; the fast tests
use a reduced-tree config or the pure decision function.

Run:
    .venv/bin/python -m pytest tests/test_model_selection.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import model_selection as ms  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402


# --- shared fixtures ---------------------------------------------------------
def _fast_cfg() -> PipelineConfig:
    """Real config but with a small forest, so the bake-off is quick. Fewer trees
    stays fully deterministic (random_state fixed) — only speed changes."""
    cfg = PipelineConfig.load()
    cfg.rf_params.n_estimators = 30
    return cfg


@pytest.fixture(scope="module")
def training_xy():
    """Real FD001 training design matrix, target, and per-row unit groups."""
    return ms.load_training_xy(PipelineConfig.load())


@pytest.fixture(scope="module")
def selection_run(tmp_path_factory):
    """Run the full bake-off once on the real config; return (summary, payload)."""
    d = tmp_path_factory.mktemp("selection")
    cfg = PipelineConfig.load()
    summary = ms.run_selection(
        cfg,
        md_path=d / "model_selection.md",
        json_path=d / "model_selection.json",
        champion_path=d / "champion.json",
    )
    payload = json.loads((d / "model_selection.json").read_text())
    md_text = (d / "model_selection.md").read_text()
    return summary, payload, md_text


# --- 1. GroupKFold has zero unit crossing ------------------------------------
def test_groupkfold_has_zero_unit_crossing(training_xy):
    X, y, groups, _ = training_xy
    folds = ms.build_folds(X, y, groups, ms.N_SPLITS)
    assert len(folds) == ms.N_SPLITS
    # the whole partition covers every row exactly once
    covered = sorted(idx for _, val in folds for idx in val)
    assert covered == list(range(len(y)))
    # no engine may straddle a train/validation boundary in any fold
    assert ms.units_cross_folds(folds, groups) is False
    for train_idx, val_idx in folds:
        assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))


# --- 2. identical folds across candidates ------------------------------------
def test_identical_folds_across_candidates(training_xy):
    X, y, groups, _ = training_xy
    folds = ms.build_folds(X, y, groups, ms.N_SPLITS)
    want = ms.fold_signature(folds)

    # Two DIFFERENT cheap estimators evaluated on the same folds must report the
    # same fold signature — the folds are the split, never the model.
    from sklearn.linear_model import Ridge

    a = ms.evaluate_candidate(lambda: Ridge(alpha=1.0), X, y, folds)
    b = ms.evaluate_candidate(lambda: Ridge(alpha=10.0), X, y, folds)
    assert a["fold_signature"] == b["fold_signature"] == want

    # A different fold count yields a different split (guards against a constant).
    other = ms.fold_signature(ms.build_folds(X, y, groups, 4))
    assert other != want


def test_selection_shares_one_fold_set(selection_run):
    _, payload, _ = selection_run
    sigs = {c["fold_signature"] for c in payload["candidates"]}
    assert len(sigs) == 1
    assert payload["protocol"]["identical_folds"] is True
    assert sigs == {payload["protocol"]["fold_signature"]}


# --- 3. selection json schema ------------------------------------------------
def test_selection_json_schema(selection_run):
    summary, payload, md_text = selection_run

    # top-level keys
    for key in ("champion", "seed", "protocol", "criteria", "candidates",
                "verdict", "champion_rationale", "halted"):
        assert key in payload, f"missing top-level key {key}"
    assert payload["seed"] == 42
    assert payload["halted"] is False
    assert payload["champion"] in {ms.RIDGE, ms.RANDOM_FOREST, ms.HIST_GBM}

    # protocol block documents the leakage-safe design
    proto = payload["protocol"]
    assert proto["cv"] == "GroupKFold"
    assert proto["n_splits"] == ms.N_SPLITS
    assert proto["group_by"] == "unit_id"
    assert proto["low_rul_threshold"] == ms.LOW_RUL_THRESHOLD

    # exactly the three candidates, each with both criterion families
    names = [c["name"] for c in payload["candidates"]]
    assert names == [ms.RIDGE, ms.RANDOM_FOREST, ms.HIST_GBM]
    for c in payload["candidates"]:
        for key in ("cv_rmse_mean", "cv_rmse_std", "cv_rmse_folds", "low_rul_rmse",
                    "low_rul_optimistic_fraction", "n_low_rul_rows", "role",
                    "how_it_works", "explains_itself", "complexity_rank"):
            assert key in c, f"candidate {c['name']} missing {key}"
        assert len(c["cv_rmse_folds"]) == ms.N_SPLITS
        assert 0.0 <= c["low_rul_optimistic_fraction"] <= 1.0

    # criteria are recorded in priority order (RMSE, calibration, simplicity)
    assert len(payload["criteria"]) == 3
    assert "RMSE" in payload["criteria"][0]

    # the lay + technical report tables and rationale are all present
    assert "Typical miss" in md_text and "CV-RMSE" in md_text
    assert "Champion rationale" in md_text
    assert summary["champion"] == payload["champion"]


# --- 4. champion-beats-floor assertion ---------------------------------------
def test_champion_must_beat_ridge_floor():
    # RandomForest no better than the Ridge floor → HALT (loud failure).
    below = {
        ms.RIDGE: {"cv_rmse_mean": 15.0, "low_rul_rmse": 20.0},
        ms.RANDOM_FOREST: {"cv_rmse_mean": 16.0, "low_rul_rmse": 22.0},
        ms.HIST_GBM: {"cv_rmse_mean": 15.5, "low_rul_rmse": 21.0},
    }
    with pytest.raises(ms.ChampionBelowFloorError):
        ms.choose_champion(below, margin_rmse=1.0)

    # RF clears the floor and the challenger is not clearly better → RF stays.
    rf_wins = {
        ms.RIDGE: {"cv_rmse_mean": 20.0, "low_rul_rmse": 30.0},
        ms.RANDOM_FOREST: {"cv_rmse_mean": 17.0, "low_rul_rmse": 25.0},
        ms.HIST_GBM: {"cv_rmse_mean": 16.8, "low_rul_rmse": 24.9},  # gaps < margin
    }
    v = ms.choose_champion(rf_wins, margin_rmse=1.0)
    assert v["champion"] == ms.RANDOM_FOREST
    assert v["swapped_from_default"] is False
    assert v["beats_floor"] is True

    # Challenger clearly better on BOTH criteria → it unseats the incumbent.
    hgb_wins = {
        ms.RIDGE: {"cv_rmse_mean": 20.0, "low_rul_rmse": 30.0},
        ms.RANDOM_FOREST: {"cv_rmse_mean": 17.0, "low_rul_rmse": 25.0},
        ms.HIST_GBM: {"cv_rmse_mean": 15.0, "low_rul_rmse": 22.0},
    }
    v2 = ms.choose_champion(hgb_wins, margin_rmse=1.0)
    assert v2["champion"] == ms.HIST_GBM
    assert v2["swapped_from_default"] is True

    # Better on CV-RMSE only (not calibration) is NOT enough → RF stays.
    cv_only = {
        ms.RIDGE: {"cv_rmse_mean": 20.0, "low_rul_rmse": 30.0},
        ms.RANDOM_FOREST: {"cv_rmse_mean": 17.0, "low_rul_rmse": 25.0},
        ms.HIST_GBM: {"cv_rmse_mean": 15.0, "low_rul_rmse": 24.6},  # low-RUL gap < margin
    }
    assert ms.choose_champion(cv_only, margin_rmse=1.0)["champion"] == ms.RANDOM_FOREST


# --- 5. deterministic rerun --------------------------------------------------
def test_deterministic_rerun(tmp_path):
    """Same seed → same champion AND byte-identical CV numbers on a rerun."""
    cfg = _fast_cfg()

    def _run(sub: str) -> dict:
        d = tmp_path / sub
        d.mkdir()
        ms.run_selection(
            cfg,
            md_path=d / "model_selection.md",
            json_path=d / "model_selection.json",
            champion_path=d / "champion.json",
        )
        return json.loads((d / "model_selection.json").read_text())

    a = _run("run_a")
    b = _run("run_b")

    assert a["champion"] == b["champion"]
    assert a["verdict"] == b["verdict"]
    a_cv = {c["name"]: c["cv_rmse_folds"] for c in a["candidates"]}
    b_cv = {c["name"]: c["cv_rmse_folds"] for c in b["candidates"]}
    assert a_cv == b_cv
    a_low = {c["name"]: c["low_rul_rmse"] for c in a["candidates"]}
    b_low = {c["name"]: c["low_rul_rmse"] for c in b["candidates"]}
    assert a_low == b_low


# --- 6. real-config champion contract ----------------------------------------
def test_default_champion_is_randomforest(selection_run):
    """On the real config the deterministic default holds: RF wins, beats floor,
    and the model artifacts are NOT swapped."""
    summary, payload, _ = selection_run
    assert summary["champion"] == ms.RANDOM_FOREST
    assert summary["beats_floor"] is True
    assert summary["swapped"] is False
    assert summary["floor_gap_cycles"] > 0
