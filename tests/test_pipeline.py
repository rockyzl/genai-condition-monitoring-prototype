"""Tests for the explicit pipeline layer (Phase A).

Covers: StageSpec completeness (the enforce-explanation guard), the stage
registry/DAG contract, config round-trip + legacy-default equivalence,
provenance skip/idempotency, the journal event schema, single-stage execution +
second-run skip, ``--from`` selection, and the EDA drop-list corroboration.

Run:
    .venv/bin/python -m pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import train_baseline as tb  # noqa: E402
from src.pipeline import provenance  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.context import PipelineContext  # noqa: E402
from src.pipeline.journal import CORE_EVENT_TYPES, Journal, read_events  # noqa: E402
from src.pipeline.runner import (  # noqa: E402
    DAG_EDGES,
    build_manifest,
    run_pipeline,
    select_stages,
)
from src.pipeline.specs import STAGE_ORDER, STAGE_SPECS, all_specs  # noqa: E402
from src.pipeline.stages import STAGE_FUNCS  # noqa: E402


# --- 1. enforce-explanation --------------------------------------------------
def test_every_stage_spec_is_fully_explained():
    """No stage may ship with an empty declared field (incl. bilingual zh_*)."""
    offenders = {s.name: s.missing_fields() for s in all_specs() if s.missing_fields()}
    assert offenders == {}, f"stages with empty spec fields: {offenders}"
    # bilingual explanations must actually be present for every stage
    for s in all_specs():
        assert s.zh_what.strip() and s.zh_why.strip()


# --- 2. stage registry / DAG contract ---------------------------------------
def test_stage_registry_and_dag_are_consistent():
    assert list(STAGE_FUNCS.keys()) == STAGE_ORDER
    assert set(STAGE_SPECS.keys()) == set(STAGE_ORDER)
    assert len(STAGE_ORDER) == 10
    # every DAG edge references declared stages
    for a, b in DAG_EDGES:
        assert a in STAGE_SPECS and b in STAGE_SPECS


# --- 3. config round-trip + legacy defaults ----------------------------------
def test_config_roundtrip_and_defaults_match_legacy():
    cfg = PipelineConfig.load()
    # YAML faithfully encodes the dataclass defaults
    assert cfg == PipelineConfig()
    # dict round-trip is loss-free
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg
    # defaults reproduce the previously hard-coded behaviour
    assert cfg.seed == 42
    assert cfg.rul_cap == 125
    assert cfg.rolling_window == 5
    assert cfg.rf_params.sklearn_kwargs(cfg.seed) == tb.RF_PARAMS
    assert (cfg.risk_thresholds.high_max, cfg.risk_thresholds.medium_max) == (
        tb.HIGH_MAX,
        tb.MEDIUM_MAX,
    )


# --- 4. provenance signature + skip ------------------------------------------
def test_provenance_skip_and_idempotency(tmp_path: Path):
    inp = tmp_path / "input.txt"
    inp.write_text("alpha")
    out = tmp_path / "output.json"
    out.write_text("{}")
    params = {"k": 1}

    sig = provenance.build_signature("sXX", [inp], params, seed=42)
    provenance.write_provenance(out, sig)
    # same inputs + params + seed → current → skip
    assert provenance.is_stage_current([out], sig) is True
    # a second identical signature has the same skip key (timestamps ignored)
    sig2 = provenance.build_signature("sXX", [inp], params, seed=42)
    assert provenance.is_stage_current([out], sig2) is True

    # changed input → not current
    inp.write_text("beta")
    sig3 = provenance.build_signature("sXX", [inp], params, seed=42)
    assert provenance.is_stage_current([out], sig3) is False
    # changed params → not current
    sig4 = provenance.build_signature("sXX", [inp], {"k": 2}, seed=42)
    assert provenance.is_stage_current([out], sig4) is False
    # missing output → not current
    out.unlink()
    assert provenance.is_stage_current([out], sig) is False


# --- 5. journal event schema -------------------------------------------------
def test_journal_event_schema(tmp_path: Path):
    jpath = tmp_path / "journal.jsonl"
    j = Journal(jpath, run_id="run_test")
    j.run_started(["s01_ingest"])
    j.stage_started("s01_ingest", what="w", why="y")
    j.stage_progress("s01_ingest", "doing")
    j.artifact("s01_ingest", "data/x.json", {"rows": 5})
    j.stage_done("s01_ingest", seconds=0.1, rows=5)
    j.run_done(stages_run=1, stages_skipped=0, seconds=0.1)

    events = read_events(jpath)
    types = [e["type"] for e in events]
    # every core event type appears, exactly once here, in order
    assert set(types) == CORE_EVENT_TYPES
    # required fields + monotonic seq + stable run_id
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert all(e["run_id"] == "run_test" and "ts" in e for e in events)
    started = next(e for e in events if e["type"] == "stage_started")
    assert started["what"] == "w" and started["why"] == "y"
    art = next(e for e in events if e["type"] == "artifact")
    assert art["path"] == "data/x.json" and art["key_metrics"] == {"rows": 5}
    done = next(e for e in events if e["type"] == "stage_done")
    assert done["rows"] == 5 and done["skipped"] is False


# --- 6. single-stage execution + second-run skip -----------------------------
def _cfg_with_tmp_journal(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig.load()
    cfg.paths.journal = str(tmp_path / "journal.jsonl")  # absolute → isolated
    return cfg


def test_run_stage_then_skip_second_run(tmp_path: Path):
    cfg = _cfg_with_tmp_journal(tmp_path)
    manifest = cfg.path("data_processed") / "ingest_manifest.json"

    # force a real run so the test is independent of prior pipeline runs
    ctx1 = run_pipeline(cfg, ["s01_ingest"], force=True, run_id="run_a")
    assert ctx1.results[0].skipped is False
    assert manifest.exists()
    first_hash = provenance.hash_file(manifest)

    # unforced rerun with unchanged inputs → skipped, output untouched
    ctx2 = run_pipeline(cfg, ["s01_ingest"], force=False, run_id="run_b")
    assert ctx2.results[0].skipped is True
    assert provenance.hash_file(manifest) == first_hash

    # manifest renders the stage card + a mermaid DAG
    md = build_manifest(ctx1)
    assert "s01_ingest" in md and "```mermaid" in md and "Assumptions" in md


# --- 7. --from / --stage selection -------------------------------------------
def test_stage_selection():
    assert select_stages(all_=True, stage=None, from_=None) == STAGE_ORDER
    assert select_stages(all_=False, stage="s05_model", from_=None) == ["s05_model"]
    assert select_stages(all_=False, stage=None, from_="s07_predict") == [
        "s07_predict",
        "s08_evidence",
        "s09_diagnose",
        "s10_eval",
    ]
    with pytest.raises(SystemExit):
        select_stages(all_=False, stage="s99_bogus", from_=None)


# --- 8. EDA corroborates the hard-coded drop list ----------------------------
def test_eda_reproduces_feature_droplist():
    from src.features.build_features import DROPPED_SENSORS, INFORMATIVE_SENSORS
    from src.pipeline import eda as eda_mod

    summary = eda_mod.compute_eda("FD001")
    flat = {d["sensor"] for d in summary["flat_sensors"]}
    assert flat == set(DROPPED_SENSORS)
    assert len(summary["sensor_monotonicity"]) == len(INFORMATIVE_SENSORS) == 14
    # the ranking is sorted by descending |corr|
    corrs = [d["abs_corr"] for d in summary["sensor_monotonicity"]]
    assert corrs == sorted(corrs, reverse=True)


# --- 9. context helper wires spec ↔ result -----------------------------------
def test_stage_result_exposes_its_spec(tmp_path: Path):
    from src.pipeline.context import StageResult

    r = StageResult(name="s02_eda", skipped=False, seconds=1.0, rows=10)
    assert r.spec is STAGE_SPECS["s02_eda"]
    assert r.spec.zh_what  # bilingual explanation reachable from a result
