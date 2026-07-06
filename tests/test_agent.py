"""Tests for the Phase-C agent layer (registry, planner, cards, autopilot, query).

Covers the reliability contract the plan calls non-negotiable: input validation
at the tool boundary, the two canonical planner intents, bounded autonomy
(dry-run raises nothing / auto refuses triage + sign-off), card signals
reproducible from artifacts, trace grounding (every claim maps to a tool output),
two-run determinism, checkpoint/resume via provenance, and the anti-silent-
weakening thresholds hash.

Run:
    .venv/bin/python -m pytest tests/test_agent.py -q

These tests read the artifacts the pipeline already produced (predictions,
evidence, diagnostics, metrics, model_selection). Autopilot walks skip every
stage via provenance, so they are fast; a session fixture rebuilds the pipeline
only if a required artifact is genuinely missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import cards as cards_mod  # noqa: E402
from src.agent.autopilot import AgentGateConfig, Autopilot  # noqa: E402
from src.agent.planner import (  # noqa: E402
    LLMPlanner,
    PlannerNotConfigured,
    RuleBasedPlanner,
    make_planner,
)
from src.agent.query import answer_query  # noqa: E402
from src.agent.registry import Registry, ToolError  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.load()


@pytest.fixture(scope="module", autouse=True)
def _pipeline_artifacts(cfg: PipelineConfig):
    """Ensure the artifacts the agent reads exist; build them once if not."""
    required = [
        cfg.path("data_processed") / "test_predictions.csv",
        cfg.path("reports") / "metrics_model.json",
        cfg.path("evidence") / "_evidence_manifest.json",
        cfg.path("diagnostics") / "_diagnostics_manifest.json",
        cfg.path("reports") / "evaluation_summary.md",
    ]
    if not all(p.exists() for p in required):
        from src.pipeline.runner import run_pipeline
        from src.pipeline.specs import STAGE_ORDER

        run_pipeline(cfg, list(STAGE_ORDER))
    return cfg


# --- 1. registry input validation --------------------------------------------
def test_registry_rejects_bad_input(cfg):
    reg = Registry(cfg)
    with pytest.raises(ToolError):
        reg.call("nonexistent_tool", {})
    with pytest.raises(ToolError):
        reg.call("get_prediction", {"unit": 81, "bogus": 1})  # unknown arg
    with pytest.raises(ToolError):
        reg.call("get_prediction", {})  # missing required arg
    with pytest.raises(ToolError):
        reg.call("get_prediction", {"unit": "not-an-int"})  # bad type
    with pytest.raises(ToolError):
        reg.call("list_units_by_risk", {"band": "purple"})  # bad choice


# --- 2. registry tools return grounded artifact facts ------------------------
def test_registry_tools_are_grounded(cfg):
    import pandas as pd

    reg = Registry(cfg)
    df = pd.read_csv(cfg.path("data_processed") / "test_predictions.csv")

    n_high = int((df["risk_band"] == "high").sum())
    listed = reg.call("list_units_by_risk", {"band": "high"})
    assert listed.ok and listed.output["n"] == n_high
    # results are ordered by ascending predicted RUL (most urgent first)
    ruls = [u["pred_rul"] for u in listed.output["units"]]
    assert ruls == sorted(ruls)

    unit = int(listed.output["units"][0]["unit_id"])
    pred = reg.call("get_prediction", {"unit": unit})
    csv_val = float(df[df["unit_id"] == unit]["pred_rul"].iloc[0])
    assert pred.ok and abs(pred.output["pred_rul"] - csv_val) < 1e-6

    diag = reg.call("diagnose", {"unit": unit})
    assert diag.ok and diag.output["citations"]
    assert diag.output["human_review_required"] is True

    hits = reg.call("retrieve", {"q": "compressor degradation maintenance", "k": 3})
    assert hits.ok and hits.output["n_hits"] >= 1


# --- 3. planner plans for the two canonical queries --------------------------
def test_planner_plans_two_canonical():
    p = RuleBasedPlanner()

    diag_plan = p.plan("diagnose unit 81")
    assert [c.tool for c in diag_plan] == ["get_evidence", "retrieve", "diagnose"]
    assert diag_plan[0].args == {"unit": 81}
    assert diag_plan[2].args == {"unit": 81}

    insp_plan = p.plan("which engines need inspection?")
    assert len(insp_plan) == 1
    assert insp_plan[0].tool == "list_units_by_risk"
    assert insp_plan[0].args == {"band": "high"}
    assert "get_prediction" in insp_plan[0].fan_out


# --- 4. LLM planner is a flagged, not-configured stub ------------------------
def test_llm_planner_stub_raises():
    with pytest.raises(PlannerNotConfigured):
        make_planner("llm").plan("anything")
    assert isinstance(make_planner("llm"), LLMPlanner)
    assert make_planner("rule").kind == "rule"


# --- 5. dry-run raises no cards but reports what WOULD raise ------------------
def test_dry_run_raises_no_cards_reports_would(cfg, tmp_path):
    pilot = Autopilot(cfg, autonomy="dry-run", out_dir=tmp_path)
    report = pilot.run()

    assert report.status == "done"
    # no card was written to the inbox
    assert list((tmp_path / "autopilot_inbox" / "pending").glob("*.json")) == []
    assert report.cards_pending == []
    # but it reports the blocking cards it WOULD raise
    kinds = {w["kind"] for w in report.would_raise}
    assert "high_risk_triage" in kinds and "report_signoff" in kinds
    assert all(w["would_block"] for w in report.would_raise)
    # the run still produced a journal + trace
    assert (tmp_path / "autopilot_journal.jsonl").exists()
    assert Path(report.trace_path).exists()


# --- 6. auto refuses to auto-pass triage and sign-off ------------------------
def test_auto_refuses_triage_and_signoff(cfg, tmp_path):
    pilot = Autopilot(cfg, autonomy="auto", out_dir=tmp_path)
    report = pilot.run()

    # auto stops at the triage card — it may never auto-pass it
    assert report.status == "awaiting_input"
    assert report.cards_pending[0]["kind"] == "high_risk_triage"
    assert not any(c["kind"] == "high_risk_triage" for c in report.cards_resolved)

    # policy is data, and it forbids auto/safe-default for both governance cards
    signoff = cards_mod.build_signoff_card(
        cfg, "reports/metrics_model.json", "reports/evaluation_summary.md", 100
    )
    assert not signoff.auto_passable()
    assert not signoff.safe_default_applicable()
    triage = cards_mod.build_triage_card(
        cfg, "data/processed/test_predictions.csv", "reports/metrics_model.json", 10.0
    )
    assert not triage.auto_passable()
    assert triage.safe_default_applicable()  # a human at the CLI may accept it


# --- 7. every card signal reproduces from its artifact + field ---------------
def test_card_signals_reproducible(cfg):
    triage = cards_mod.build_triage_card(
        cfg, "data/processed/test_predictions.csv", "reports/metrics_model.json", 10.0
    )
    signoff = cards_mod.build_signoff_card(
        cfg, "reports/metrics_model.json", "reports/evaluation_summary.md", 100
    )
    for card in (triage, signoff):
        assert len(card.signals) <= 3
        assert cards_mod.all_signals_grounded(cfg, card), card.id
        # each signal's field genuinely resolves from its named artifact
        for s in card.signals:
            val = cards_mod.resolve_field(cfg, s.artifact, s.field)
            assert val is not cards_mod._MISSING, (s.artifact, s.field)


# --- 8. answer claims are grounded in tool outputs ---------------------------
def test_trace_grounding(cfg, tmp_path):
    for query in ("which engines need inspection?", "diagnose unit 81"):
        result = answer_query(cfg, query, out_dir=tmp_path)
        trace = result["trace"]
        assert result["grounded"] is True
        assert trace.claims  # the answer actually made claims
        # every claim's value is present at its cited field of its source call
        for claim in trace.claims:
            assert trace.claim_is_grounded(claim), (query, claim.field)
        # a claim that points at a wrong field must NOT verify (guards the check)
        from src.agent.trace import Claim

        bogus = Claim("x", trace.claims[0].source_seq, "t", "no_such_field", 123)
        assert trace.claim_is_grounded(bogus) is False


# --- 9. determinism: same inputs + seed → identical decisions ----------------
def test_determinism_excluding_timestamps(cfg, tmp_path):
    r1 = Autopilot(cfg, autonomy="dry-run", out_dir=tmp_path / "a").run()
    r2 = Autopilot(cfg, autonomy="dry-run", out_dir=tmp_path / "b").run()
    assert r1.decisions == r2.decisions
    assert [w["card_id"] for w in r1.would_raise] == [w["card_id"] for w in r2.would_raise]
    assert r1.thresholds_hash == r2.thresholds_hash


# --- 10. checkpoint/resume: answer a card → walk continues from provenance ----
def test_checkpoint_resume(cfg, tmp_path):
    pilot = Autopilot(cfg, autonomy="gated", out_dir=tmp_path)
    run1 = pilot.run()
    assert run1.status == "awaiting_input"
    triage = run1.cards_pending[0]
    assert triage["kind"] == "high_risk_triage"
    assert (tmp_path / "autopilot_inbox" / "pending" / f"{triage['id']}.json").exists()

    # a human answers the triage card
    answered = tmp_path / "autopilot_inbox" / "answered"
    answered.mkdir(parents=True, exist_ok=True)
    (answered / f"{triage['id']}.json").write_text(
        json.dumps({"card_id": triage["id"], "action": "schedule_inspection"})
    )

    run2 = Autopilot(cfg, autonomy="gated", out_dir=tmp_path).run()
    # triage now resolves via the answer, and the walk advances to sign-off
    assert any(c["kind"] == "high_risk_triage" and c["source"] == "answered"
               for c in run2.cards_resolved)
    assert run2.status == "awaiting_input"
    assert run2.cards_pending[0]["kind"] == "report_signoff"
    # earlier stages were skipped via provenance, later stages executed the walk
    stages = {row["stage"]: row for row in run2.stages}
    assert stages["s01_ingest"]["skipped"] is True
    for later in ("s08_evidence", "s09_diagnose", "s10_eval"):
        assert later in stages  # the resumed walk reached them


# --- 11. anti-silent-weakening: gate thresholds are hashed into the trace -----
def test_thresholds_hash_detects_weakening(cfg, tmp_path):
    base = AgentGateConfig.from_pipeline(cfg)
    weaker = AgentGateConfig.from_pipeline(cfg)
    import dataclasses

    weaker = dataclasses.replace(weaker, champion_margin_rmse=base.champion_margin_rmse + 5)
    assert base.hash() != weaker.hash()

    report = Autopilot(cfg, autonomy="dry-run", out_dir=tmp_path).run()
    trace = json.loads(Path(report.trace_path).read_text())
    assert trace["thresholds_hash"] == base.hash()


# --- 12. the supervisor streams gate events through the journal ---------------
def test_journal_emits_gate_events(cfg, tmp_path):
    from src.pipeline.journal import read_events

    Autopilot(cfg, autonomy="gated", out_dir=tmp_path).run()
    events = read_events(tmp_path / "autopilot_journal.jsonl")
    types = [e["type"] for e in events]
    assert "run_started" in types and "stage_started" in types
    raised = [e for e in events if e["type"] == "gate_raised"]
    assert raised and raised[0]["kind"] == "high_risk_triage"
    assert "card_id" in raised[0] and "payload_summary" in raised[0]
