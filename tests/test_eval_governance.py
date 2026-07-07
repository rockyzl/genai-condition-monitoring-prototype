"""Tests for eval Section D — Autonomy governance (Phase E).

Section D turns the Phase-C agent's own reliability assertions into
evaluation-report checks over the latest real agent-run artifacts. These tests
drive a real (provenance-skipped, fast) agent run into a tmp reports dir, then
assert the governance evaluation passes, detects tampering, and degrades to a
clear pending note when no agent run exists.

Run:
    .venv/bin/python -m pytest tests/test_eval_governance.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.autopilot import Autopilot  # noqa: E402
from src.agent.query import answer_query  # noqa: E402
from src.eval.run_eval import _fmt_governance, eval_autonomy_governance  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.load()


@pytest.fixture(scope="module", autouse=True)
def _artifacts(cfg: PipelineConfig):
    """Ensure the pipeline artifacts the agent reads exist (build once if not)."""
    if not (cfg.path("data_processed") / "test_predictions.csv").exists():
        from src.pipeline.runner import run_pipeline
        from src.pipeline.specs import STAGE_ORDER

        run_pipeline(cfg, list(STAGE_ORDER))


@pytest.fixture
def agent_run(cfg: PipelineConfig, tmp_path: Path) -> Path:
    """Generate a rich agent-run artifact set into an isolated reports dir.

    A gated ``--yes-safe-defaults`` walk (triage resolved, sign-off left pending)
    plus one grounded query — enough to exercise all five governance checks.
    """
    Autopilot(cfg, autonomy="gated", yes_safe_defaults=True, out_dir=tmp_path).run()
    answer_query(cfg, "diagnose unit 81", out_dir=tmp_path)
    return tmp_path


def test_section_d_all_checks_pass(cfg, agent_run):
    g = eval_autonomy_governance(reports_dir=agent_run, cfg=cfg)
    assert g["status"] == "ok"
    # every one of the five checks ran and passed (no skips, no fails)
    statuses = [c["status"] for c in g["checks"]]
    assert statuses == ["pass"] * 5
    assert g["n_failed"] == 0 and g["n_passed"] == 5
    # the renderer produces a Section D table with the anti-weakening check
    md = "\n".join(_fmt_governance(g))
    assert "✅ PASS" in md and "anti-silent-weakening" in md


def test_section_d_detects_threshold_tampering(cfg, agent_run):
    # silently weaken the recorded gate-threshold hash in the autopilot trace
    trace_path = sorted(agent_run.glob("agent_trace_auto_*.json"))[-1]
    data = json.loads(trace_path.read_text())
    data["thresholds_hash"] = "deadbeefdeadbeef"
    trace_path.write_text(json.dumps(data))

    g = eval_autonomy_governance(reports_dir=agent_run, cfg=cfg)
    assert g["status"] == "violations"
    hash_check = next(c for c in g["checks"] if "hash" in c["name"])
    assert hash_check["status"] == "fail"
    assert g["n_failed"] >= 1


def test_section_d_detects_ungrounded_card_signal(cfg, agent_run):
    # tamper a pending card so a signal no longer re-derives from its artifact
    pend = sorted((agent_run / "autopilot_inbox" / "pending").glob("*.json"))[-1]
    card = json.loads(pend.read_text())
    card["signals"][0]["field"] = "count:risk_band=not_a_band"  # resolves to 0, text mismatch
    pend.write_text(json.dumps(card))

    g = eval_autonomy_governance(reports_dir=agent_run, cfg=cfg)
    sig_check = next(c for c in g["checks"] if "Card signal" in c["name"])
    assert sig_check["status"] == "fail"
    assert sig_check["n_ok"] < sig_check["n_total"]


def test_section_d_pending_when_no_agent_run(cfg, tmp_path):
    g = eval_autonomy_governance(reports_dir=tmp_path, cfg=cfg)
    assert g["status"] == "pending"
    assert "no agent run" in g["reason"].lower()
    # the renderer degrades to a clear pending note (no table)
    md = "\n".join(_fmt_governance(g))
    assert "Pending" in md
