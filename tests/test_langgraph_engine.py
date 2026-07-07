"""Tests for the LangGraph engine lane (Phase: two-engine upgrade).

The LangGraph engine must be governance-equivalent to the hand-written native
supervisor: same tools, same gates, same cards, same trace/journal artifacts.
These tests exercise LangGraph's two headline mechanics (typed state, tool
calling) and assert the flagship property — **parity**: native and langgraph
produce identical decisions and cards on the same inputs + seed.

Skipped cleanly when ``langgraph`` is not installed (dev-only dependency).

Run:
    .venv/bin/python -m pytest tests/test_langgraph_engine.py -q
"""

from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("langgraph", reason="langgraph is a dev-only engine dependency")

from langgraph.types import Command  # noqa: E402

from src.agent.autopilot import Autopilot  # noqa: E402
from src.agent.langgraph_engine import (  # noqa: E402
    AskState,
    AutopilotState,
    LangGraphAutopilot,
    answer_query_langgraph,
    build_ask_graph,
    make_lc_tools,
)
from src.agent.query import answer_query  # noqa: E402
from src.agent.registry import Registry, ToolError  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.load()


@pytest.fixture(scope="module", autouse=True)
def _artifacts(cfg: PipelineConfig):
    if not (cfg.path("data_processed") / "test_predictions.csv").exists():
        from src.pipeline.runner import run_pipeline
        from src.pipeline.specs import STAGE_ORDER

        run_pipeline(cfg, list(STAGE_ORDER))


# --- 1. typed state schema round-trip ----------------------------------------
def test_typed_state_schema_and_accumulation(cfg, tmp_path):
    # the typed schemas declare the documented channels...
    ap = typing.get_type_hints(AutopilotState, include_extras=True)
    for ch in ("run_id", "autonomy", "thresholds_hash", "seed", "stage_results",
               "gate_outcomes", "decisions", "cards_resolved", "would_raise",
               "cursor", "halted"):
        assert ch in ap
    # accumulator channels carry a reducer (Annotated metadata), control ones don't
    import operator

    assert operator.add in typing.get_args(ap["decisions"])
    assert typing.get_args(ap["cursor"]) == ()  # last-write-wins
    assert "messages" in typing.get_type_hints(AskState, include_extras=True)

    # ...and the reducers actually accumulate across the 10 stage nodes
    prep = LangGraphAutopilot(cfg, autonomy="dry-run", out_dir=tmp_path).prepare()
    final = prep["app"].invoke(prep["init"], prep["thread"])
    assert len(final["stage_results"]) == 10
    assert len(final["decisions"]) == 12  # 10 stages, s07 + s10 each add a 2nd gate


# --- 2. tool-calling loop executes registry tools, validation preserved -------
def test_tool_calling_loop_and_validation(cfg):
    reg = Registry(cfg)
    tools = make_lc_tools(reg)
    assert {t.name for t in tools} == set(reg.names())  # all 7 wrapped

    # a bad argument is still rejected at the registry boundary (validation kept)
    lst = next(t for t in tools if t.name == "list_units_by_risk")
    with pytest.raises(ToolError):
        lst.func(band="purple")

    # the real ToolNode loop executes list + fanned get_prediction calls
    graph = build_ask_graph(reg)
    plan = [{"tool": "list_units_by_risk", "args": {"band": "high"},
             "fan_out": ["get_prediction"]}]
    final = graph.invoke({"question": "q", "plan": plan, "phase": "start", "messages": []})
    tool_msgs = [m for m in final["messages"] if m.__class__.__name__ == "ToolMessage"]
    names = {m.name for m in tool_msgs}
    assert "list_units_by_risk" in names and "get_prediction" in names
    assert json.loads(tool_msgs[0].content)["output"]["n"] >= 1


# --- 3. interrupt actually pauses at the triage card -------------------------
def test_interrupt_pauses_at_triage(cfg, tmp_path):
    prep = LangGraphAutopilot(cfg, autonomy="gated", out_dir=tmp_path).prepare()
    final = prep["app"].invoke(prep["init"], prep["thread"])
    assert "__interrupt__" in final
    assert final["__interrupt__"][0].value["kind"] == "high_risk_triage"


# --- 4. resume continues to sign-off (in-process Command + cross-process) -----
def test_resume_continues_to_signoff(cfg, tmp_path):
    # in-process: interrupt at triage → Command(resume) → interrupt at sign-off
    prep = LangGraphAutopilot(cfg, autonomy="gated", out_dir=tmp_path / "ip").prepare()
    app, thread = prep["app"], prep["thread"]
    f1 = app.invoke(prep["init"], thread)
    assert f1["__interrupt__"][0].value["kind"] == "high_risk_triage"
    f2 = app.invoke(Command(resume="schedule_inspection"), thread)
    assert "__interrupt__" in f2
    assert f2["__interrupt__"][0].value["kind"] == "report_signoff"

    # cross-process: answered-card file + fresh run resumes via provenance
    out = tmp_path / "xp"
    r1 = LangGraphAutopilot(cfg, autonomy="gated", out_dir=out).run()
    triage = r1.cards_pending[0]
    answered = out / "autopilot_inbox" / "answered"
    answered.mkdir(parents=True, exist_ok=True)
    (answered / f"{triage['id']}.json").write_text(
        json.dumps({"card_id": triage["id"], "action": "schedule_inspection"}))
    r2 = LangGraphAutopilot(cfg, autonomy="gated", out_dir=out).run()
    assert any(c["kind"] == "high_risk_triage" for c in r2.cards_resolved)
    assert r2.cards_pending[0]["kind"] == "report_signoff"
    stages = {s["stage"]: s for s in r2.stages}
    assert stages["s01_ingest"]["skipped"] is True  # earlier stages skip via provenance


# --- 5. two-run determinism ---------------------------------------------------
def test_determinism(cfg, tmp_path):
    r1 = LangGraphAutopilot(cfg, autonomy="dry-run", out_dir=tmp_path / "a").run()
    r2 = LangGraphAutopilot(cfg, autonomy="dry-run", out_dir=tmp_path / "b").run()
    assert r1.decisions == r2.decisions
    assert [w["card_id"] for w in r1.would_raise] == [w["card_id"] for w in r2.would_raise]
    assert r1.thresholds_hash == r2.thresholds_hash


# --- 6. eval Section D governs the langgraph engine unmodified ----------------
def test_section_d_passes_on_langgraph_artifacts(cfg, tmp_path):
    from src.eval.run_eval import eval_autonomy_governance

    LangGraphAutopilot(cfg, autonomy="gated", yes_safe_defaults=True, out_dir=tmp_path).run()
    answer_query_langgraph(cfg, "diagnose unit 81", out_dir=tmp_path)
    g = eval_autonomy_governance(reports_dir=tmp_path, cfg=cfg)
    assert g["status"] == "ok"
    assert [c["status"] for c in g["checks"]] == ["pass"] * 5


# --- 7. PARITY — native vs langgraph produce identical decisions + cards ------
@pytest.mark.parametrize("autonomy", ["dry-run", "gated", "auto"])
def test_parity_native_vs_langgraph(cfg, tmp_path, autonomy):
    native = Autopilot(cfg, autonomy=autonomy, out_dir=tmp_path / "n").run()
    lang = LangGraphAutopilot(cfg, autonomy=autonomy, out_dir=tmp_path / "l").run()
    assert native.decisions == lang.decisions
    assert native.status == lang.status
    assert ([c["id"] for c in native.cards_pending]
            == [c["id"] for c in lang.cards_pending])
    assert ([c["card_id"] for c in native.cards_resolved]
            == [c["card_id"] for c in lang.cards_resolved])
    assert ([w["card_id"] for w in native.would_raise]
            == [w["card_id"] for w in lang.would_raise])


# --- 8. PARITY — ask answers byte-identical across engines -------------------
@pytest.mark.parametrize("query", ["which engines need inspection?", "diagnose unit 81"])
def test_ask_parity(cfg, tmp_path, query):
    n = answer_query(cfg, query, out_dir=tmp_path / "n")["answer"]
    l = answer_query_langgraph(cfg, query, out_dir=tmp_path / "l")["answer"]
    assert n["answer_en"] == l["answer_en"]
    assert n["answer_zh"] == l["answer_zh"]
    assert n["claims"] == l["claims"]
    assert n.get("citations") == l.get("citations")
