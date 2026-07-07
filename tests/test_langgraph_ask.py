"""Tests for the LangGraph ask-path engine (narrow scope).

LangGraph is used only on the deterministic tool-calling (ask) path — never the
pipeline or supervisor. These tests assert the ask path is **equivalent** to the
native ask path: same grounded bilingual answer, same input-validation boundary,
and determinism. Skipped cleanly when ``langgraph`` is not installed (dev-only).

Run:
    .venv/bin/python -m pytest tests/test_langgraph_ask.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("langgraph", reason="langgraph is a dev-only engine dependency")

from src.agent.graph import answer_query_langgraph, build_ask_graph, make_lc_tools  # noqa: E402
from src.agent.query import answer_query  # noqa: E402
from src.agent.registry import Registry, ToolError  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402

CANONICAL = ["which engines need inspection?", "diagnose unit 81"]


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.load()


@pytest.fixture(scope="module", autouse=True)
def _artifacts(cfg: PipelineConfig):
    if not (cfg.path("data_processed") / "test_predictions.csv").exists():
        from src.pipeline.runner import run_pipeline
        from src.pipeline.specs import STAGE_ORDER

        run_pipeline(cfg, list(STAGE_ORDER))


# --- 1. equivalence: native ask == langgraph ask -----------------------------
@pytest.mark.parametrize("query", CANONICAL)
def test_ask_equivalence(cfg, tmp_path, query):
    native = answer_query(cfg, query, out_dir=tmp_path / "n")
    lang = answer_query_langgraph(cfg, query, out_dir=tmp_path / "l")

    na, la = native["answer"], lang["answer"]
    # identical grounded bilingual answer content + citations
    assert na["answer_en"] == la["answer_en"]
    assert na["answer_zh"] == la["answer_zh"]
    assert na["claims"] == la["claims"]
    assert na.get("citations") == la.get("citations")
    assert lang["grounded"] is True

    # same tool calls executed, in the same order (same trace schema)
    n_calls = [(t.tool, t.args) for t in native["trace"].tool_calls]
    l_calls = [(t.tool, t.args) for t in lang["trace"].tool_calls]
    assert n_calls == l_calls
    assert lang["trace"].kind == native["trace"].kind == "query"
    assert lang["trace"].all_claims_grounded()


# --- 2. out-of-vocab rejection parity ----------------------------------------
def test_out_of_vocab_rejection_parity(cfg, tmp_path):
    reg = Registry(cfg)
    # the registry boundary rejects a bad choice natively...
    with pytest.raises(ToolError):
        reg.call("list_units_by_risk", {"band": "purple"})
    # ...and the SAME validation is preserved through the LangGraph tool wrapper
    lst = next(t for t in make_lc_tools(reg) if t.name == "list_units_by_risk")
    with pytest.raises(ToolError):
        lst.func(band="purple")

    # an unmapped query yields the same honest "couldn't map" answer on both
    native = answer_query(cfg, "what is the weather", out_dir=tmp_path / "n")["answer"]
    lang = answer_query_langgraph(cfg, "what is the weather", out_dir=tmp_path / "l")["answer"]
    assert native["intent"] == lang["intent"] == "unmapped"
    assert native["answer_en"] == lang["answer_en"]
    assert lang["claims"] == []


# --- 3. determinism ----------------------------------------------------------
def test_determinism(cfg, tmp_path):
    a = answer_query_langgraph(cfg, "which engines need inspection?", out_dir=tmp_path / "a")
    b = answer_query_langgraph(cfg, "which engines need inspection?", out_dir=tmp_path / "b")
    assert a["answer"]["answer_en"] == b["answer"]["answer_en"]
    assert a["answer"]["claims"] == b["answer"]["claims"]
    # the graph really executed a tool-calling loop (ToolNode ran the registry)
    graph = build_ask_graph(Registry(cfg))
    plan = [{"tool": "list_units_by_risk", "args": {"band": "high"},
             "fan_out": ["get_prediction"]}]
    final = graph.invoke({"query": "q", "plan": plan, "phase": "start", "messages": [],
                          "answer": None, "citations": [], "trace_ref": None})
    names = {m.name for m in final["messages"] if m.__class__.__name__ == "ToolMessage"}
    assert "list_units_by_risk" in names and "get_prediction" in names
