"""LangGraph on the tool-calling (ask) path — narrow, deterministic, key-free.

This is the *only* place LangGraph touches the system, and it touches **only the
query/ask path** — never the pipeline, the autopilot supervisor, or resume. The
hand-written pipeline supervisor (``src/agent/autopilot.py``) stays as-is; the
rationale for keeping the governance hand-rolled (provenance-hash resume → byte
determinism, inbox files → crash-safe replay, gates in the native layer) is in
``docs/langgraph-mapping.md``.

What this module does: express the deterministic ask loop as a LangGraph
``StateGraph`` with typed state and a real ``ToolNode``.

* **Typed state** (:class:`AskState`) carries the query, the standard
  tool_calls / tool_results message channel, and the composed answer + citations
  + trace ref.
* **Deterministic planner node** — the same :class:`RuleBasedPlanner` emits
  standard ``tool_calls`` (no LLM, no keys, runs today).
* **ToolNode** wraps the *same* :class:`src.agent.registry.Registry`, so the
  seven tools and their input validation are exactly the native ones.
* **Composer node** builds the same grounded, cited bilingual answer via the
  shared :func:`src.agent.query._compose`, writing the same trace schema.

The answer is byte-identical to :func:`src.agent.query.answer_query` — the graph
only changes *how* the tools are driven. **Future occupant:** the ``--planner
llm`` stub would emit ``tool_calls`` into this very ToolNode loop unchanged; the
loop is model-agnostic by construction, so wiring a real planner later requires
no change to the tools, the validation, or the grounding. No model is called
here today.

Requires ``langgraph`` (dev-only; ``requirements-agent.txt``). The base install
and the native ask path never import this module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import create_model

from src.agent.planner import ToolCall, make_planner
from src.agent.query import FANOUT_LIMIT, Step, _compose
from src.agent.registry import Registry, ToolResult, digest
from src.agent.trace import Trace
from src.pipeline.config import PipelineConfig


# =============================================================================
# Typed state
# =============================================================================
class AskState(TypedDict):
    """State flowing through the ask graph.

    ``messages`` is the standard LangGraph tool-calling channel (``add_messages``
    reducer): the planner node appends an ``AIMessage`` carrying **tool_calls**,
    the ``ToolNode`` appends ``ToolMessage`` **tool_results**. ``plan``/``phase``
    drive the deterministic loop (initial plan → optional fan-out → done).
    ``answer``/``citations``/``trace_ref`` are the composer's outputs.
    """

    query: str
    plan: list
    phase: str
    messages: Annotated[list, add_messages]
    answer: dict | None
    citations: list
    trace_ref: str | None


# =============================================================================
# Registry tools → LangGraph StructuredTools (validation preserved)
# =============================================================================
def _pydantic_args_model(reg: Registry, name: str):
    """Build a pydantic args schema for a registry tool from its ArgSpec list."""
    spec = reg.tools[name]
    fields: dict[str, tuple] = {}
    for a in spec.args:
        fields[a.name] = (a.type, ... if a.required else a.default)
    return create_model(f"{name}_args", **fields)


def make_lc_tools(reg: Registry) -> list[StructuredTool]:
    """Wrap the seven registry tools as LangGraph-executable StructuredTools.

    Each wrapper calls ``reg.call`` — so the registry's schema validation and its
    ``ToolError`` boundary are preserved — and returns the JSON-encoded
    :class:`ToolResult` the ToolNode carries as a ``ToolMessage``.
    """
    tools: list[StructuredTool] = []
    for name in reg.names():
        spec = reg.tools[name]
        args_model = _pydantic_args_model(reg, name)

        def _make(tool_name: str):
            def _fn(**kwargs) -> str:
                res = reg.call(tool_name, kwargs)  # validation preserved here
                return json.dumps({
                    "tool": res.tool, "args": res.args, "ok": res.ok,
                    "output": res.output, "error": res.error,
                })

            return _fn

        tools.append(StructuredTool.from_function(
            _make(name), name=name, description=spec.description,
            args_schema=args_model,
        ))
    return tools


def _plan_to_dicts(plan: list[ToolCall]) -> list[dict]:
    return [{"tool": c.tool, "args": c.args, "fan_out": list(c.fan_out)} for c in plan]


def build_ask_graph(reg: Registry):
    """Compile the deterministic tool-calling loop: planner → tools → composer.

    The ``agent`` node is a deterministic controller (no LLM): it emits the
    plan's tool_calls, then — for a fan-out plan — a second round of
    ``get_prediction`` calls per returned unit, then stops.
    """
    tool_node = ToolNode(make_lc_tools(reg))

    def agent(state: AskState) -> dict:
        plan = state["plan"]
        phase = state.get("phase", "start")
        if phase == "start":
            calls = [
                {"name": c["tool"], "args": c["args"], "id": f"call_{i}"}
                for i, c in enumerate(plan)
            ]
            if not calls:  # unmapped plan → nothing to execute
                return {"messages": [AIMessage(content="")], "phase": "done"}
            next_phase = "fanout" if any(c.get("fan_out") for c in plan) else "final"
            return {"messages": [AIMessage(content="", tool_calls=calls)],
                    "phase": next_phase}
        if phase == "fanout":
            fan_call = next((c for c in plan if c.get("fan_out")), None)
            units = _last_tool_output(state["messages"], fan_call["tool"]).get("units", [])
            calls = []
            for j, u in enumerate(units[:FANOUT_LIMIT]):
                for ft in fan_call["fan_out"]:
                    calls.append({"name": ft, "args": {"unit": u["unit_id"]},
                                  "id": f"fan_{ft}_{j}"})
            if not calls:
                return {"messages": [AIMessage(content="")], "phase": "done"}
            return {"messages": [AIMessage(content="", tool_calls=calls)],
                    "phase": "final"}
        return {"messages": [AIMessage(content="")], "phase": "done"}

    def route(state: AskState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    g = StateGraph(AskState)
    g.add_node("agent", agent)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def _last_tool_output(messages: list, tool_name: str) -> dict:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == tool_name:
            return json.loads(m.content).get("output") or {}
    return {}


def _ordered_tool_results(messages: list) -> list[ToolResult]:
    """Reconstruct ToolResults in tool_call order from the graph's messages.

    Pairs each ``AIMessage.tool_calls`` entry (name+args+id) with its
    ``ToolMessage`` (by id), preserving planned order so the reconstructed steps —
    and therefore the composed answer — match the native ask path exactly.
    """
    by_id: dict[str, ToolMessage] = {
        m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)
    }
    results: list[ToolResult] = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            tm = by_id.get(tc["id"])
            if tm is None:
                continue
            payload = json.loads(tm.content)
            output = payload.get("output")
            results.append(ToolResult(
                tool=tc["name"], args=tc["args"], ok=payload.get("ok", False),
                output=output, error=payload.get("error"),
                digest=digest(output) if payload.get("ok") else "",
            ))
    return results


def _steps_from_results(
    plan: list[ToolCall], results: list[ToolResult], trace: Trace
) -> list[Step]:
    """Rebuild the native :class:`Step` structure from executed results.

    Mirrors the native ask loop (plan call, then its fan-out) so seqs recorded
    into the trace match native and :func:`_compose` yields an identical answer.
    """
    steps: list[Step] = []
    idx = 0
    for call in plan:
        if idx >= len(results):
            break
        res = results[idx]
        seq = trace.record_tool(res)
        idx += 1
        fanned: list[tuple] = []
        if call.fan_out and res.ok and isinstance(res.output, dict):
            n_units = min(len(res.output.get("units", [])), FANOUT_LIMIT)
            for _ in range(n_units):
                for ft in call.fan_out:
                    if idx >= len(results):
                        break
                    fr = results[idx]
                    fseq = trace.record_tool(fr)
                    idx += 1
                    fanned.append((fr.args.get("unit"), ft, fr, fseq))
        steps.append(Step(call, res, seq, fanned))
    return steps


def answer_query_langgraph(
    cfg: PipelineConfig, query: str, planner_kind: str = "rule",
    out_dir: Path | str | None = None,
) -> dict:
    """Ask path on the LangGraph engine — identical grounded answer to native.

    Planning + composition are the shared rule planner and
    :func:`src.agent.query._compose`; only the tool *execution* runs through the
    LangGraph ToolNode loop. Returns the same dict shape as
    :func:`src.agent.query.answer_query`, plus ``engine="langgraph"``.
    """
    reg = Registry(cfg)
    planner = make_planner(planner_kind)
    plan = planner.plan(query)  # raises PlannerNotConfigured for the llm stub

    run_id = "ask_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trace = Trace(run_id=run_id, kind="query", planner=planner_kind)

    graph = build_ask_graph(reg)
    final = graph.invoke({
        "query": query, "plan": _plan_to_dicts(plan), "phase": "start",
        "messages": [], "answer": None, "citations": [], "trace_ref": None,
    })
    results = _ordered_tool_results(final["messages"])
    steps = _steps_from_results(plan, results, trace)
    answer = _compose(cfg, query, steps, trace)
    answer["engine"] = "langgraph"
    trace.set_answer(answer)
    path = trace.write(out_dir or cfg.path("reports"))
    return {
        "answer": answer, "trace": trace, "trace_path": str(path),
        "grounded": trace.all_claims_grounded(), "engine": "langgraph",
    }
