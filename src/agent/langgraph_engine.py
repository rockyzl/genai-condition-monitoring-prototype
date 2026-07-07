"""LangGraph engine — a second, framework-based lane for the same governed agent.

This is a *parallel engine*, not a rewrite. It implements the exact same
deterministic pipeline supervisor and grounded query answerer as the hand-written
:mod:`src.agent.autopilot` / :mod:`src.agent.query`, but expresses the
orchestration as LangGraph graphs. The point is to show the **same governance**
running on two engines — and to make LangGraph's two headline mechanics
first-class:

* **Typed state control** — :class:`AutopilotState` / :class:`AskState` are
  ``TypedDict`` schemas that flow through the graph, with documented reducers on
  the channels that accumulate.
* **Tool calling** — the ask path is a real LangGraph tool-calling loop: a
  deterministic planner node emits standard ``tool_calls``, a prebuilt
  :class:`~langgraph.prebuilt.ToolNode` executes the seven registry tools (their
  input validation preserved), and a composer node builds the grounded answer.

**Zero logic duplication.** Graph nodes call the *same* registry tools, the
*same* gate-evaluation methods (:class:`src.agent.autopilot.Autopilot` is reused
as a governance toolkit), the *same* card builders, and the *same* answer
composer (:func:`src.agent.query._compose`). Only the wiring differs. Both
engines write the identical trace/journal/inbox/state artifacts, so eval
Section D governs this engine unmodified, and parity is testable
(native vs langgraph → identical decisions and cards on the same inputs+seed).

**Framing:** this is *not* an LLM agent. LangGraph here runs deterministic nodes;
no model is called. ``--planner llm`` would emit tool_calls into the same
ToolNode unchanged — the loop is model-agnostic by construction.

Why the native engine remains the default, and what we deliberately did *not*
hand to the framework, is spelled out in ``docs/langgraph-engine.md``: the
pipeline's byte-identical determinism (provenance hashes) and its crash-safe,
cross-process human-in-the-loop resume (inbox files) are durable on-disk state,
which the hand-written layer expresses directly and an in-process graph
checkpointer would only weaken. The framework earns its place on the tool-calling
loop; the governance stays in the shared, reviewed native layer.

Requires ``langgraph`` (dev-only; see ``requirements-agent.txt``). The base
install and the native engine never import this module.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import create_model

from src.agent import cards as cards_mod
from src.agent.autopilot import (
    CARD,
    EVAL_SUMMARY_REL,
    HALT,
    METRICS_REL,
    PASS,
    PREDS_REL,
    SKIP,
    AgentGateConfig,
    Autopilot,
    RunReport,
)
from src.agent.planner import ToolCall, make_planner
from src.agent.query import FANOUT_LIMIT, Step, _compose
from src.agent.registry import Registry, ToolResult, digest
from src.agent.trace import Trace
from src.pipeline.config import PipelineConfig
from src.pipeline.context import PipelineContext
from src.pipeline.journal import Journal
from src.pipeline.specs import STAGE_ORDER
from src.pipeline.stages import STAGE_FUNCS

# =============================================================================
# Typed state (Lu emphasis #1)
# =============================================================================
class AutopilotState(TypedDict):
    """State flowing through the autopilot graph.

    Reducer policy (documented per the plan):

    * ``stage_results``/``gate_outcomes``/``decisions``/``cards_resolved``/
      ``would_raise`` accumulate across stage nodes → ``operator.add`` reducer.
    * ``run_id``/``autonomy``/``thresholds_hash``/``seed`` are set once at START
      and never change → default last-write-wins channel.
    * ``cursor``/``halted``/``halt`` are control signals a node overwrites →
      default last-write-wins channel.
    """

    run_id: str
    autonomy: str
    thresholds_hash: str
    config_hash: str
    seed: int
    stage_results: Annotated[list, operator.add]
    gate_outcomes: Annotated[list, operator.add]
    decisions: Annotated[list, operator.add]
    cards_resolved: Annotated[list, operator.add]
    would_raise: Annotated[list, operator.add]
    cursor: str
    halted: bool
    halt: dict | None


class AskState(TypedDict):
    """State for the tool-calling ask graph.

    ``messages`` is the standard tool-calling channel (``add_messages`` reducer):
    the planner appends an ``AIMessage`` carrying ``tool_calls``, the ToolNode
    appends ``ToolMessage`` results. ``plan``/``phase`` drive the deterministic
    loop (initial plan → optional fan-out → done).
    """

    question: str
    plan: list
    phase: str
    messages: Annotated[list, add_messages]


# =============================================================================
# Tool calling (Lu emphasis #2): registry tools → LangGraph StructuredTools
# =============================================================================
def _pydantic_args_model(reg: Registry, name: str):
    """Build a pydantic args schema for a registry tool from its ArgSpec list."""
    spec = reg.tools[name]
    fields: dict[str, tuple] = {}
    for a in spec.args:
        default = ... if a.required else a.default
        fields[a.name] = (a.type, default)
    return create_model(f"{name}_args", **fields)


def make_lc_tools(reg: Registry) -> list[StructuredTool]:
    """Wrap all seven registry tools as LangGraph-executable StructuredTools.

    Each wrapper calls ``reg.call`` (so schema validation and the ToolError
    boundary are preserved) and returns the JSON-encoded :class:`ToolResult` the
    ToolNode carries as a ``ToolMessage``.
    """
    import json

    tools: list[StructuredTool] = []
    for name in reg.names():
        spec = reg.tools[name]
        args_model = _pydantic_args_model(reg, name)

        def _make(tool_name: str):
            def _fn(**kwargs) -> str:
                res = reg.call(tool_name, kwargs)  # validation preserved here
                return json.dumps(
                    {
                        "tool": res.tool,
                        "args": res.args,
                        "ok": res.ok,
                        "output": res.output,
                        "error": res.error,
                    }
                )

            return _fn

        tools.append(
            StructuredTool.from_function(
                _make(name), name=name, description=spec.description,
                args_schema=args_model,
            )
        )
    return tools


def _plan_to_dicts(plan: list[ToolCall]) -> list[dict]:
    return [{"tool": c.tool, "args": c.args, "fan_out": list(c.fan_out)} for c in plan]


def build_ask_graph(reg: Registry):
    """Compile the deterministic tool-calling loop: planner → tools → composer.

    The ``agent`` node is a deterministic controller (no LLM): it emits the
    plan's tool_calls, then — for a fan-out plan — a second round of
    ``get_prediction`` calls per returned unit, then stops. A ``--planner llm``
    would emit into this same ToolNode unchanged.
    """
    tools = make_lc_tools(reg)
    tool_node = ToolNode(tools)

    def agent(state: AskState) -> dict:
        plan = state["plan"]
        phase = state.get("phase", "start")
        if phase == "start":
            calls = [
                {"name": c["tool"], "args": c["args"], "id": f"call_{i}"}
                for i, c in enumerate(plan)
            ]
            needs_fanout = any(c.get("fan_out") for c in plan)
            next_phase = "fanout" if needs_fanout else "final"
            if not calls:  # unmapped plan → nothing to execute
                return {"messages": [AIMessage(content="")], "phase": "final"}
            return {"messages": [AIMessage(content="", tool_calls=calls)],
                    "phase": next_phase}
        if phase == "fanout":
            # find the list_units_by_risk result and fan get_prediction over it
            fan_call = next((c for c in plan if c.get("fan_out")), None)
            units = _last_tool_output(state["messages"], fan_call["tool"]).get("units", [])
            calls = []
            for j, u in enumerate(units[:FANOUT_LIMIT]):
                for ft in fan_call["fan_out"]:
                    calls.append({"name": ft, "args": {"unit": u["unit_id"]},
                                  "id": f"fan_{ft}_{j}"})
            if not calls:
                return {"messages": [AIMessage(content="")], "phase": "final"}
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
    import json

    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == tool_name:
            payload = json.loads(m.content)
            return payload.get("output") or {}
    return {}


def _ordered_tool_results(messages: list) -> list[ToolResult]:
    """Reconstruct ToolResults in tool_call order from the graph's messages.

    Pairs each ``AIMessage.tool_calls`` entry (name+args+id) with its
    ``ToolMessage`` (by id), preserving planned order so the reconstructed steps —
    and therefore the composed answer — match the native engine exactly.
    """
    import json

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
            results.append(
                ToolResult(
                    tool=tc["name"], args=tc["args"], ok=payload.get("ok", False),
                    output=output, error=payload.get("error"),
                    digest=digest(output) if payload.get("ok") else "",
                )
            )
    return results


def _steps_from_results(
    plan: list[ToolCall], results: list[ToolResult], trace: Trace
) -> list[Step]:
    """Rebuild the native :class:`Step` structure from executed results.

    Mirrors the native answer loop (plan call, then its fan-out) so seqs recorded
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
    :func:`src.agent.query.answer_query`.
    """
    reg = Registry(cfg)
    planner = make_planner(planner_kind)
    plan = planner.plan(query)  # raises PlannerNotConfigured for the llm stub

    run_id = "ask_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trace = Trace(run_id=run_id, kind="query", planner=planner_kind)

    graph = build_ask_graph(reg)
    final = graph.invoke(
        {"question": query, "plan": _plan_to_dicts(plan), "phase": "start",
         "messages": []}
    )
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


# =============================================================================
# Autopilot graph
# =============================================================================
class LangGraphAutopilot:
    """The autopilot supervisor expressed as a LangGraph StateGraph.

    Reuses a native :class:`Autopilot` as a governance toolkit: stage execution
    on the shared pipeline context, the ``_gate_sNN`` evaluations, the card
    builders, the autonomy policy, and the trace/journal/inbox/state writers are
    all the native ones. The graph supplies typed state, conditional edges, an
    ``interrupt`` at each blocking decision card, and a checkpointer.
    """

    def __init__(
        self, cfg: PipelineConfig, autonomy: str = "gated",
        yes_safe_defaults: bool = False, force: bool = False,
        out_dir: Path | str | None = None, gcfg: AgentGateConfig | None = None,
    ):
        self.pilot = Autopilot(cfg, autonomy, yes_safe_defaults, force, out_dir, gcfg)
        self.cfg = cfg
        self.autonomy = autonomy
        self.yes_safe_defaults = yes_safe_defaults

    # --- graph construction --------------------------------------------------
    def _make_stage_node(self, stage: str, ctx, journal: Journal, trace: Trace):
        pilot = self.pilot

        def node(state: AutopilotState) -> dict:
            result = STAGE_FUNCS[stage](ctx)  # EXECUTE (shared ctx, provenance-skip)
            pilot._record_stage_exec(trace, result)
            outcomes = pilot._validate(stage, ctx, result, journal, trace)

            updates: dict = {
                "cursor": stage,
                "stage_results": [{"stage": stage, "skipped": result.skipped,
                                   "seconds": round(result.seconds, 4)}],
                "decisions": [], "gate_outcomes": [], "cards_resolved": [],
                "would_raise": [],
            }
            for oc in outcomes:
                trace.record_gate(oc.stage, oc.gate, oc.disposition,
                                  oc.card.id if oc.card else None, oc.detail)
                kind = oc.card.kind if oc.card else None
                updates["decisions"].append([oc.stage, oc.gate, oc.disposition, kind])
                updates["gate_outcomes"].append(
                    {"stage": oc.stage, "gate": oc.gate, "disposition": oc.disposition,
                     "card_id": oc.card.id if oc.card else None, "detail": oc.detail}
                )

                if oc.disposition in (PASS, SKIP):
                    journal.stage_progress(
                        stage, f"gate {oc.gate}: {oc.disposition.lower()}")
                    continue
                if oc.disposition == HALT:
                    journal.halt(stage, oc.gate, oc.detail)
                    updates["halted"] = True
                    updates["halt"] = {"stage": stage, "gate": oc.gate,
                                       "reason": oc.detail}
                    return updates

                # disposition == CARD — apply the shared autonomy policy
                card = oc.card
                trace.add_card(card.to_dict())
                if self.autonomy == "dry-run":
                    updates["would_raise"].append(
                        {"card_id": card.id, "kind": card.kind, "priority": card.priority,
                         "verdict_en": card.verdict_en,
                         "would_block": not card.auto_passable()})
                    journal.stage_progress(
                        stage, f"[dry-run] would raise {card.kind} ({card.id})")
                    continue

                answer = pilot._read_answer(card)  # cross-process answered file
                source = "answered"
                if answer is None and self.autonomy == "auto" and card.auto_passable():
                    answer, source = card.safe_action().id, "auto"
                elif (answer is None and self.autonomy == "gated"
                      and self.yes_safe_defaults and card.safe_default_applicable()):
                    answer, source = card.safe_action().id, "yes-safe-defaults"

                if answer is None:
                    # BLOCK: raise the card and interrupt the graph
                    if not (pilot.pending_dir / f"{card.id}.json").exists():
                        journal.gate_raised(card.id, card.kind, stage,
                                            card.payload_summary())
                    pilot._write_pending(card)
                    answer = interrupt(card.to_dict())  # pauses; resume → action id
                    source = "resume"
                else:
                    journal.gate_raised(card.id, card.kind, stage,
                                        card.payload_summary())

                journal.gate_resolved(card.id, answer, stage)
                (pilot.pending_dir / f"{card.id}.json").unlink(missing_ok=True)
                updates["cards_resolved"].append(
                    {"card_id": card.id, "kind": card.kind, "action": answer,
                     "source": source})
            return updates

        return node

    def _build_graph(self, ctx, journal: Journal, trace: Trace):
        g = StateGraph(AutopilotState)
        for stage in STAGE_ORDER:
            g.add_node(stage, self._make_stage_node(stage, ctx, journal, trace))
        g.add_edge(START, STAGE_ORDER[0])
        for i, stage in enumerate(STAGE_ORDER):
            nxt = STAGE_ORDER[i + 1] if i + 1 < len(STAGE_ORDER) else END

            def route(state: AutopilotState, _nxt=nxt) -> str:
                return END if state.get("halted") else _nxt

            g.add_conditional_edges(stage, route, {nxt: nxt, END: END})
        return g.compile(checkpointer=InMemorySaver())

    # --- run -----------------------------------------------------------------
    def prepare(self) -> dict:
        """Build the compiled graph + typed initial state for one run.

        Returns the app, thread config, initial :class:`AutopilotState`, and the
        shared ctx/journal/trace. ``run()`` uses it; tests drive the raw
        ``interrupt``/``Command(resume=...)`` loop through the returned ``app``.
        """
        pilot = self.pilot
        run_id = "auto_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        pilot.pending_dir.mkdir(parents=True, exist_ok=True)
        pilot.answered_dir.mkdir(parents=True, exist_ok=True)

        journal = Journal(pilot.journal_path, run_id)
        ctx = PipelineContext(cfg=self.cfg, journal=journal, run_id=run_id,
                              force=pilot.force)
        trace = Trace(run_id=run_id, kind="autopilot", autonomy=self.autonomy,
                      thresholds_hash=pilot.gcfg.hash())
        journal.run_started(STAGE_ORDER)

        app = self._build_graph(ctx, journal, trace)
        thread = {"configurable": {"thread_id": run_id}}
        init: AutopilotState = {
            "run_id": run_id, "autonomy": self.autonomy,
            "thresholds_hash": pilot.gcfg.hash(),
            "config_hash": digest(self.cfg.to_dict()), "seed": self.cfg.seed,
            "stage_results": [], "gate_outcomes": [], "decisions": [],
            "cards_resolved": [], "would_raise": [], "cursor": "",
            "halted": False, "halt": None,
        }
        return {"app": app, "thread": thread, "init": init, "run_id": run_id,
                "ctx": ctx, "journal": journal, "trace": trace}

    def run(self) -> RunReport:
        prep = self.prepare()
        app, thread, ctx = prep["app"], prep["thread"], prep["ctx"]
        journal, trace, run_id = prep["journal"], prep["trace"], prep["run_id"]
        final = app.invoke(prep["init"], thread)

        pending_card = None
        if "__interrupt__" in final:
            status = "awaiting_input"
            pending_card = final["__interrupt__"][0].value
        elif final.get("halted"):
            status = "halted"
        else:
            status = "done"

        return self._assemble_report(
            run_id, status, final, trace, ctx, journal, pending_card
        )

    def _assemble_report(
        self, run_id, status, final, trace, ctx, journal, pending_card
    ) -> RunReport:
        pilot = self.pilot
        report = RunReport(
            run_id=run_id, autonomy=self.autonomy, status=status,
            thresholds_hash=pilot.gcfg.hash(),
            journal_path=str(pilot.journal_path), state_path=str(pilot.state_path),
            inbox_pending_dir=str(pilot.pending_dir),
        )
        # decisions come from the trace's gate records (side-effect-recorded even
        # when a node interrupts) → byte-identical to the native engine's list.
        kind_by_card = {c["id"]: c["kind"] for c in trace.cards if "id" in c}
        report.decisions = [
            (g.stage, g.gate, g.disposition,
             kind_by_card.get(g.card_id) if g.card_id else None)
            for g in trace.gates
        ]
        report.stages = self._stages_from_trace(trace)
        report.cards_resolved = list(final.get("cards_resolved", []))
        report.would_raise = list(final.get("would_raise", []))
        report.halt = final.get("halt")
        completed = [s["stage"] for s in report.stages]

        if status == "awaiting_input" and pending_card is not None:
            report.cards_pending = [pending_card]
        if status == "done":
            self._attach_digest(report, trace)
            n_skipped = sum(1 for r in ctx.results if r.skipped)
            journal.run_done(stages_run=len(ctx.results) - n_skipped,
                             stages_skipped=n_skipped,
                             seconds=sum(r.seconds for r in ctx.results))

        cursor = report.cards_pending[0]["id"] if report.cards_pending else None
        pilot._save_state(report, completed, cursor if status == "awaiting_input"
                          else None, ctx)
        trace.write(pilot.out_dir)
        report.trace_path = str(pilot.out_dir / f"agent_trace_{run_id}.json")
        return report

    def _stages_from_trace(self, trace: Trace) -> list[dict]:
        """Reconstruct per-stage rows from the trace (run_stage tool calls + gates)."""
        rows: dict[str, dict] = {}
        order: list[str] = []
        for rec in trace.tool_calls:
            if rec.tool != "run_stage":
                continue
            prev = rec.output_preview or {}
            stage = prev.get("stage") or rec.args.get("stage")
            if stage not in rows:
                rows[stage] = {"stage": stage, "skipped": prev.get("skipped", False),
                               "seconds": rec.seconds, "gates": []}
                order.append(stage)
        for g in trace.gates:
            if g.stage in rows:
                rows[g.stage]["gates"].append(
                    {"gate": g.gate, "disposition": g.disposition,
                     "card_id": g.card_id, "detail": g.detail})
        return [rows[s] for s in order]

    def _attach_digest(self, report: RunReport, trace: Trace) -> None:
        m = self.pilot._read_json(METRICS_REL)
        preds = (self.cfg.root / PREDS_REL).resolve()
        if m and preds.exists():
            bands = m.get("risk_band_counts", {})
            n_flagged = int(bands.get("high", 0)) + int(bands.get("medium", 0))
            report.digest = cards_mod.build_healthy_digest(self.cfg, PREDS_REL, n_flagged)
            trace.add_card(report.digest)


def resume_langgraph(
    cfg: PipelineConfig, run_id: str, app, thread: dict, action: str
) -> Any:
    """In-process resume helper: feed an action back into a paused graph.

    (Exposed for tests/tools; the CLI resumes cross-process via answered-card
    files + provenance, exactly like the native engine.)
    """
    return app.invoke(Command(resume=action), thread)
