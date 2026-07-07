# LangGraph mapping — where a framework fits, and where it doesn't

This project uses **LangGraph narrowly, on the deterministic tool-calling (ask)
path only** (`src/agent/graph.py`, `--engine langgraph` on `ask`). The pipeline
supervisor and its human-in-the-loop gates stay **hand-written** in
`src/agent/autopilot.py`. This note explains that split honestly: what LangGraph
primitives *would* abstract if we pointed them at the pipeline, and why the
governance stays hand-rolled.

> **Claim discipline.** What is true here: *"a LangGraph `StateGraph` + `ToolNode`
> on the tool-calling path, driven deterministically; the LLM planner is the
> documented future occupant of the same loop."* What is **not** claimed and
> would be false: that LangGraph orchestrates the pipeline, that a checkpointer
> manages pipeline state, that there is an "LLM agent" or any model in the loop.
> No model is called anywhere in this repo today.

## Pipeline supervisor: what a framework would abstract vs. what we keep

| Native mechanism (kept hand-written) | LangGraph primitive that *could* abstract it | Why we keep it hand-rolled |
|---|---|---|
| 10-stage DAG walk (`s01…s10`) | `StateGraph` nodes + edges | The DAG is fixed and tiny; a hand-written loop is fully auditable and adds no dependency to the pipeline. |
| **Resume = provenance hashes** (unchanged stages skip; same inputs+seed → byte-identical artifacts) | in-memory `checkpointer` snapshots of graph state | Provenance is *durable content state on disk*, not in-process graph state. It survives crashes and new processes and gives **byte determinism** a checkpointer cannot: a checkpointer would manage *graph* state, not the artifact hashes the pipeline actually resumes from. |
| **Decision gates = files** (`autopilot_inbox/pending/…`, answered files) | `interrupt()` + `Command(resume=…)` | File-based cards are **crash-safe and replayable across processes**: a human can answer a card hours later, from another process, and the run resumes. `interrupt` pauses *within one process/checkpointer* — it does not, by itself, give the cross-process, file-durable HITL contract the demo needs. |
| Per-stage reliability gates (leakage canary, champion-beats-floor, governance, recompute) | conditional edges | The gate *logic* is the governance and must live in the reviewed native layer regardless of engine; edges would only route it. |
| HALT on violation | edge → `END` | Same — routing is trivial; the decision is the point. |

Short version: the pipeline's two hardest guarantees — **byte-identical
determinism** and **crash-safe, cross-process human-in-the-loop resume** — come
from provenance hashes and inbox files, which are durable disk state. An
in-process graph checkpointer solves a different problem (pausing one running
graph), so pointing it at the pipeline would replace a stronger guarantee with a
weaker one. The framework earns its place where it genuinely helps — the
tool-calling loop — and stays out of the governance.

## Where LangGraph *does* fit: the ask (tool-calling) path

The ask path is a natural, honest fit for a `StateGraph` + `ToolNode`:

```
   "which engines           ┌─────────── plan (RuleBasedPlanner, deterministic) ──────────┐
    need inspection?"        ▼                                                             │
        │            ┌──> agent ──has tool_calls?──▶ ToolNode ────────────────────────────┘
        └── query ──▶│      │  yes  (AIMessage.tool_calls)     (executes the 7 registry
                     │      │                                   tools; reg.call validation
                     │      │  no                               preserved)
                     │      ▼
                     └──> composer ── src.agent.query._compose ──▶ grounded, cited,
                                                                   bilingual answer + trace
```

* **Typed state** — `AskState` (`TypedDict`) carries the `query`, the standard
  `messages` channel that transports **tool_calls** (`AIMessage`) and
  **tool_results** (`ToolMessage`), and the composer's `answer` / `citations` /
  `trace_ref`.
* **Deterministic planner node** — the same `RuleBasedPlanner` emits standard
  `tool_calls`. Round one runs the plan; a fan-out plan (e.g. inspection) runs a
  second round of `get_prediction` per flagged unit; then it stops.
* **ToolNode** wraps the *same* `Registry`, so the seven tools and their input
  validation (unknown tool/arg, bad type, out-of-vocab choice → `ToolError`) are
  exactly the native ones.
* **Composer** reuses `src.agent.query._compose`, so the grounded bilingual
  answer, its per-claim citations, and the trace schema are **byte-identical** to
  `--engine native`. Equivalence is enforced by tests.

**Future occupant (documented, not built):** a real LLM planner would emit
`tool_calls` into this *same* `ToolNode` loop unchanged — the loop is
model-agnostic by construction. Swapping the deterministic planner node for a
model later requires no change to the tools, the validation, or the grounding.
Today the loop runs key-free with the rule planner; no model is invoked.

## Resume, in one honest sentence

Because the pipeline's durable state is **provenance hashes on disk** and its
decisions are **inbox files**, re-running the supervisor after answering a card
resumes from that checkpoint (earlier stages skip, the answered gate resolves)
rather than restarting at stage one — a crash-safe, cross-process contract that a
hand-written layer expresses directly and an in-process graph checkpointer would
not; so LangGraph is deliberately kept off that path and used only on the
tool-calling loop above.

## Try it

```bash
.venv/bin/pip install -r requirements-agent.txt      # dev-only; base stays lean
.venv/bin/python -m src.agent ask "which engines need inspection?" --engine langgraph
.venv/bin/python -m src.agent ask "diagnose unit 81" --engine langgraph
```

The answer matches `--engine native` exactly; only the tool-driving mechanism
differs. There is no `--engine` flag on `run` — the pipeline supervisor is native
only.
