# LangGraph engine — the second lane over the same governance

The agent ships **two orchestration engines** behind one CLI:

* **native** (default) — the hand-written supervisor in `src/agent/autopilot.py`
  and the ask path in `src/agent/query.py`. No framework, fully auditable.
* **langgraph** (optional, `--engine langgraph`) — the *same* governed workflow
  expressed with [LangGraph](https://langchain-ai.github.io/langgraph/): a
  `StateGraph` with typed state, a `ToolNode` tool-calling loop, and
  `interrupt`-based decision gates.

They are provably equivalent. On the same inputs + seed, native and LangGraph
produce **identical decisions and cards** (parity test), and **both pass the same
autonomy-governance evaluation** (eval Section D) over identical trace / journal /
inbox / state artifacts. The framework demonstrates typed-state and tool-calling
patterns; it does **not** own the governance — that stays in the shared,
hand-written layer (registry tools, gate functions, card builders, grounded
composer), which both engines call. Zero logic is duplicated.

> **Not an LLM agent.** LangGraph here runs **deterministic** nodes; no model is
> called. `--planner llm` would emit tool_calls into the *same* `ToolNode`
> unchanged — the loop is model-agnostic by construction. The correct framing
> stays "a deterministic pipeline agent/supervisor with human-in-the-loop
> decision gates."

## Why the native engine remains the default (what we did *not* hand to the framework)

The native, hand-written engine stays the **default**, and LangGraph is added as
an optional lane — not a replacement — on purpose. Two of the pipeline's hardest
guarantees are *durable on-disk state*, and we deliberately keep them in the
hand-written layer rather than delegating them to the framework:

* **Byte-identical determinism comes from provenance hashes**, not from graph
  state. Unchanged inputs + seed skip and re-produce byte-identical artifacts; a
  StateGraph checkpointer would snapshot *graph* state, which is a weaker, in-
  process notion of "resume."
* **Crash-safe, cross-process human-in-the-loop resume comes from inbox files.**
  A reviewer can answer a decision card hours later, from another process, and
  the run resumes. `interrupt()` pauses *one running graph under one
  checkpointer* — a genuinely useful mechanic (we use it in-process), but not the
  cross-process, file-durable contract the demo needs.

So the checkpointer is used only for **in-run** pause/resume, while cross-process
resume stays anchored on provenance + answered-card files (the honest split
below). And the **gate logic itself** — leakage canary, champion-beats-floor,
governance, recompute — lives in the reviewed native layer regardless of engine;
LangGraph edges only route it. The framework earns its place on the tool-calling
loop; the governance does not move.

## Concept → LangGraph primitive (mapping table)

| Our concept (native) | LangGraph primitive (this engine) |
|---|---|
| Autopilot supervisor loop over `s01…s10` | `StateGraph` with one node per stage, linear edges |
| Per-stage `EXECUTE → VALIDATE → pass / card / halt` | stage node (execute + `Autopilot._gate_sNN`) + conditional edge |
| Stage execution (`run_stage` over the shared context) | stage node calls `STAGE_FUNCS[stage](ctx)`; recorded as a `run_stage` trace entry |
| Blocking decision card (triage, sign-off) | `interrupt(card)` inside the node; the resume value is the chosen action id |
| Bounded autonomy (`NEVER_AUTO_KINDS`, `NEVER_SAFE_DEFAULT_KINDS`) | node applies the **same** `card.auto_passable()` / `safe_default_applicable()` policy *before* interrupting |
| HALT (leakage / governance / schema) | node sets `halted=True` + `journal.halt(...)`; conditional edge routes to `END` |
| Decision-card LOG (healthy / dry-run) | node records to `would_raise` / digest; no interrupt |
| In-run checkpoint / resume | built-in `InMemorySaver` checkpointer + `Command(resume=action)` |
| Cross-process resume (CLI re-run) | answered-card files + provenance skip (checkpointer is not persisted — see below) |
| Typed run state | `AutopilotState` `TypedDict` with documented reducers |
| Rule planner → tool-call plan | `agent` node emits an `AIMessage` carrying standard `tool_calls` |
| The 7 registry tools | `make_lc_tools()` → `StructuredTool` (validation preserved) → `ToolNode` |
| Fan-out (`get_prediction` per flagged unit) | second `agent`→`tools` loop round, driven by the prior tool result |
| Grounded, cited answer | shared `src.agent.query._compose`; composer builds the same claims/trace |
| Execution trace + step journal | the **same** `Trace` + `Journal` writers → same on-disk schema |

## Typed state (Lu emphasis #1)

`AutopilotState` (`TypedDict`) flows through the autopilot graph. Reducer policy:

```python
class AutopilotState(TypedDict):
    # set once at START, never change (last-write-wins channel):
    run_id: str; autonomy: str; thresholds_hash: str; config_hash: str; seed: int
    # accumulate across the 10 stage nodes (operator.add reducer):
    stage_results:  Annotated[list, operator.add]
    gate_outcomes:  Annotated[list, operator.add]
    decisions:      Annotated[list, operator.add]
    cards_resolved: Annotated[list, operator.add]
    would_raise:    Annotated[list, operator.add]
    # control signals a node overwrites (last-write-wins channel):
    cursor: str; halted: bool; halt: dict | None
```

`AskState` carries the standard tool-calling channel
`messages: Annotated[list, add_messages]` (the planner appends an `AIMessage`
with `tool_calls`, the `ToolNode` appends `ToolMessage` results), plus `plan` and
`phase` to drive the deterministic loop.

## Tool calling (Lu emphasis #2) — the ask loop

```
                        ┌──────────── plan (rule planner) ────────────┐
   "which engines       ▼                                             │
    need inspection?" → agent ──has tool_calls?──▶ ToolNode ──────────┘
                          │  yes  (AIMessage.tool_calls)   (executes the 7
                          │                                 registry tools,
                          │  no                             validation kept)
                          ▼
                       composer ── src.agent.query._compose ──▶ grounded,
                                                                 cited answer
```

Round 1 emits the plan's calls (e.g. `list_units_by_risk(high)`); for a fan-out
plan, round 2 emits `get_prediction` per flagged unit; then the `agent` emits a
message with no `tool_calls` and the graph routes to the `composer`. Every tool
runs through `reg.call`, so the same input validation that rejects unknown
tools/args/bad-choices is preserved. The composer is the *shared* native
composer, so the answer and its grounded claims are byte-identical to `--engine
native`.

## Checkpoint / resume — the honest split

Resume works at two levels, and the engine uses both deliberately:

* **In-run (same process):** the compiled graph uses LangGraph's `InMemorySaver`.
  Hitting a decision card calls `interrupt(card)`, which pauses the graph; the
  caller resumes with `Command(resume=<action id>)` and the paused node returns
  that action. This is the mechanism the `interrupt`/`resume` tests exercise.
* **Cross-process (CLI re-run):** the in-memory checkpointer does **not** survive
  process exit, so cross-process resume is anchored — exactly like the native
  engine — on **provenance** (unchanged stages skip on re-run) plus
  **answered-card files** (`reports/autopilot_inbox/answered/<card_id>.json`). A
  stage node checks for an answer file *before* interrupting, so a re-run
  resolves the card and continues without pausing.

**Resume-safe in one sentence:** *Because provenance hashes are the durable state
and decision cards are files, re-running the LangGraph engine after answering a
card resumes from the checkpoint (earlier stages skip, the answered gate
resolves) rather than restarting at stage one — the same crash-safe, replayable
contract as the native engine.*

## Try it

```bash
.venv/bin/pip install -r requirements-agent.txt      # dev-only; base stays lean

.venv/bin/python -m src.agent run --all --engine langgraph --autonomy dry-run
.venv/bin/python -m src.agent run --all --engine langgraph            # gated
.venv/bin/python -m src.agent ask "which engines need inspection?" --engine langgraph
```

The output — decisions, cards, and the grounded answer — matches `--engine
native` exactly; the difference is only *how* the workflow is orchestrated.
