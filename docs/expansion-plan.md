# Expansion Plan v2 — Explicit Pipeline + Autopilot Agent with HITL Decision Gates

> v1 planned 2026-07-06 by a 5-role team (Architect/DS/MLE/DA/PA). **v2 same
> day**: Lu refined the agent vision — "the agent does ALL the heavy lifting
> (ingest→model→predict→summarize) automatically and visibly; the human faces
> only condensed decision points with the strongest signals extracted" — and a
> 4-role agentic-specialist team (Agentic Architect, HITL Decision-UX,
> Agent Reliability, Agentic PM) corrected and completed that framing.
> **Status: awaiting Lu's approval — no implementation yet.** Scope: ~2–2.5
> days total. All truthfulness guardrails carry over.

## v2 — The agentic upgrade (supersedes v1's agent layer)

**Lu's framing, completed by the specialists.** What he described is a
**deterministic pipeline supervisor with HITL approval gates** (never "an
autonomous/LLM agent"): planner-executor over the fixed 10-stage DAG +
critic/summarizer emitting decision cards + observable execution. The three
things his framing was missing, now designed in:

1. **Bounded-autonomy contract** (testable): the agent may NEVER mark an
   engine safe, suppress a high-risk finding, alter thresholds/cap, sign off
   a report, or pick a champion on a calibration tie. Each becomes a
   governance check.
2. **Failure IS a decision point**: gate trips escalate as cards, not silent
   aborts. Leakage/grounding/safety violations auto-HALT.
3. **Checkpoint/resume**: provenance hashes are the state; answering a gate
   resumes from the checkpoint, never from s01. Deterministic: same inputs +
   seed → byte-identical decisions and trace.

### Two entry modes, one registry
- **Autopilot（自动巡检）** — flagship: `agent run --all` walks
  s01→s10 with a per-stage state machine (EXECUTE → VALIDATE gate → pass /
  RAISE card / HALT), streaming an append-only NDJSON journal
  (`reports/autopilot_journal.jsonl`) the app renders live. `--autonomy
  gated|auto|dry-run`, default gated; triage + sign-off cards never auto-pass.
- **Query（问答）** — v1's rule planner ("diagnose unit 81"), same tools,
  same trace schema.
- **Teaching mode（教学模式）** — the existing wizard, renamed and reused as
  the evidence viewer; decision cards deep-link into it. Top-level segmented
  toggle, Autopilot is the landing default.

### Decision Inbox（决策收件箱）
Card anatomy: one-sentence bilingual verdict → ≤3 signals (one is always the
uncertainty/optimism caveat) → evidence deep-link → safe-default actions →
consequence preview ("Export = draft work orders only, never commands").
**Max 5 actionable cards per run**; everything else batches into the digest.
The five cards for this fleet (real content, bilingual, from HITL-UX):
🔴 P1 "3 engines need inspection first — 39/57/81 (≤10 cycles)";
🟠 P2 "15 more high-risk — schedule this cycle"; 🟡 "1 newly escalated —
unit 24 fell 45→28"; 🟡 "7 engines can't be trusted yet — too little
history" (default action: Mark as unknown); 🟢 "75 healthy" (collapsed).
Done-state sentence: "Agent scored 100 engines, flagged 18, prepared evidence
per unit. **You have 4 decisions.** 75 healthy auto-cleared."

### Watch-it-work view
Narrative progress lines, not logs ("② Cleaning: dropped 7 flat sensors — 14
carry wear signal"), detail behind expanders; file-based (journal poll ~1s),
no websockets. Human actions write `reports/autopilot_inbox/<card_id>.json`;
the supervisor polls to resume. Crash-safe, replayable.

### Reliability layer (per-stage gates + Section D eval)
Stage gate table (ingest schema → HALT; leakage canary → HALT; champion
worse-than-Ridge-floor → HALT, marginal → CARD; degenerate risk distribution
→ CARD; diagnosis governance violation → HALT; etc.). New eval Section D:
recommendation→trace-step citation, card-signal reproducible from artifact
hash, no unrecorded stage skips, gate outcomes logged, two-run determinism,
no orphan claims. ~9 new tests; suite target ≈40. Gate thresholds live in
config and are hashed into the trace (anti-silent-weakening meta-test).

### Naming & claims (enforced)
Safe: "deterministic pipeline agent/supervisor with human-in-the-loop
decision gates", "automates the analysis; the human owns the decision",
"step-level execution trace for auditability". Banned: "fully autonomous",
"autonomous agent", "self-healing", "no human needed", "real-time" (say
"live step-by-step"), "LLM agent".

### Flagship demo moment (optimize for this one)
Run autopilot → watch it walk 10 stages live → it hands back a ~4-card
decision inbox with evidence + citations + recommended actions → approve one
card → see the grounded report. "The agent did the heavy lifting and can
prove every claim; you made the decisions."

---

# v1 base plan (pipeline + science + tools — still current except where v2 supersedes)

## What Lu asked for

Explicit stages — data ingestion, EDA, preprocessing, feature engineering,
modeling, model selection, prediction — plus an **agent** layer, with
everything clearly explained (bilingual, lay-readable).

## Design principles (unanimous across roles)

1. **Wrap, never rewrite.** Existing `src/{data,features,models,diagnostics,
   rag,app}` and all 16 green tests stay untouched. New `src/pipeline/` and
   `src/agent/` are thin layers over the same functions — one source of truth,
   no drift.
2. **Deterministic and key-free by default.** The agent is a **rule-based
   orchestrator** (never marketed as an "LLM agent" or "autonomous agent" —
   reproducible + auditable is the selling point). An LLM planner exists only
   as a flagged interface stub (subscription-CLI adapter shape, default off).
3. **"Clearly explained" enforced by code.** Every stage declares a
   `StageSpec(what, why, inputs, outputs, assumptions)`; the runner
   auto-generates `reports/pipeline_manifest.md` (spec + observed row counts /
   sizes / timings + DAG); a test fails if any spec field is empty. Bilingual
   plain-language stage explanations (DA's `STAGE_EXPLAIN` dict) feed both the
   manifest and the app.

## Lead decisions on flagged disagreements

| Question | Decision | Rationale |
|---|---|---|
| Config format | `config/pipeline.yaml` + PyYAML in base deps | Hiring-manager-legible; one small ubiquitous dep |
| MCP server | **Stretch, sequenced last** (Architect said IN; MLE/PA said only-if-cheap) | PA rule: MCP has value only if it truly runs; registry-first makes it a ~2–4h adapter later |
| Uncertainty display | CV-residual "typical miss ±N cycles" phrasing (canonical); RF per-tree spread as stretch band | One honest number beats two competing ones |
| Fleet triage table | **In scope** | Cheap (reads existing CSV), most decision-relevant lay view, powers the agent's flagship query |
| App hardcode drift (RUL_CAP=125 at `streamlit_app.py:37` + zh text) | Fix in Phase D: app reads cap + typical-miss from `reports/metrics_model.json` | DA's single-source-of-truth rule |

## Pipeline shape (Architect + MLE)

```
src/pipeline/            # NEW, thin
  specs.py stages.py context.py runner.py provenance.py
config/pipeline.yaml     # paths, seed=42, RUL_CAP, windows, model grid, risk thresholds
CLI: python -m src.pipeline run --all | --stage model | --from eda
```

Stages: `s01 ingest → s02 eda → s03 preprocess → s04 features → s05 model →
s06 select → s07 predict → s08 evidence → s09 diagnose → s10 eval`.
File contracts under `data/processed/` + `reports/` (each stage independently
runnable and reusable as an agent tool). Artifacts get `_provenance` stamps
(stage, params, input hashes, seed, git sha, timestamp); unchanged inputs →
skip, `--force` reruns. Only s02 (EDA) and s06 (selection) are genuinely new
logic; the rest are ~20-line wrappers.

## Science plan (DS)

- **EDA** (`notebooks/01_eda.ipynb` + `reports/eda_summary.md`): sensor-vs-RUL
  monotonicity ranking (names the degradation carriers), flat-sensor drop list
  with evidence, unit-lifetime distribution (why cap=125, why short units are
  hard), op-settings collapse (why no condition normalization on FD001).
- **Preprocessing**: per-sensor z-score fit on train only; drop near-constant
  sensors; leakage guards (no future-window info, cap on target only, stats
  fit inside CV folds); **GroupKFold by unit_id** — the single most important
  guard.
- **Features**: keep rolling mean/std; add rolling slope, delta-from-own-
  baseline, cycles-elapsed; EWMA only if it beats boxcar in CV. NOT adding
  FFT/spectral, cross-sensor interactions, deep sequence features.
- **Model selection**: Ridge (floor) + RF (champion-to-beat) + HistGBM
  (challenger), optional small MLP; identical GroupKFold(5) folds; criteria =
  (1) grouped-CV RMSE, (2) calibration near end-of-life (RMSE on true-RUL<50
  rows + optimistic-error fraction — the operational risk), (3) simplicity
  tiebreak. Artifact: `reports/model_selection.md` comparison table + explicit
  champion rationale. Champion swaps behind the existing prediction interface;
  evidence/RAG layers unchanged (permutation importances mapped if HistGBM
  wins).
- **Prediction**: batch + single-unit interfaces; point estimate + risk band +
  residual-based error bar + explicit optimistic-error flag.
- Always report capped AND uncapped metrics.

## Agent layer (MLE + Architect, PA naming rules)

- `src/agent/registry.py`: ~7 hand-written typed tools wrapping existing
  functions — `run_stage`, `get_prediction(unit)`, `get_evidence(unit)`,
  `retrieve(q,k)`, `diagnose(unit)`, `list_units_by_risk(band)`,
  `report(unit)`. No plugin systems, no schema DSLs.
- `planner.py`: rule-based intent → tool-call plan ("diagnose unit 81",
  "which engines need inspection?" → predict + list_units_by_risk(high);
  "this week" documented as a threshold mapping, no real calendar).
- `orchestrator.py` + `trace.py`: executes the plan; every call logged to
  `reports/agent_trace_<ts>.json`. **Grounding hard constraint**: final
  answers are composed only from tool outputs, every claim carries a trace id
  / KB citation (reuses assistant citations).
- `Planner` protocol with `--planner rule|llm`; LLM adapter = interface stub
  (subscription CLI shape, no API keys, default off, clear error if invoked
  unconfigured).
- **Stretch**: `mcp_server.py` stdio (tools/list + tools/call over the same
  registry; `mcp` dep isolated in `requirements-agent.txt`).

## Explanation & app layer (DA)

- `STAGE_EXPLAIN` EN/中文 dict (plain-language what/why per stage; approved
  register, e.g. model selection: "比了几个模型后选随机森林，不是因为分最高，
  而是它够准（误差约 12 个周期）又能说清每次判断靠哪些传感器").
- 5-chart fleet EDA story (lifespan histogram, degradation fingerprint,
  which-dials-carry-signal, fleet spaghetti aligned to failure, fleet risk
  snapshot) in the notebook AND a new app "Explore the data / 看看数据"
  section.
- **Fleet triage table**: all 100 test engines sortable by risk/RUL.
- Lay model-selection table: Model · How it works · Typical miss (±cycles) ·
  Explains itself? · Why (not) picked. Jargon swaps: RMSE→"typical miss",
  R²→"share of the wear pattern captured".
- Honest uncertainty phrasing: "estimate typically off by ±12 cycles, more
  near end-of-life, and when wrong it tends to guess too healthy."
- Metrics single source of truth: `reports/metrics_model.json` +
  `reports/evaluation_summary.md`; README/app/docs quote, never restate.

## Implementation phases (v2-revised, after approval)

| Phase | Content | Effort |
|---|---|---|
| A | Pipeline wrappers + config + provenance + manifest + EDA/predict stages **+ step-event journal writes** | ~0.5d |
| B | Model-selection bake-off + selection report + champion contract | ~0.3d |
| C | Agent registry + rule planner + trace → **autopilot supervisor** (state machine, gates, cards, checkpoint/resume) + grounding tests | ~0.5d |
| D | App: **Autopilot progress page (live journal render) + Decision Inbox** + Teaching-mode rename + fleet EDA + per-stage captions + metrics-from-JSON fix | ~0.6–0.8d (biggest) |
| E | Docs/README/resume-bullets + eval Section D (trace/condensation governance) + final audit refresh | ~0.3d |
| Stretch | MCP stdio server (over the same registry) | last |

Total ≈2–2.5 days. The button-wizard rework has landed, so Phase D is
unblocked once A–C exist.

## Testing & reproducibility (MLE)

~15 new tests (stage contracts, leakage guards, provenance/idempotency,
planner plans, trace grounding) → target ≈31, existing 16 untouched.
Seeds in config; `requirements.lock.txt` pinned; `make demo` = pipeline +
eval + one agent query; CI-ready but Actions deferred.

## Success criteria (v2 — done = all six)

- [ ] `agent run --all` reproducibly drives all 10 stages, streaming a
      step-event journal the app renders live.
- [ ] Run ends in a Decision Inbox: every card carries evidence + KB citation
      + recommended action; triage and sign-off cards never auto-pass.
- [ ] **Condensation (flagship): a first-time user reaches the 3-4 key
      decisions in <2 minutes without opening any chart.**
- [ ] Every agent recommendation traces to a journal step + evidence/KB
      citation; grounding + determinism tests pass (~40 total, 16 legacy
      untouched).
- [ ] Both entry modes work off one registry: autopilot + query
      ("diagnose unit 81"); wizard preserved as Teaching mode / evidence
      viewer.
- [ ] Truthfulness audit passes; "deterministic pipeline agent + HITL
      decision gates" naming enforced; `model_selection.md` compares ≥2
      candidates with explicit rationale.

## Explicit CUT list

Deep learning/LSTM; FD002–FD004; cloud deploy; dense embeddings/vector DB;
live drift monitoring; real LLM in the loop by default; orchestration
frameworks (Airflow/Prefect/LangChain); async/HTTP/auth infra beyond the
single stdio MCP stretch; real-calendar scheduling.
