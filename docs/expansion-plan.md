# Expansion Plan — Explicit Pipeline + Deterministic Agent Layer

> Planned 2026-07-06 by a 5-role team (Architect, Data Scientist, ML Engineer,
> Data Analyst, Product Analyst) + lead synthesis. **Status: awaiting Lu's
> approval — no implementation yet.** Scope: ~1.5–2 days on top of the working
> prototype. All build-spec truthfulness guardrails carry over unchanged.

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

## Implementation phases (after approval)

| Phase | Content | Owner roles |
|---|---|---|
| A | Pipeline wrappers + config + provenance + manifest + EDA & predict stages | MLE + Architect |
| B | Model-selection bake-off + selection report + champion contract | DS |
| C | Agent registry + planner + orchestrator + trace + grounding tests | MLE |
| D | App: pipeline overview page, fleet EDA, triage table, per-stage captions, metrics-from-JSON fix | DA + app engineer (after the in-flight wizard rework lands) |
| E | Docs/README/resume-bullets/eval extension (trace governance checks) + final audit refresh | Writer + lead |
| Stretch | MCP stdio server | MLE |

Note: Phase D depends on the button-wizard rework currently in flight — that
lands first.

## Testing & reproducibility (MLE)

~15 new tests (stage contracts, leakage guards, provenance/idempotency,
planner plans, trace grounding) → target ≈31, existing 16 untouched.
Seeds in config; `requirements.lock.txt` pinned; `make demo` = pipeline +
eval + one agent query; CI-ready but Actions deferred.

## Success criteria (PA — done = all five)

- [ ] One command runs ingestion→prediction reproducibly.
- [ ] Agent answers "which engines need inspection?" with every claim backed
      by a trace step citing evidence + KB.
- [ ] `model_selection.md` compares ≥2 candidates with explicit rationale.
- [ ] README + manifest explain each stage plainly; first-time reader runs it
      in <5 min.
- [ ] Truthfulness audit passes; "deterministic orchestrator" naming enforced;
      resume bullets updated with produced facts only.

## Explicit CUT list

Deep learning/LSTM; FD002–FD004; cloud deploy; dense embeddings/vector DB;
live drift monitoring; real LLM in the loop by default; orchestration
frameworks (Airflow/Prefect/LangChain); async/HTTP/auth infra beyond the
single stdio MCP stretch; real-calendar scheduling.
