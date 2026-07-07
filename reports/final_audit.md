# Final Audit

> Independent R&D prototype on public NASA C-MAPSS data. Not affiliated with
> Caterpillar or any equipment manufacturer; no proprietary or field data; not
> production-ready. This audit is the honest, end-of-build accounting required
> by the build spec (Phase 10), refreshed after the v2 expansion.

## What Was Built

An end-to-end GenAI-assisted condition-monitoring prototype. The v1 core:

- Sensor time-series ingestion and feature prep on NASA C-MAPSS **FD001**.
- A Remaining Useful Life (RUL) **baseline model** with saved metrics and error
  analysis. *(Results: RMSE 17.2 / MAE 12.1 / R² 0.82 vs capped truth (cap=125);
  RMSE 18.2 / MAE 13.1 vs uncapped truth — see `reports/metrics_model.json`.)*
- A **diagnostic evidence layer** converting model outputs into structured JSON
  per unit/window (sensor summary, predicted RUL, risk band, top contributing
  signals, uncertainty note).
- A **TF-IDF RAG assistant** over a five-file local knowledge base
  (`docs/knowledge_base/`) producing cited, uncertainty-aware, maintenance-style
  summaries with a mandatory human-review warning.
- An **evaluation harness** (`src/eval/`). *(Results: retrieval hit@4 = 1.00
  (10/10, all rank 1); diagnostic governance checks 100/100 units pass all six
  checks, zero violations — see `reports/evaluation_summary.md`.)*

### Added in v2 (expansion phases A–D)

- **Phase A — deterministic 10-stage pipeline** (`src/pipeline/`): `s01 ingest →
  s02 eda → s03 preprocess → s04 features → s05 model → s06 select → s07 predict
  → s08 evidence → s09 diagnose → s10 eval`. Each stage declares a typed
  `StageSpec` (what/why/inputs/outputs/assumptions); artifacts carry provenance
  stamps so unchanged inputs skip; every step is written to an append-only NDJSON
  journal; the runner auto-generates `reports/pipeline_manifest.md` with
  bilingual stage explanations. Thin wrappers over the existing functions — one
  source of truth, no rewrite.
- **Phase B — model-selection bake-off** (`src/models/model_selection.py`,
  `reports/model_selection.md`): Ridge floor (**21.01 ± 1.38** CV-RMSE),
  RandomForest champion (**18.20 ± 0.52**, beats the floor by 2.81 cycles), and
  a HistGradientBoosting challenger (**18.59**) rejected for failing the
  clear-win bar — under identical unit-grouped 5-fold CV, judged on grouped-CV
  RMSE, end-of-life calibration, then simplicity. Champion unchanged, so the
  downstream prediction contract and test metrics are stable.
- **Phase C — pipeline agent + autopilot supervisor** (`src/agent/`): a typed
  tool registry over the pipeline functions, a **rule-based planner**, a trace
  writer, and an autopilot supervisor that walks the ten stages with a per-stage
  state machine (EXECUTE → VALIDATE gate → pass / raise card / HALT), four
  gate/card types, checkpoint/resume via provenance hashes, and a two-run
  determinism test. A second **query** entry mode ("which engines need
  inspection?" → units 81/34/35 with citations) shares the same registry and
  trace schema. Triage and sign-off cards never auto-pass.
- **Phase D — Streamlit app** (`src/app/streamlit_app.py`): an **Autopilot**
  landing page that launches the agent as a subprocess and renders its live
  journal as a narrative timeline, a **Decision Inbox** of grounded/cited cards
  with safe-default actions, and the existing wizard renamed to **Teaching
  mode** as the per-unit evidence viewer that cards deep-link into. Model
  constants (cap, typical-miss, risk thresholds) are read from
  `reports/metrics_model.json` + `config/pipeline.yaml`, not hard-coded.
- **Tests:** 48 total (16 original untouched), covering stage contracts, leakage
  guards, provenance/idempotency, planner plans, trace grounding, gate
  dispositions, and two-run determinism.

## What Was Adapted From Open Source

Inspiration sources are listed in `docs/build-spec.md`. The build **borrowed
ideas and discipline, not code**:

- **LGDiMaggio/predictive-maintenance-mcp** — inspiration for the MCP/RAG-style
  condition-monitoring tooling and diagnostic-report framing. Not forked; a
  smaller original version was built.
- **kpeters/exploring-nasas-turbofan-dataset** — reference for the C-MAPSS RUL
  modeling approach (piecewise RUL cap, sensor handling).
- **TheDatumOrg/TSB-AD** — borrowed the evaluation mindset for time-series
  anomaly detection, not the benchmark itself.
- **NASA PCoE Data Repository** — authoritative data source and citation.

The pipeline and agent layers use **no orchestration framework** (no Airflow,
Prefect, or LangChain) — the DAG runner, provenance, planner, and supervisor are
hand-written for auditability.

## What Is Original

- This codebase in its entirety (`src/`, `tests/`, notebooks, app), including
  the `src/pipeline/` DAG runner + provenance + auto-manifest and the
  `src/agent/` registry, rule planner, trace, and autopilot supervisor.
- The five knowledge-base documents and all framing/limitations/audit docs.
- The diagnostic-evidence JSON schema, the decision-card schema, and the
  deterministic, key-free grounded summary template.
- The evaluation set (hand-written queries), the diagnostic-output governance
  checks, and the model-selection bake-off criteria and champion contract.

## What Can and Cannot Be Claimed on a Résumé

**Can claim:** an independent, end-to-end prototype on public data
demonstrating a **deterministic pipeline agent with human-in-the-loop decision
gates**, a cross-validated **model-selection bake-off**, RUL baseline modeling,
structured evidence extraction, a retrieval-grounded and uncertainty-aware
diagnostic workflow, evaluation discipline, and human-in-the-loop governance.
Metrics are truthful and traceable to `reports/` (`docs/resume-bullets.md`).

**Cannot claim:** Caterpillar affiliation or data; heavy-equipment domain
expertise; CAN/J1939 or telematics-platform experience; production digital-twin
or production-telemetry ownership; production readiness; state-of-the-art
performance; or that any result transfers to a fielded asset.

**Banned descriptors (naming law — never describe the agent as):** "fully
autonomous", "autonomous agent", "self-healing", "no human needed", "real-time"
(say "live step-by-step"), or "an LLM agent". The correct framing is always a
**deterministic pipeline agent/supervisor with human-in-the-loop decision
gates** — it automates the analysis; the human owns the decision.

## Next Improvements (Given Another Day)

- **MCP stdio server (stretch):** expose the same tool registry over an MCP
  stdio server (`tools/list` + `tools/call`), dependency-isolated, so external
  clients can drive the pipeline through one adapter.
- **LLM planner behind a flag:** wire the default-off `--planner llm` interface
  stub to a real subscription-CLI adapter, keeping the same grounding, citation,
  and determinism guardrails and the rule planner as the default.
- **Beyond FD001:** extend to the multi-condition/multi-fault subsets
  (FD002–FD004) with operating-condition normalization.
- Swap TF-IDF for dense/semantic embeddings and compare retrieval hit-rate.
- Add basic drift/performance monitoring and a small alerting rule.
- Add uncertainty quantification (quantile or per-tree spread RUL bounds)
  alongside the point estimate.

## Audit Sign-Off

- [x] Truthfulness guardrails re-checked against `docs/build-spec.md`
      (QA whole-repo grep 2026-07-06: every Caterpillar mention is a
      non-affiliation disclaimer; "production-ready" appears only negated;
      CAN/J1939 and digital-twin mentions are explicit cannot-claim negations).
- [x] Metrics filled from `reports/` and résumé bullets updated
      (RMSE 17.2 / MAE 12.1 capped; hit@4 1.00; bake-off Ridge 21.0 / RF 18.2;
      filled 2026-07-06).
- [x] No forbidden claims present in README or docs; naming law enforced
      (no banned descriptors; 48/48 tests collected; eval harness
      model=ok retrieval=ok diagnostics=ok).
