# Final Audit

> Independent R&D prototype on public NASA C-MAPSS data. Not affiliated with
> Caterpillar or any equipment manufacturer; no proprietary or field data; not
> production-ready. This audit is the honest, end-of-build accounting required
> by the build spec (Phase 10).

## What Was Built

An end-to-end GenAI-assisted condition-monitoring prototype:

- Sensor time-series ingestion and feature prep on NASA C-MAPSS **FD001**.
- A Remaining Useful Life (RUL) **baseline model** (Random Forest regression)
  with saved metrics and error analysis. *(Results: RMSE 17.2 / MAE 12.1 /
  R² 0.82 vs capped truth (cap=125); RMSE 18.2 / MAE 13.1 vs uncapped truth —
  see `reports/metrics_model.json`.)*
- A **diagnostic evidence layer** that converts model outputs into structured
  JSON per unit/window (sensor summary, predicted RUL, risk band, top
  contributing signals, uncertainty note).
- A **TF-IDF RAG assistant** over a five-file local knowledge base
  (`docs/knowledge_base/`) producing cited, uncertainty-aware,
  maintenance-style summaries with a mandatory human-review warning.
- A **Streamlit demo** (`src/app/`) for interactive review.
- An **evaluation harness** (`src/eval/`) covering the model metric, retrieval
  hit-rate on a hand-written query set, and diagnostic-output checks. *(Results:
  retrieval hit@4 = 1.00 (10/10, all rank 1); diagnostic governance checks
  100/100 units pass all six checks, zero violations — see
  `reports/evaluation_summary.md`.)*

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

## What Is Original

- This codebase in its entirety (`src/`, `tests/`, notebooks, app).
- The five knowledge-base documents and all framing/limitations/audit docs.
- The diagnostic-evidence JSON schema and the deterministic, key-free grounded
  summary template.
- The evaluation set (hand-written queries) and diagnostic-output checks.

## What Can and Cannot Be Claimed on a Résumé

**Can claim:** an independent, end-to-end prototype on public data
demonstrating RUL baseline modeling, structured evidence extraction, a
retrieval-grounded and uncertainty-aware diagnostic workflow, evaluation
discipline, and human-in-the-loop governance. Metrics once they exist
(`docs/resume-bullets.md`).

**Cannot claim:** Caterpillar affiliation or data; heavy-equipment domain
expertise; CAN/J1939 or telematics-platform experience; production digital-twin
or production-telemetry ownership; production readiness; state-of-the-art
performance; or that any result transfers to a fielded asset.

## Next Improvements (Given Another Day)

- Swap TF-IDF for dense/semantic embeddings and compare retrieval hit-rate.
- Add an optional LLM-in-the-loop summarizer with the same citation and
  uncertainty guardrails, gated behind the deterministic baseline.
- Extend beyond FD001 to the multi-condition/multi-fault subsets
  (FD002–FD004) and normalize by operating condition.
- Add basic drift/performance monitoring and a small alerting rule.
- Add uncertainty quantification (e.g., quantile or ensemble RUL bounds)
  instead of a single point estimate.

## Audit Sign-Off

- [x] Truthfulness guardrails re-checked against `docs/build-spec.md`
      (QA whole-repo grep 2026-07-06: every Caterpillar mention is a
      non-affiliation disclaimer; "production-ready" appears only negated;
      CAN/J1939 and digital-twin mentions are explicit cannot-claim negations).
- [x] Metrics filled from `reports/` and résumé bullets updated
      (RMSE 17.2 / MAE 12.1 capped; hit@4 1.00; filled 2026-07-06).
- [x] No forbidden claims present in README or docs (QA verdict: clean;
      16/16 tests green; eval harness model=ok retrieval=ok diagnostics=ok).
