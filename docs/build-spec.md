# Build Spec — GenAI-Assisted Condition Monitoring Prototype

> Execution prompt approved by Lu (2026-07-06). Target: Caterpillar
> "Lead Data Scientist – Gen AI for Condition Monitoring Analytics".
> One complete project, not five small ones. 1–2 day scope.

## Goal

Build a small but credible prototype:

sensor / time-series data → anomaly detection or RUL modeling → diagnostic
evidence extraction → RAG or MCP-style diagnostic assistant → human-readable
maintenance-style report → evaluation and limitations.

## Truthfulness guardrails (HARD)

- Do not claim Caterpillar data.
- Do not claim heavy-equipment expertise.
- Do not claim CAN/J1939.
- Do not claim production telemetry platform ownership.
- Do not claim production digital twin ownership.
- Do not claim this is production-ready.
- Label clearly as an independent R&D prototype using public data.

## Source seeds

1. **LGDiMaggio/predictive-maintenance-mcp** — inspiration for MCP-style
   condition-monitoring tools, vibration analysis, diagnostic workflows, RAG
   search, RUL estimation, report generation. Do NOT fork; build our own
   smaller version.
2. **kpeters/exploring-nasas-turbofan-dataset** — main RUL / degradation
   modeling starting point (NASA C-MAPSS).
3. **TheDatumOrg/TSB-AD** — evaluation-discipline reference for time-series
   anomaly detection (borrow the metric mindset, not the whole benchmark).
4. **NASA Prognostics Center of Excellence Data Repository** — authoritative
   data-source citation.

Optional: mohyunho/N-CMAPSS_DL (deeper RUL prep), lestercardoz11/
fault-detection-for-predictive-maintenance-in-industry-4.0 (bearing
baselines), Azure-Samples/azure-search-openai-demo (only if a cloud demo is
added later).

## Phases

### Phase 1 — Repository setup
Structure (already scaffolded): README.md, docs/{problem-framing,
data-sources, limitations, resume-bullets}.md, docs/knowledge_base/,
data/{raw,processed}, notebooks/{01_eda,02_baseline_model}.ipynb,
src/{data,features,models,diagnostics,rag,app}, tests/, reports/.

### Phase 2 — Data
NASA C-MAPSS turbofan degradation (FD001 first). Document: source, asset
type, sensor columns, train/test split, target variable, limitations, why
this is a proxy for condition monitoring (not Caterpillar equipment). If
download blocked, use a clearly-labeled sample/synthetic subset.

### Phase 3 — Baseline modeling
ONE simple baseline first (no SOTA chasing):
- Option A: RUL regression (Random Forest / XGBoost / LightGBM / simple NN)
- Option B: anomaly detection (IsolationForest / PCA reconstruction / AE)
Required output: training script, evaluation script, saved metrics, plots,
short error analysis, high-error / FP / FN case review.

### Phase 4 — Diagnostic evidence layer
Convert model outputs to structured JSON evidence per asset/time window:
asset_id, cycle/timestamp, sensor summary, anomaly score or predicted RUL,
threshold/risk band, top contributing signals, uncertainty/limitation note.

### Phase 5 — RAG / diagnostic assistant
Local knowledge base under docs/knowledge_base/: failure_modes.md,
maintenance_review_checklist.md, rul_interpretation.md,
anomaly_review_guidelines.md, human_in_loop_policy.md.
Lightweight retriever (TF-IDF / FAISS / Chroma). Assistant takes structured
evidence and produces: plain-English diagnostic summary, supporting sensor
evidence, possible failure modes, recommended next inspection steps,
uncertainty statement, citations to retrieved snippets, human-review
warning. Never makes safety-critical decisions.

### Phase 6 — App / demo
Streamlit preferred (FastAPI + simple frontend acceptable). Minimum UI:
select asset/time window; sensor trend plot; model output; retrieved
evidence; diagnostic summary; limitations.

### Phase 7 — Evaluation
- Model metric (RMSE/MAE or precision/recall/F1/AUC as fits the task)
- Retrieval hit-rate on a small hand-written query set
- Diagnostic-output checks: cites evidence; includes uncertainty; no
  unsupported root-cause claims; recommends human review.
Output: reports/evaluation_summary.md.

### Phase 8 — README
Hiring-manager-readable in 60 seconds. Sections: what this is; why it
matters for GenAI-assisted condition monitoring; architecture diagram; data
source; modeling approach; RAG diagnostic workflow; evaluation; limitations;
how to run locally; resume-safe description. Explicit disclaimer:
independent R&D prototype on public data, not affiliated with Caterpillar.

### Phase 9 — Resume bullets
2–3 truthful bullets in docs/resume-bullets.md (style provided in the
original prompt). No metrics unless actually produced.

### Phase 10 — Final audit
reports/final_audit.md: what was built; what was adapted from open source;
what is original; what can/cannot be claimed on a resume; next improvements
if given another day.

## Success criteria

Demonstrates transferable capability in condition monitoring, predictive
analytics, anomaly/RUL modeling, RAG diagnostic workflows, evaluation
discipline, and human-in-the-loop governance — without pretending to be
production industrial-equipment experience.
