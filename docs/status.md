# Project Status — what is done

> Last updated: 2026-07-07. Canonical "what exists and works" summary for
> collaborators (human or agent). Numbers are traceable to `reports/`.

## Live entry points

| Entry | URL / command |
|---|---|
| Public demo (HF Space, docker, self-bootstrapping) | https://rockyaaos-condition-monitoring-agent.hf.space |
| Personal-site subpage (iframe embed, en+zh) | https://sciencesloop.com/agent/condition-monitoring/ |
| 68-second demo video | `docs/media/demo.mp4` (linked at top of README) |
| Local app | `.venv/bin/streamlit run src/app/streamlit_app.py` |
| Agent CLI | `python -m src.agent run --all [--engine native\|langgraph]` · `ask "..." [--engine ...]` |
| Pipeline CLI | `python -m src.pipeline run --all \| --stage sXX \| --from sXX` |

## Done (build order)

1. **v1 core** — C-MAPSS FD001 RandomForest RUL baseline (RMSE 17.2 capped /
   18.2 uncapped, MAE 12.1/13.1, R² 0.82), diagnostic-evidence JSON layer,
   five-file knowledge base + TF-IDF retrieval, deterministic cited assistant,
   eval harness (retrieval hit@4 = 1.00; diagnostic governance 100/100).
2. **Bilingual wizard UI** — button-gated 5-step walkthrough (now "Teaching
   mode"), EN default + full 中文, canonical per-step descriptions.
3. **Pipeline layer (Phase A)** — 10 wrapped stages (`s01 ingest … s10 eval`),
   StageSpec-enforced explanations (EN+ZH), provenance-stamped artifacts with
   idempotent skip, NDJSON step journal, auto-generated
   `reports/pipeline_manifest.md`.
4. **Model selection (Phase B)** — GroupKFold(5)-by-unit bake-off: Ridge floor
   21.01±1.38, RF champion 18.20±0.52 (beats floor by 2.81), HistGBM 18.59
   rejected (clear-win bar). `reports/model_selection.{md,json}`.
5. **Autopilot agent (Phase C)** — typed 7-tool registry, rule-based planner
   (LLM planner = flagged stub), supervisor state machine
   (EXECUTE→VALIDATE→pass/card/HALT), 4 decision-card types (triage + sign-off
   NEVER auto-pass), checkpoint/resume via provenance + answered-card files,
   grounded bilingual `ask`, two-run byte determinism.
6. **Autopilot UI + Decision Inbox (Phase D)** — live journal timeline, pending
   cards with grounded signals + consequence previews, fleet triage table,
   metrics single-sourced from `reports/metrics_model.json` + config.
7. **Docs + governance eval (Phase E)** — README (hiring-manager-readable),
   resume bullets, limitations, final audit; eval **Section D — autonomy
   governance** (5 checks over real run artifacts, incl. thresholds-hash
   anti-weakening) — all PASS.
8. **Agent Chat mode** (default landing) — confirmation-first plan preview,
   one in-place progress bubble (anti-conversation-tax: "Skip to results"),
   in-chat pinned decision cards that morph on answer, mid-run read-only
   queries answered/queued (never refused), trust badge
   "Deterministic · Grounded · No LLM", reporting-not-reasoning voice.
9. **Cloud packaging** — self-bootstrapping root `streamlit_app.py`
   (downloads C-MAPSS ~12 MB + runs the pipeline on cold start, ~2-3 min;
   HF killed persistent storage so every cold start re-bootstraps), Dockerfile
   (HF dropped the native streamlit SDK for new Spaces), fresh-clone verified
   49 s end-to-end with byte-identical metrics.
10. **Site embedding** — `sciencesloop.com/agent/condition-monitoring/`
    (en+zh subpages), additive cross-link card on `/agent/` (ChemGraph demo
    untouched — verified post-deploy).
11. **LangGraph dual engine** — `src/agent/langgraph_engine.py`: the same
    governed workflow as a StateGraph (typed state, conditional gate edges,
    `interrupt`-based HITL that applies the same never-auto policy, in-memory
    checkpointer for in-run resume only — cross-process resume stays on
    provenance+files), plus a ToolNode tool-calling ask path. **Byte-parity
    with the native engine** across dry-run/gated/auto and both canonical
    queries; eval Section D passes on langgraph artifacts unmodified. Native
    remains default; `langgraph` isolated in `requirements-agent.txt` (the
    HF Space does not ship it). Rationale docs: `docs/langgraph-engine.md`.
12. **Demo video** — 68 s, frame-verified 6-beat storyboard, reproducible via
    `scripts/record_demo.py`.

## Test suite

**68 passing** (the original 16 never modified): data/model 11 · RAG 5 ·
pipeline 9 · model-selection 7 · agent 12 · eval-governance 4 · app 8 ·
bootstrap 2 · langgraph-engine 11 (importorskip). Plus `health` checks:
`python src/eval/run_eval.py` → `model=ok retrieval=ok diagnostics=ok
governance=ok`.

## Hard rules that bind all future work

- Truthfulness guardrails in `docs/build-spec.md` (no Caterpillar
  data/affiliation, no production claims). Banned agent descriptors in
  `reports/final_audit.md` ("fully autonomous", "self-healing", bare
  "LLM agent", "real-time").
- Voice rule in the app: reporting, not reasoning (no first-person mental
  verbs).
- Metrics single source of truth: `reports/metrics_model.json` +
  `reports/evaluation_summary.md`; everything else quotes.
- `data/`, `models/*.joblib`, runtime journals/traces/inbox are gitignored;
  samples live in `reports/samples/`.

## Known open items

- Xiangju-facing UX polish: none planned (demo is EN-first for hiring
  managers).
- Stretch (documented, not built): MCP stdio server over the registry; LLM
  planner behind `--planner llm` (same ToolNode loop); FD002–FD004; dense
  retrieval comparison; drift monitoring.
- WSL2 gotcha: HistGBM OpenMP deadlock → threadpoolctl 1-thread cap (in code).
