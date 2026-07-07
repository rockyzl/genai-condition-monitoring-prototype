# GenAI-Assisted Condition Monitoring Prototype

> ⚠️ **Independent R&D prototype on public data.** Not affiliated with
> Caterpillar or any equipment manufacturer, and using **no** proprietary or
> field equipment data. Built to explore how sensor time-series modeling and a
> retrieval-grounded (RAG) diagnostic workflow fit together for
> condition-monitoring analytics. Uses public NASA C-MAPSS turbofan simulation
> data as a proxy. **Not production-ready.** Caterpillar is named here only as
> the target-role inspiration for the exercise.

## What This Is

A small but complete, end-to-end prototype that ingests sensor time-series
data, predicts Remaining Useful Life (RUL), converts predictions into
structured diagnostic evidence, grounds a diagnostic assistant on a curated
knowledge base via retrieval, and condenses the whole run into a short
**decision queue** for a human reviewer — all behind a Streamlit demo. The
analysis runs as a **deterministic 10-stage pipeline** driven by a
**pipeline agent (supervisor) with human-in-the-loop decision gates**. It is a
portfolio prototype demonstrating transferable capability in predictive
analytics, pipeline engineering, RAG diagnostic workflows, evaluation
discipline, and human-in-the-loop governance.

## Why It Matters for GenAI-Assisted Condition Monitoring

Most predictive-maintenance work stops at a score — an RUL number or an anomaly
flag. But a score is not a decision. The value (and the risk) lives in the
translation from score to defensible action: which sensors moved, is this real
degradation or a sensor glitch, how confident are we near end-of-life, and what
should a reviewer do next. This prototype puts a **grounded, cited diagnostic
layer** between the model and the human, and an agent that does the heavy
lifting while leaving every decision to a person. See
[`docs/problem-framing.md`](docs/problem-framing.md).

## Autopilot — One Command

**One command. The agent runs the full analysis end-to-end, shows its work at
every step, then hands you a short decision queue.** You make the decisions;
the agent did the heavy lifting and can prove every claim.

`python -m src.agent run --all` launches the deterministic pipeline agent. It
walks all ten stages, streaming a live step-by-step journal the Streamlit app
renders as a narrative timeline (not raw logs). At each stage it runs a
validation gate: pass, raise a decision card, or HALT on a governance
violation (schema break, leakage canary, a champion that fails to beat the
linear floor). The run ends in a **Decision Inbox**: at most five grounded,
cited decision cards, each with plain-language signals (one is always the
uncertainty/optimism caveat), a deep-link to the underlying evidence, and
safe-default actions with a consequence preview ("Export = draft work orders
only, never commands"). Triage and sign-off cards **never auto-pass** — a human
must answer them. Runs are checkpoint/resume (provenance hashes are the state,
so answering a gate resumes from the checkpoint, never from stage one) and
deterministic (same inputs + seed → byte-identical decisions and trace, covered
by a two-run determinism test).

## Architecture

```
  $ python -m src.agent run --all            ← ONE COMMAND (flagship)
  ────────────────────────────────────────────────────────────────────
  PIPELINE AGENT / SUPERVISOR — deterministic · human-in-the-loop gates
  per stage:  EXECUTE → VALIDATE (gate) → pass · raise card · HALT
  ────────────────────────────────────────────────────────────────────
        │  walks the 10-stage pipeline, streaming a live NDJSON journal
        ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  10-STAGE PIPELINE  src/pipeline · provenance-skip · auto-manifest │
  │  s01 ingest → s02 eda → s03 preprocess → s04 features → s05 model  │
  │      → s06 select → s07 predict → s08 evidence → s09 diagnose      │
  │      → s10 eval                                                    │
  └──────────────────────────────────────────────────────────────────┘
     s06 selects champion          s08/s09 build + ground the diagnosis
     ┌───────────────────┐        ┌────────────────┐ TF-IDF ┌──────────┐
     │ RUL champion:     │        │ evidence JSON  │◀──────▶│ knowledge│
     │ RandomForest      │        │ + RAG assistant│ retr.  │ base kb/*│
     │ (beat Ridge floor)│        └────────────────┘        └──────────┘
     └───────────────────┘                │ cited, uncertainty-aware
                                           ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  DECISION INBOX — ≤5 grounded, cited cards with safe-default       │
  │  actions; triage + sign-off cards NEVER auto-pass                  │
  │        →  the human makes the call                                 │
  └──────────────────────────────────────────────────────────────────┘

  Second entry mode (same registry, same trace schema):
  $ python -m src.agent ask "which engines need inspection?"
      → grounded bilingual answer: units 81 / 34 / 35, each with a citation
```

## Data Source

NASA C-MAPSS Turbofan Engine Degradation Simulation Data, from the NASA
Prognostics Center of Excellence (PCoE) Data Repository — a public dataset. This
prototype uses the **FD001** subset: one operating condition (sea level), one
fault mode (high-pressure compressor degradation), 100 train and 100 test
engine trajectories, 21 sensor channels plus 3 operational settings. The data is
simulated, not field telematics, and is used as a public proxy for industrial
condition monitoring. Full provenance and column details:
[`docs/data-sources.md`](docs/data-sources.md).

## Modeling Approach

A transparent RUL-regression **baseline** chosen by an explicit **model
bake-off** (stage s06), not by chasing a leaderboard. Three candidates compete
on identical **unit-grouped 5-fold cross-validation** folds (GroupKFold by
engine, so no unit is scored on itself): a Ridge linear **floor**, a
RandomForest **champion**, and a HistGradientBoosting **challenger**. Selection
criteria in priority order: (1) grouped-CV RMSE, (2) end-of-life calibration
(RMSE on near-failure rows plus the optimistic-error fraction), (3) a simplicity
tiebreak. RandomForest wins at **18.2 ± 0.5 cycles CV-RMSE**, beating the Ridge
floor (**21.0**) by 2.8 cycles; the gradient-boosting challenger (**18.6**) does
not clear the clear-win bar, so the incumbent stays for determinism. Full
rationale: [`reports/model_selection.md`](reports/model_selection.md). The
training target uses a piecewise-linear RUL cap (125 cycles) so the model
focuses near end-of-life; output is a point estimate mapped to three advisory
risk bands (high ≤30, medium 30–80, low >80 cycles). See
[`docs/knowledge_base/rul_interpretation.md`](docs/knowledge_base/rul_interpretation.md).

## RAG Diagnostic Workflow

1. **Evidence extraction** (stage s08) — model outputs become structured JSON:
   unit, cycle, sensor summary, predicted RUL, risk band, top contributing
   signals, and an uncertainty note.
2. **Retrieval** (stage s09) — a lightweight TF-IDF retriever pulls the most
   relevant sections from the local knowledge base (failure modes, review
   checklist, RUL interpretation, anomaly guidelines, human-in-the-loop policy).
3. **Grounded summary** — a deterministic template composes a plain-English
   diagnostic: supporting sensor evidence, plausible failure modes, recommended
   next inspection steps, an explicit uncertainty statement, citations to the
   retrieved snippets, and a mandatory human-review warning. It never makes
   safety-critical decisions.

The assistant and the agent's planner are deliberately **deterministic and
key-free** (no LLM in the loop) so every output is reproducible and auditable —
see [`docs/limitations.md`](docs/limitations.md).

## Evaluation

Measured on FD001, a small hand-written query set, and the agent trace (see
[`reports/evaluation_summary.md`](reports/evaluation_summary.md),
[`reports/model_selection.md`](reports/model_selection.md)).

| Metric | Value |
| --- | --- |
| RUL RMSE (test) | **17.2** cycles vs capped truth (18.2 vs uncapped) |
| RUL MAE (test) | **12.1** cycles vs capped truth (13.1 vs uncapped) |
| Model bake-off (GroupKFold-5 CV-RMSE) | Ridge floor **21.0** · RandomForest champion **18.2 ± 0.5** · HistGBM **18.6** (not picked) |
| Retrieval hit-rate | **1.00** (hit@4, 10/10 hand-written queries, all rank 1) |
| Diagnostic-output governance | **100/100** evidence records cite evidence, state uncertainty, and force human review; zero violations |
| Agent governance | triage + sign-off cards never auto-pass; every recommendation traces to a journal step + citation; two-run determinism test |
| Test suite | **48** tests (16 original untouched) |

## Limitations

Simulated data (not field telematics); single subset and single fault mode
(FD001); a baseline model, not SOTA; TF-IDF lexical retrieval (no dense
embeddings); a deterministic, rule-based agent and template assistant (no LLM in
the loop, by design); threshold-based gates; decision cards are advisory only;
no drift or live monitoring; and **not production-ready**. Full accounting in
[`docs/limitations.md`](docs/limitations.md).

## How to Run Locally

```bash
# 1. Create and activate a virtual environment, install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. FLAGSHIP: run the pipeline agent end-to-end with decision gates
#    (walks s01→s10, streams a live journal, ends in the Decision Inbox)
python -m src.agent run --all

# 3. Or run the raw 10-stage pipeline directly (writes reports/ + manifest)
python -m src.pipeline run --all

# 4. Ask a grounded, cited question (second entry mode, same registry)
python -m src.agent ask "which engines need inspection?"

# 5. Launch the Streamlit demo (Autopilot page + Decision Inbox + Teaching mode)
streamlit run src/app/streamlit_app.py
```

Gated runs with no UI print the pending card and its safe default, then exit
leaving the card in `reports/autopilot_inbox/pending/`; answer it and re-run to
resume (earlier stages skip via provenance).

## Resume-Safe Description

*Independent R&D prototype (public NASA C-MAPSS data) demonstrating an
end-to-end GenAI-assisted condition-monitoring workflow: a deterministic
10-stage pipeline driven by a pipeline agent with human-in-the-loop decision
gates, a cross-validated model-selection bake-off, structured
diagnostic-evidence extraction, a retrieval-grounded and uncertainty-aware
diagnostic assistant, evaluation discipline, and human-in-the-loop governance —
with no proprietary data and no production claims.* Full bullet drafts in
[`docs/resume-bullets.md`](docs/resume-bullets.md).
