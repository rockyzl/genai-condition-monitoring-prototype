# GenAI-Assisted Condition Monitoring Prototype

> ⚠️ **Independent R&D prototype on public data.** Not affiliated with
> Caterpillar or any equipment manufacturer, and using **no** proprietary or
> field equipment data. Built to explore how sensor time-series modeling and a
> retrieval-grounded (RAG) diagnostic workflow fit together for
> condition-monitoring analytics. Uses public NASA C-MAPSS turbofan simulation
> data as a proxy. **Not production-ready.** Caterpillar is named here only as
> the target-role inspiration for the exercise.

## What This Is

A small but complete, end-to-end prototype that takes sensor time-series data,
predicts Remaining Useful Life (RUL), turns that prediction into structured
diagnostic evidence, grounds a diagnostic assistant on a curated knowledge base
via retrieval, and produces a cited, uncertainty-aware, maintenance-style
report — all behind a simple Streamlit demo. It is a portfolio prototype, built
in a 1–2 day scope, to demonstrate transferable capability in predictive
analytics, RAG diagnostic workflows, evaluation discipline, and
human-in-the-loop governance.

## Why It Matters for GenAI-Assisted Condition Monitoring

Most predictive-maintenance work stops at a score — an RUL number or an anomaly
flag. But a score is not a decision. The value (and the risk) lives in the
translation from score to defensible action: which sensors moved, is this real
degradation or a sensor glitch, how confident are we near end-of-life, and what
should a reviewer do next. This prototype puts a **grounded, cited diagnostic
layer** between the model and the human, so every recommendation is traceable
to both the sensor evidence and the guidance it drew on. See
[`docs/problem-framing.md`](docs/problem-framing.md).

## Architecture

```
   NASA C-MAPSS turbofan sensor time-series (FD001)
                     │
                     ▼
        ┌──────────────────────────┐
        │  RUL baseline model      │   Random Forest regression
        │  (src/models)            │   → reports/metrics_model.json
        └──────────────────────────┘
                     │  predicted RUL + risk band + top sensors
                     ▼
        ┌──────────────────────────┐
        │  Diagnostic evidence     │   structured JSON per unit/window
        │  (src/diagnostics)       │
        └──────────────────────────┘
                     │  evidence JSON
                     ▼
        ┌──────────────────────────┐        ┌────────────────────────┐
        │  RAG assistant           │◀──────▶│  Knowledge base         │
        │  (src/rag)               │ TF-IDF │  docs/knowledge_base/*  │
        └──────────────────────────┘retrieval└────────────────────────┘
                     │  cited, uncertainty-aware summary
                     ▼
        ┌──────────────────────────┐
        │  Streamlit demo          │   trends · model output · evidence
        │  (src/app)               │   · diagnostic report · limitations
        └──────────────────────────┘
                     │
                     ▼
        Evaluation harness (src/eval) → reports/evaluation_summary.md
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

A transparent RUL-regression **baseline** (Random Forest), chosen for
explainability over leaderboard performance. The training target uses a
piecewise-linear RUL cap so the model focuses accuracy near end-of-life rather
than chasing large "healthy" values. Output is a point-estimate RUL mapped to
three advisory risk bands: high (≤30 cycles), medium (30–80), low (>80). Metrics
and error analysis are saved to `reports/`. See
[`docs/knowledge_base/rul_interpretation.md`](docs/knowledge_base/rul_interpretation.md)
for how these estimates should be read.

## RAG Diagnostic Workflow

1. **Evidence extraction** — model outputs become structured JSON: unit,
   cycle, sensor summary, predicted RUL, risk band, top contributing signals,
   and an uncertainty note.
2. **Retrieval** — a lightweight TF-IDF retriever pulls the most relevant
   sections from the local knowledge base (failure modes, review checklist, RUL
   interpretation, anomaly guidelines, human-in-the-loop policy).
3. **Grounded summary** — a deterministic template composes a plain-English
   diagnostic: supporting sensor evidence, plausible failure modes, recommended
   next inspection steps, an explicit uncertainty statement, citations to the
   retrieved snippets, and a mandatory human-review warning. It never makes
   safety-critical decisions.

The assistant is deliberately **deterministic and key-free** (no LLM in the
loop) so every output is reproducible and auditable — see
[`docs/limitations.md`](docs/limitations.md).

## Evaluation

Measured on FD001 and a small hand-written query set (see
`reports/evaluation_summary.md`). Placeholders are filled after the eval runs:

| Metric | Value |
| --- | --- |
| RUL RMSE (test) | **17.2** cycles vs capped truth (18.2 vs uncapped) |
| RUL MAE (test) | **12.1** cycles vs capped truth (13.1 vs uncapped) |
| Retrieval hit-rate | **1.00** (hit@4, 10/10 hand-written queries, all rank 1) |
| Outputs citing evidence + stating uncertainty + recommending human review | target 100% |

Diagnostic-output checks confirm each summary cites its evidence, includes an
uncertainty statement, makes no unsupported root-cause claim, and recommends
human review.

## Limitations

Simulated data (not field telematics); single subset and single fault mode
(FD001); a baseline model, not SOTA; TF-IDF lexical retrieval (no dense
embeddings); a deterministic template assistant (no LLM in the loop, by design);
no drift or live monitoring; and **not production-ready**. Full accounting in
[`docs/limitations.md`](docs/limitations.md).

## How to Run Locally

```bash
# 1. Create and activate a virtual environment, install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Train the RUL baseline (writes reports/metrics_model.json)
.venv/bin/python src/models/train_baseline.py

# 3. Build structured diagnostic evidence from model outputs
.venv/bin/python src/diagnostics/build_evidence.py

# 4. Run the evaluation harness (writes reports/evaluation_summary.md)
.venv/bin/python src/eval/run_eval.py

# 5. Launch the Streamlit demo
.venv/bin/streamlit run src/app/streamlit_app.py
```

## Resume-Safe Description

*Independent R&D prototype (public NASA C-MAPSS data) demonstrating an
end-to-end GenAI-assisted condition-monitoring workflow: RUL baseline modeling,
structured diagnostic-evidence extraction, a retrieval-grounded and
uncertainty-aware diagnostic assistant, evaluation discipline, and
human-in-the-loop governance — with no proprietary data and no production
claims.* Full bullet drafts in [`docs/resume-bullets.md`](docs/resume-bullets.md).
