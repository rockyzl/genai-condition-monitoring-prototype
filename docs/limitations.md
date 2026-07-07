# Limitations

An explicit, honest accounting of what this prototype does not do and cannot
claim. This list is a feature, not a disclaimer to skim past: knowing the
boundary is part of the deliverable.

## Data Is Simulated, Not Field Telematics

The prototype uses NASA C-MAPSS, a physics-based *simulation* of turbofan
degradation. It is a clean, well-behaved public proxy for condition-monitoring
data — not real field telematics. Real assets bring messier realities:
missing data, clock skew, maintenance actions mid-life, varying duty cycles,
and instrumentation faults that the simulation does not fully reproduce.
Results here do not transfer directly to any fielded asset.

## Single Dataset, Single Fault Mode

Modeling is limited to the FD001 subset: one operating condition (sea level)
and one fault mode (high-pressure compressor degradation). It does not cover
the multi-condition or multi-fault subsets (FD002–FD004), and it is not a
cross-asset or cross-domain evaluation.

## Baseline Model, Not State of the Art

The RUL model is a straightforward baseline chosen for transparency, not a
tuned or state-of-the-art architecture. The model-selection bake-off compares
only a small classical set (Ridge floor, RandomForest, HistGradientBoosting)
under grouped cross-validation; there are no deep sequence models and no
extensive hyperparameter search. The champion is a self-explaining ensemble of
trees, deliberately kept simple. The goal is a credible, explainable reference
point — not the lowest possible error.

## Lexical Retrieval, Not Dense Embeddings

The RAG layer uses TF-IDF lexical retrieval over a small local knowledge base.
It has no dense/semantic embeddings and no vector database, so it can miss
paraphrases that share no vocabulary with the query. This keeps the system
dependency-light and fully offline, at the cost of semantic recall.

## Deterministic Template Assistant — No LLM in the Loop (By Design)

The diagnostic summary is produced by a deterministic template over the
retrieved evidence, not a generative LLM. This is a deliberate choice: it keeps
the system key-free, fully reproducible, offline, and auditable, and it
removes any risk of hallucinated root-cause claims. The trade-off is less
fluent, less flexible prose. Swapping in an LLM is a natural next step but is
out of scope here.

## Rule-Based Agent and Threshold Gates — No LLM (By Design)

The pipeline agent is a **rule-based** supervisor and planner, not a generative
one. Its intent parsing, stage orchestration, and validation gates are
deterministic and threshold-based: gates fire on fixed, config-declared
conditions (schema checks, a leakage canary, the champion-beats-floor bound,
risk-distribution bounds), and those thresholds are hashed into the trace so
they cannot be silently weakened. This is a deliberate choice — reproducible and
auditable over flexible — and its limits are the flip side: the agent handles
only the intents and failure conditions it was written for, and cannot reason
about situations outside its rules. An LLM planner exists only as a
default-off, not-configured interface stub. There is no LLM in the loop.

## Decision Cards Are Advisory Only

Every decision card the agent raises is advisory decision support, not an
action. Card actions produce drafts and views only — "Export = draft work
orders only, never commands" — and triage and sign-off cards never auto-pass; a
human must answer them. The agent does not close any loop, dispatch
maintenance, or mark an engine safe. All governance rests on human review — see
`knowledge_base/human_in_loop_policy.md`.

## No Drift or Live Monitoring

The prototype is a static, batch demonstration. There is no data-drift
monitoring, no model-performance monitoring over time, no retraining pipeline,
no alerting, and no streaming ingestion.

## Not Production-Ready

This is an independent R&D prototype, not a production system. It is not
hardened, not load-tested, not integrated with any asset, and makes no
safety-critical or unattended decisions. All outputs are advisory and require
human review — see `knowledge_base/human_in_loop_policy.md`.
