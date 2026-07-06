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
tuned or state-of-the-art architecture. No deep sequence models, no
hyperparameter search beyond the essentials, no ensembling. The goal is a
credible, explainable reference point — not the lowest possible error.

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

## No Drift or Live Monitoring

The prototype is a static, batch demonstration. There is no data-drift
monitoring, no model-performance monitoring over time, no retraining pipeline,
no alerting, and no streaming ingestion.

## Not Production-Ready

This is an independent R&D prototype, not a production system. It is not
hardened, not load-tested, not integrated with any asset, and makes no
safety-critical or autonomous decisions. All outputs are advisory and require
human review — see `knowledge_base/human_in_loop_policy.md`.
