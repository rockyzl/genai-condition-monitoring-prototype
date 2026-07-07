---
title: Condition Monitoring Agent
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
suggested_hardware: cpu-basic
short_description: Deterministic RUL pipeline + HITL decision agent on NASA C-MAPSS
---

# Condition Monitoring Agent — live demo

Deterministic 10-stage remaining-useful-life (RUL) pipeline with a
human-in-the-loop decision agent, on the public NASA C-MAPSS FD001 turbofan
dataset. Independent R&D prototype — not production-validated, not affiliated
with any equipment manufacturer.

**This Space self-bootstraps.** The dataset, trained model, and all pipeline
artifacts are gitignored, so on the first visit after a cold start the app spends
about **1 minute** downloading NASA C-MAPSS (~12 MB) and running the pipeline
(train → predict → evidence → diagnose → evaluate). After that the same warm
container serves instantly. See `docs/deploy-demo.md` for full deploy + embed
instructions, and `README.md` for the project itself.

> HF Spaces storage is ephemeral (no persistent storage tier), so every cold
> start re-bootstraps. It is only ~1 minute and requires no committed data.
