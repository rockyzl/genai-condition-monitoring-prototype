# Problem Framing — Why GenAI-Assisted Condition Monitoring

> Independent R&D prototype on public data. Not affiliated with any equipment
> manufacturer and using no proprietary equipment data.

## The Gap Between a Model Score and a Maintenance Decision

Predictive-maintenance projects usually stop at a number: a Remaining Useful
Life (RUL) estimate, an anomaly score, a probability of failure. But a number
is not a decision. A reviewer who receives "RUL = 27 cycles" still has to ask:
which sensors drove this? Is it a real degradation trend or a sensor glitch?
How confident should I be this close to end-of-life? What do I actually do
next? In practice that translation work — from score to defensible action —
is where most of the value and most of the risk live, and it is exactly what a
bare model output leaves unaddressed.

## What a Grounded, Cited Diagnostic Layer Adds

This prototype puts a retrieval-grounded (RAG) diagnostic layer between the
model and the human. Instead of emitting only a score, it assembles structured
evidence — the predicted RUL, the risk band, and the sensor signals that moved
— and retrieves relevant guidance from a small, curated knowledge base
(failure modes, review checklists, RUL interpretation, anomaly guidelines,
human-in-the-loop policy). The output is a plain-English, uncertainty-aware
summary with citations back to both the evidence and the guidance it used.
Every claim is traceable; nothing is asserted that the retrieved snippets do
not support. The point is auditability and reviewer trust, not a higher
leaderboard score.

## This Prototype's Honest Scope

The scope is deliberately narrow and clearly labeled. It uses NASA C-MAPSS
turbofan simulation data as a public *proxy* for industrial condition
monitoring — not heavy-equipment telemetry, and not field data. The model is a
transparent baseline, not a state-of-the-art system. The assistant is a
deterministic, key-free template, chosen so every output is reproducible and
auditable. The aim is to demonstrate transferable capability — sensor
modeling, evidence extraction, RAG grounding, evaluation discipline, and
human-in-the-loop governance — without pretending to be production
industrial-equipment experience. See `limitations.md` for the full boundary.
