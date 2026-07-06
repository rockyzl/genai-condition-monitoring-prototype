# GenAI-Assisted Condition Monitoring Prototype

> **Independent R&D prototype using public data.** Not affiliated with any
> equipment manufacturer; does not use proprietary equipment data. Built to
> explore how sensor time-series modeling and a retrieval-grounded (RAG)
> diagnostic workflow fit together for condition-monitoring analytics.

**Status: scaffolding — build in progress.** See `docs/build-spec.md` for the
full plan and truthfulness guardrails.

## What this will be

```
sensor time-series (NASA C-MAPSS)
   → RUL / anomaly baseline model
   → structured diagnostic evidence (JSON)
   → RAG diagnostic assistant (cited, uncertainty-aware)
   → maintenance-style report + Streamlit demo
   → evaluation & limitations
```

## Honest scope

- Public prognostics data (NASA C-MAPSS turbofan degradation) as a *proxy*
  for industrial condition monitoring — not heavy-equipment telemetry.
- Baseline models, not SOTA; the point is diagnostic traceability,
  evaluation discipline, and human-in-the-loop safeguards.
- The assistant cites its evidence, states uncertainty, and always
  recommends human review. It makes no safety-critical decisions.
