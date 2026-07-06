# Resume Bullets

Truthful, resume-safe bullets describing this project. Placeholders in
`{curly braces}` are to be filled by the lead **after** the evaluation runs —
do not invent numbers. Every bullet must remain accurate once the numbers land.

## Guardrails for These Bullets

- Describe it as an **independent R&D prototype on public data (NASA C-MAPSS)**.
- No claim of Caterpillar affiliation, heavy-equipment expertise, CAN/J1939,
  production telemetry, or production readiness.
- Cite metrics only once they exist in `reports/`. Until then, keep the
  `{...}` markers in place.

## Bullets (fill placeholders after eval)

- Built an independent, end-to-end GenAI-assisted condition-monitoring
  prototype on public NASA C-MAPSS turbofan data: engineered sensor
  time-series features and trained a Remaining Useful Life (RUL) baseline
  reaching **RMSE 17.2** / **MAE 12.1** cycles (vs capped truth, cap=125) on
  the FD001 test set.

- Designed a retrieval-grounded (RAG) diagnostic layer that converts model
  outputs into structured evidence and produces cited, uncertainty-aware,
  maintenance-style summaries — with a hand-built evaluation set showing
  **100% retrieval hit-rate (hit@4, 10/10)** and 100% of outputs citing evidence,
  stating uncertainty, and recommending human review.

- Enforced human-in-the-loop governance and evaluation discipline throughout:
  every output is advisory, traceable to its evidence, and reviewed against
  explicit checks for unsupported root-cause claims — demonstrating
  transferable predictive-analytics and responsible-AI capability without
  proprietary or production data.

## Placeholder Key (for the lead)

| Placeholder | Filled value (2026-07-06) | Source |
| --- | --- | --- |
| `{RMSE}` | 17.2 cycles (capped truth; 18.2 uncapped) | `reports/metrics_model.json` |
| `{MAE}` | 12.1 cycles (capped truth; 13.1 uncapped) | `reports/metrics_model.json` |
| `{hit_rate}` | 1.00 (hit@4, 10/10) | `reports/evaluation_summary.md` |
