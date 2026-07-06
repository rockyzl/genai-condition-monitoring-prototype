# Evaluation Summary

Automated evaluation of the GenAI-assisted condition-monitoring prototype. Independent R&D prototype on public NASA C-MAPSS data — not production-validated, not affiliated with any equipment manufacturer.

## A. Model Metrics (RUL regression)

- Units scored: **100**
- RMSE (recomputed from `test_predictions.csv`): **18.188** cycles
- MAE (recomputed): **13.141** cycles
- Cross-check vs `metrics_model.json` (uncapped truth = the same target we recompute against): DS reports RMSE 18.1877 / MAE 13.1409 → match ✅.
- DS headline metrics vs capped truth (cap=125, the trained target): RMSE 17.1989 / MAE 12.0709 / R² 0.8158. These are the numbers to quote for the model; our recomputation validates the prediction file, not the cap policy.

## B. Retrieval Quality (hit@k on hand-written queries)

- Queries: **10**, hits within top-4: **10/10** (hit@4 = **1.0**)

| # | Query | Expected | Hit | Rank | Top result |
|---|-------|----------|-----|------|------------|
| 1 | high-pressure compressor degradation sensor signature ris… | failure_modes.md | ✅ | 1 | failure_modes.md |
| 2 | bearing wear vibration and how it shows in thermodynamic … | failure_modes.md | ✅ | 1 | failure_modes.md |
| 3 | ordered checklist steps when a unit is flagged high risk | maintenance_review_checklist.md | ✅ | 1 | maintenance_review_checklist.md |
| 4 | verify data quality frozen or railed sensor values before… | maintenance_review_checklist.md | ✅ | 1 | maintenance_review_checklist.md |
| 5 | how to interpret a RUL point estimate as a range not an e… | rul_interpretation.md | ✅ | 1 | rul_interpretation.md |
| 6 | RUL cap piecewise linear target ceiling 125 cycles healthy | rul_interpretation.md | ✅ | 1 | rul_interpretation.md |
| 7 | separate a sensor fault from genuine asset degradation co… | anomaly_review_guidelines.md | ✅ | 1 | anomaly_review_guidelines.md |
| 8 | distinguish a transient spike from a sustained persistent… | anomaly_review_guidelines.md | ✅ | 1 | anomaly_review_guidelines.md |
| 9 | are model outputs advisory decision support or decisions | human_in_loop_policy.md | ✅ | 1 | human_in_loop_policy.md |
| 10 | no safety-critical automation the system must not command… | human_in_loop_policy.md | ✅ | 1 | human_in_loop_policy.md |

## C. Diagnostic-Output Governance Checks

- Evidence records evaluated: **100**

| Check | Pass | Rate |
|-------|------|------|
| Report includes citations | 100/100 | 1.0 |
| Report states uncertainty | 100/100 | 1.0 |
| human_review_required == true | 100/100 | 1.0 |
| Summary echoes predicted RUL | 100/100 | 1.0 |
| Failure modes grounded in retrieved KB | 100/100 | 1.0 |
| Next steps grounded in retrieved KB | 100/100 | 1.0 |

**No violations.** Every diagnostic report cited evidence, carried uncertainty, forced human review, echoed the predicted RUL, and grounded every failure-mode and next-step claim in a retrieved knowledge-base chunk (no invented root causes).

## Commentary

The baseline RUL model lands at RMSE 18.188 / MAE 13.141 cycles (vs uncapped truth) on the FD001 test units. That is a credible baseline for a simple model on capped RUL targets, not a tuned state-of-the-art result — error concentrates near end-of-life, which is exactly why the assistant foregrounds uncertainty and human review. Our independent recomputation from the prediction file matches the DS uncapped metrics exactly, confirming the artifact is consistent.

Retrieval hit@4 is 1.0 across 10 hand-written queries — the TF-IDF index reliably surfaces the intended knowledge-base file. Misses, if any, reflect lexical overlap between sections (e.g. checklist vs. policy) rather than retrieval failure.

The governance checks are the point of this project: they enforce that no diagnostic ships without citations, uncertainty, a human-review flag, and claims traceable to retrieved evidence. All records passed.
