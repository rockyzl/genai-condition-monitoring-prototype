# Model selection — bake-off report

Champion: **RandomForestRegressor**  ·  seed **42**  ·  protocol: GroupKFold(5) by `unit_id` on 20,631 training rows / 100 engines. The test set is never touched during selection.

## How to read this (plain language)

Three models competed on the same engines, judged on how many cycles they typically miss by (lower is better), how they behave close to failure, and whether they can explain themselves.

| Model | How it works | Typical miss (± cycles) | Explains itself? | Why (not) picked |
|---|---|---|---|---|
| Ridge | Fits one straight-line weight per sensor feature (a linear model). | ±21.0 | Yes — one signed weight per signal. | Reference floor the champion must beat (champion beats it by 2.8 cycles). |
| RandomForestRegressor | Averages 200 decision trees that each vote on remaining life. | ±18.2 | Yes — ranks which sensors drove the estimate. | Picked: most accurate self-explaining model; clears the floor and holds the incumbent contract. |
| HistGradientBoostingRegressor | Builds trees in sequence, each correcting the previous one's misses. | ±18.6 | Indirectly — needs a follow-up permutation test. | Not picked: not clearly better than the incumbent on both criteria; keeps determinism. |

## Technical comparison

| Model | CV-RMSE (mean ± std) | Low-RUL RMSE (true RUL<50) | Optimistic % (pred>true, low-RUL) | Role |
|---|---|---|---|---|
| Ridge | 21.01 ± 1.38 | 21.20 | 77.8% | floor |
| RandomForestRegressor ⬅ champion | 18.20 ± 0.52 | 17.50 | 72.0% | incumbent champion |
| HistGradientBoostingRegressor | 18.59 ± 0.61 | 17.51 | 66.0% | challenger |

Criteria in priority order: (1) grouped-CV RMSE, (2) end-of-life calibration (low-RUL RMSE + optimistic fraction), (3) simplicity tiebreak. Identical folds across candidates: **True** (fold signature `37b9d8bb5226988d`).

## Champion rationale

Champion: **RandomForestRegressor**. It posts a grouped-CV typical miss of 18.20 ± 0.52 cycles (5 folds, split by engine so no unit is scored on itself), and beats the Ridge floor by 2.81 cycles — the ensemble earns its complexity. The HistGradientBoostingRegressor challenger did NOT clear the clear-win bar (needs >1.0 cycles better than the RandomForest on BOTH criteria; observed gaps: CV-RMSE -0.39, end-of-life RMSE -0.01 — positive means RF is already ahead). Determinism and the unchanged downstream contract outweigh a fractional-cycle change, so the RandomForest stays champion (simplicity tiebreak favours the incumbent). End-of-life calibration (true RUL < 50): the champion's low-RUL RMSE is 17.50 cycles and it guesses too healthy on 72.0% of near-failure rows — the honest, watch-this caveat that the uncertainty note carries downstream.

## Guardrails

- **Floor gate:** the champion must beat the Ridge linear floor on grouped-CV RMSE, or the stage HALTs. A straight line matching the ensemble would mean the ensemble is unjustified.
- **Incumbent bias by design:** the RandomForest stays champion unless a challenger is clearly better (> 1.0 cycles on BOTH overall and end-of-life RMSE). Determinism and the fixed downstream prediction contract outrank fractional-cycle wins.
- **No leakage:** folds are grouped by engine; every preprocessing step (e.g. Ridge's standardiser) is fit inside the training fold only.
