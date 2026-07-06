# Data Sources

## Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation

This prototype uses the **C-MAPSS** (Commercial Modular Aero-Propulsion System
Simulation) run-to-failure dataset published by the **NASA Prognostics Center
of Excellence (PCoE)**. It is a widely used public benchmark for Remaining
Useful Life (RUL) prognostics.

- **Provider:** NASA Ames Prognostics Center of Excellence (PCoE) Data Repository.
- **Local copy:** `data/raw/CMAPSSData/` (train/test/RUL text files for
  FD001–FD004 plus the original `readme.txt` and the damage-propagation paper).
- **This prototype uses FD001 only.**

### Citation

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling
> for Aircraft Engine Run-to-Failure Simulation," in *Proceedings of the 1st
> International Conference on Prognostics and Health Management (PHM08)*,
> Denver, CO, Oct. 2008.

Dataset published via the NASA PCoE Data Repository (Prognostics Data
Repository), NASA Ames Research Center. The included PDF
(`Damage Propagation Modeling.pdf`) is the source paper.

## Asset type: simulated turbofan engines

The data is **simulated**, not measured on physical hardware. C-MAPSS is a
high-fidelity thermodynamic model of a commercial turbofan engine. Each "unit"
is one simulated engine flown to failure under an injected fault that grows over
time. A dataset is a fleet of same-type engines, each starting with a different
(unknown) degree of initial wear and manufacturing variation — treated as
normal, not a fault — with sensor noise added on top.

FD001 specifics (from the dataset `readme.txt`):

- **Operating conditions:** ONE (Sea Level).
- **Fault modes:** ONE (High-Pressure Compressor / HPC degradation).
- **Train trajectories:** 100 engines run to failure.
- **Test trajectories:** 100 engines, each truncated some time before failure.

## Columns

Each row is one operational cycle (a flight) of one engine. 26 space-separated
columns, no header:

| # | column | meaning |
|--:|--------|---------|
| 1 | `unit` | engine id (1..100) |
| 2 | `cycle` | operational cycle index (time), starts at 1 |
| 3–5 | `op_setting_1..3` | three operational settings that affect performance |
| 6–26 | `sensor_1..21` | 21 simulated sensor measurements |

The 21 sensors correspond to standard turbofan measurements (temperatures,
pressures, fan/core speeds, fuel flow, bleed, coolant flows, etc.). The dataset
does not ship exact per-sensor engineering labels, and this prototype does not
invent them — sensors are referred to by index. On FD001, the following are
constant or near-constant and are dropped as non-informative:
`sensor_1, 5, 6, 10, 16, 18, 19` and all three operational settings (single
operating condition). The 14 informative sensors used as features are
`sensor_2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`. See
`src/features/build_features.py`.

## Train / test split

The split is provided by the dataset authors, not created here:

- **`train_FD001.txt`** — 100 engines, each run **all the way to failure**. The
  last cycle of each engine is the failure point.
- **`test_FD001.txt`** — 100 engines, each time series **truncated before
  failure**. The task is to predict how many cycles remain after the last
  observed cycle.
- **`RUL_FD001.txt`** — the ground-truth remaining cycles for each of the 100
  test engines, one value per line in unit order. Used only for evaluation.

## Target variable

**Remaining Useful Life (RUL)** — the number of operational cycles an engine
will keep running after the current cycle.

- On **training** data, RUL is derived as `max_cycle_for_unit − current_cycle`,
  then **capped at 125 cycles** (piecewise-linear target). Rationale: early in
  an engine's life the true remaining life is large but barely reflected in the
  sensors; regressing against very large RUL values teaches the model to fit
  noise. The cap encodes "beyond ~125 cycles of remaining life, just call it
  healthy," which both matches maintenance practice and is the standard C-MAPSS
  convention. The cap is a documented modelling choice
  (`src/data/load_cmapss.py`), configurable, and reported alongside uncapped
  metrics so nothing is hidden.
- Evaluation compares predicted RUL at each test engine's last cycle against
  `RUL_FD001.txt`, reporting metrics against **both** the capped truth (the
  target the model was trained on) and the raw uncapped truth.

## Limitations of this data

- **Simulated, not real.** No physical sensor drift, calibration issues,
  missing data, communication dropouts, or maintenance interventions that real
  fleet telemetry has.
- **Idealised failure.** A single fault mode grows monotonically to failure;
  real assets show intermittent faults, repairs, and multiple interacting
  degradation paths.
- **Single condition (FD001).** One operating regime and one fault mode. FD002
  and FD004 add six operating conditions and are substantially harder; results
  here will not transfer without condition-aware normalisation.
- **No true timestamps or units.** "Cycle" is an abstract index; sensors are
  unlabelled indices without physical engineering units.
- **Clean labels.** Exact run-to-failure ground truth is a luxury real
  maintenance datasets rarely have.

## Why this is a *proxy* for condition monitoring — not Caterpillar data

This is an **independent R&D prototype built on public NASA aircraft-engine
simulation data**. It is **not** Caterpillar data, not heavy-equipment data,
not production telemetry, and not a digital twin of any real asset. It contains
no CAN/J1939 signals and no field-collected measurements of any kind.

It is used here as a *transferable proxy* for the condition-monitoring problem
shape: multivariate sensor time series from a fleet of same-type assets, each
degrading toward failure, where the job is to turn raw signals into a
calibrated remaining-life estimate, a risk band, and an evidence-backed,
human-reviewable diagnostic. That end-to-end workflow — data → RUL/anomaly
modelling → structured evidence → retrieval-grounded diagnostic assistant →
evaluation and human-in-the-loop governance — is what the prototype
demonstrates. The *methods* (feature engineering on degrading sensors, capped
RUL regression, error analysis, risk banding, evaluation discipline) carry over
to industrial equipment; the *specific model and numbers* are valid only for
simulated turbofans and would have to be re-derived on real Caterpillar
equipment data. No claim is made that this prototype has been validated on, or
is representative of, any real heavy-equipment fleet.
