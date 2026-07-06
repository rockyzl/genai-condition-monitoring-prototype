# Interpreting RUL Estimates

Guidance on reading Remaining Useful Life (RUL) point estimates from the
baseline model. RUL is expressed in operational cycles remaining before
predicted failure.

## Point Estimates Are Not Guarantees

The model returns a single number — a point estimate — but the true RUL is a
distribution. A predicted RUL of 40 cycles means "around 40," not "exactly
40." Treat every point estimate as the center of a range, and widen that range
as degradation advances and the data becomes noisier.

## RUL Caps and Piecewise Targets

Early in life an asset's true RUL is large and hard to estimate, so a common
convention (used here) caps the training RUL target at a ceiling — for example
125 cycles. This piecewise-linear target keeps the model from chasing
meaningless large values while the asset is healthy and focuses its accuracy
on the region that matters, near end-of-life. Because of the cap, high RUL
predictions should be read as "healthy," not as a precise countdown.

## Risk Bands

This prototype maps RUL to three advisory risk bands:

- **High risk:** predicted RUL <= 30 cycles — near end-of-life, prioritize review.
- **Medium risk:** 30 < RUL <= 80 cycles — degrading, monitor closely.
- **Low risk:** RUL > 80 cycles — healthy, routine monitoring.

Bands are a communication aid, not a control setpoint.

## Why Point Estimates Mislead Near End-of-Life

Prediction error is not uniform. Models are typically most accurate mid-life
and least accurate exactly when it matters — the final cycles — because
degradation accelerates and failure dynamics grow noisier. A confident-looking
low RUL can still be optimistic. Near end-of-life, weight the uncertainty and
the direction of the trend more heavily than the exact number, and default to
human review.
