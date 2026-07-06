# Maintenance Review Checklist

A practical, ordered checklist a reviewer follows when a unit is flagged
high-risk by the model. This is advisory workflow guidance, not a certified
maintenance procedure.

## Step 1 — Verify Data Quality

Before trusting any flag, confirm the input is sound. Check for missing cycles,
frozen or railed sensor values, out-of-range readings, and duplicated
timestamps. A high-risk score built on bad data is a data problem, not a
maintenance event. Reject or quarantine the window if data quality fails.

## Step 2 — Cross-Check Sensors

Confirm that the flag is supported by more than one physically related sensor.
Look for a coherent degradation pattern — correlated temperature, pressure,
and speed drift — rather than a single moving channel. A lone moving sensor
points to a sensor fault; corroborated movement points to real degradation.

## Step 3 — Review Trend and History

Compare the current window against the unit's own history and against the
fleet. Is the trend monotonic and accelerating, or a transient excursion?
Short spikes that recover are usually not end-of-life degradation. Note how
many cycles the trend has persisted.

## Step 4 — Schedule Inspection

If the data is clean and the degradation is corroborated, schedule a physical
inspection proportional to the risk band. High-risk (short predicted RUL)
warrants prompt inspection; medium-risk warrants closer monitoring and a
planned check. Record the predicted RUL and the risk band that triggered the
action.

## Step 5 — Escalate When Uncertain

If the model output and the sensor evidence disagree, or if uncertainty is
high near end-of-life, escalate to a senior reviewer rather than acting on the
score alone. Document the disagreement. The model is a decision aid; the human
owns the maintenance decision. See human_in_loop_policy.
