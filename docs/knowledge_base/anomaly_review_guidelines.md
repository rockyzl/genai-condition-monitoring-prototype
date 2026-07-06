# Anomaly Review Guidelines

How to review an anomaly or a high-risk flag and separate genuine asset
degradation from sensor artifacts. Applies whether the trigger is an anomaly
score or a low RUL estimate.

## Sensor Fault vs. Asset Degradation

The first question on any flag is: is the asset degrading, or is a sensor
lying? A sensor fault typically affects one channel, appears abruptly, and is
not physically consistent with related signals. Asset degradation shows up as
a coordinated shift across thermodynamically linked sensors, consistent with a
known failure mode. When in doubt, trust corroboration over any single channel.

## Trend vs. Spike

Distinguish sustained trends from transient spikes. A short excursion that
returns to baseline is usually an operational transient, noise, or a momentary
sensor glitch — not degradation. A trend that persists and grows over many
cycles, especially one that accelerates, is the signature of real wear.
Persistence and direction matter more than any single reading's magnitude.

## Corroboration Across Sensors

Real degradation obeys physics: if efficiency drops, several coupled signals
move together in a predictable way. Before accepting a flag, confirm that the
implicated sensors move in a physically coherent pattern. If only one sensor
moves while its physically coupled neighbors stay flat, suspect
instrumentation, not the asset.

## Operating-Condition Context

Apparent anomalies can be artifacts of a change in operating condition rather
than degradation. Where operational settings are available, normalize or
condition on them before judging a shift. In the single-condition FD001 subset
this risk is small, but in multi-condition data it is a leading cause of false
alarms.

## When to Escalate

If the evidence is ambiguous — mixed signals, a borderline trend, or a flag
near end-of-life — escalate for human review rather than resolving it
automatically. Record what was and was not corroborated so the next reviewer
starts with context.
