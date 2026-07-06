# Failure Modes and Sensor Signatures

Generic degradation modes seen in turbomachinery prognostics, described in
terms of observable sensor behavior. These descriptions are educational and
dataset-agnostic; they are not tuned to any specific asset or manufacturer.

## High-Pressure Compressor (HPC) Degradation

HPC degradation is the dominant fault mode in the FD001 subset used by this
prototype. As compressor efficiency falls, more work is needed to reach the
same pressure ratio. Typical signatures: a gradual rise in compressor-outlet
and turbine-gas temperatures, a small drift in core speed to hold thrust, and
a slow decline in efficiency-related derived signals. The trend is monotonic
and accelerates near end-of-life rather than appearing as a single spike.

## Fan and Blade Wear

Fan or blade-path wear reduces airflow efficiency. Signatures include a slow
shift in fan speed relative to commanded thrust, changes in bypass-related
pressures and temperatures, and increased scatter around otherwise smooth
trends. Fan wear often co-occurs with compressor degradation, so the signals
should be read together rather than in isolation.

## Bearing Wear

Bearing wear is a mechanical degradation mode. In vibration-instrumented
systems it shows as rising vibration amplitude at characteristic frequencies;
in thermodynamic-only datasets like C-MAPSS it appears only indirectly through
temperature and speed drift. Because the FD001 data has no vibration channel,
bearing wear is described here for completeness, not as something this
prototype detects directly.

## Sensor Drift vs. True Degradation

Not every trend is asset degradation. A single channel that drifts while its
thermodynamically coupled channels stay flat suggests sensor drift or a
failing transducer, not engine wear. True degradation shows correlated
movement across several physically related sensors. Always corroborate before
attributing a trend to the asset — see anomaly_review_guidelines.
