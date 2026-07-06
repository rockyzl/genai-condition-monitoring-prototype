"""Feature engineering for the C-MAPSS RUL baseline.

Deliberately plain and documented — the goal of this prototype is a credible,
readable baseline, not a leaderboard model.

Two steps:

1. **Drop uninformative signals.** On FD001 several sensors and operational
   settings are constant or near-constant (single operating condition, single
   fault mode), so they carry no degradation information. We drop them:

   - Constant sensors (std == 0 on FD001 train): 1, 5, 10, 16, 18, 19
   - Near-constant sensor 6 (only two distinct values, std ~1e-3)
   - All three operational settings (FD001 is a single "Sea Level" condition,
     so op_setting_1/2 vary only by sensor noise and op_setting_3 is constant)

   That leaves the 14 informative sensors that are standard in the FD001
   literature: 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21.

2. **Add short rolling statistics.** Per unit, a rolling mean and rolling std
   (window 5) of each informative sensor. The mean denoises the raw signal; the
   std captures how volatile a sensor has become, which tends to rise as a
   fault develops. ``min_periods=1`` so early cycles are not dropped, and the
   rolling std of a length-1 window (NaN) is filled with 0.

The same :func:`build_features` function is used for train and test so the
feature columns line up exactly.
"""

from __future__ import annotations

import pandas as pd

# Sensors dropped as constant / near-constant on FD001 (see module docstring).
DROP_SENSOR_IDS = [1, 5, 6, 10, 16, 18, 19]
DROPPED_SENSORS = [f"sensor_{i}" for i in DROP_SENSOR_IDS]

# The 14 informative sensors used as base features.
INFORMATIVE_SENSOR_IDS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
INFORMATIVE_SENSORS = [f"sensor_{i}" for i in INFORMATIVE_SENSOR_IDS]

#: Rolling-window length (in cycles) for the smoothing/volatility features.
ROLLING_WINDOW = 5


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Add per-unit rolling mean and std columns for the given sensors.

    Returns a new DataFrame with ``{sensor}_roll_mean`` and ``{sensor}_roll_std``
    columns appended. The input is not mutated. Rolling windows are computed
    within each unit so history never leaks across engines.
    """
    sensors = sensors if sensors is not None else INFORMATIVE_SENSORS
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")[sensors]
    roll = grouped.rolling(window=window, min_periods=1)
    roll_mean = roll.mean().reset_index(level=0, drop=True)
    roll_std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
    roll_mean.columns = [f"{s}_roll_mean" for s in sensors]
    roll_std.columns = [f"{s}_roll_std" for s in sensors]
    return pd.concat([df, roll_mean, roll_std], axis=1)


def feature_columns(sensors: list[str] | None = None) -> list[str]:
    """Return the ordered list of feature column names produced by
    :func:`build_features` (base sensor value + rolling mean + rolling std)."""
    sensors = sensors if sensors is not None else INFORMATIVE_SENSORS
    cols: list[str] = []
    for s in sensors:
        cols.extend([s, f"{s}_roll_mean", f"{s}_roll_std"])
    return cols


def build_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    window: int = ROLLING_WINDOW,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the modelling feature matrix from a raw (named-column) frame.

    Parameters
    ----------
    df:
        Raw frame from :func:`src.data.load_cmapss.load_raw` (must contain
        ``unit``, ``cycle`` and the sensor columns). May already carry a ``rul``
        column, which is preserved.

    Returns
    -------
    (frame, feature_cols)
        ``frame`` is the input plus rolling-feature columns (still one row per
        cycle, sorted by unit then cycle). ``feature_cols`` is the ordered list
        of columns to feed the model.
    """
    sensors = sensors if sensors is not None else INFORMATIVE_SENSORS
    frame = add_rolling_features(df, sensors=sensors, window=window)
    return frame, feature_columns(sensors)
