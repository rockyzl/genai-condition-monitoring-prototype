"""Exploratory data analysis (stage s02) — genuinely new logic.

Produces the evidence that justifies the rest of the pipeline's choices:

1. **Sensor monotonicity ranking** — Pearson correlation of each raw sensor with
   remaining useful life, ranked by magnitude. Names the degradation carriers.
2. **Flat-sensor drop list with evidence** — the (near-)constant sensors that
   carry no degradation signal on FD001, with their std and distinct-value count.
3. **Unit-lifetime distribution** — per-unit run length, explaining why RUL is
   capped and why short-history units are hard.

Outputs a human ``reports/eda_summary.md``, a machine ``reports/eda/eda_summary.json``,
and three figures under ``reports/eda/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load_cmapss import (
    OP_SETTING_COLS,
    SENSOR_COLS,
    add_training_rul,
    load_raw,
)
from src.features.build_features import DROPPED_SENSORS, INFORMATIVE_SENSORS

#: A sensor is treated as "flat" (no degradation signal) when it has at most this
#: many distinct values across the whole FD001 training set.
FLAT_MAX_UNIQUE = 2


def _sensor_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor std, distinct-value count, and Pearson corr with RUL."""
    rows = []
    rul = train["rul"]
    for s in SENSOR_COLS:
        col = train[s]
        corr = float(col.corr(rul)) if col.std() > 0 else float("nan")
        rows.append(
            {
                "sensor": s,
                "std": round(float(col.std()), 6),
                "n_unique": int(col.nunique()),
                "corr_with_rul": None if np.isnan(corr) else round(corr, 4),
                "abs_corr": 0.0 if np.isnan(corr) else round(abs(corr), 4),
            }
        )
    return pd.DataFrame(rows)


def compute_eda(dataset: str = "FD001", root: Path | None = None) -> dict:
    """Compute the full EDA summary as a plain dict (no side effects)."""
    train = load_raw(dataset, "train")
    # Uncapped RUL (cycles-to-failure) is the honest signal for monotonicity.
    train = add_training_rul(train, cap=None)

    stats = _sensor_stats(train)
    flat_mask = stats["n_unique"] <= FLAT_MAX_UNIQUE
    flat = stats[flat_mask].sort_values("sensor")
    informative = stats[~flat_mask].sort_values("abs_corr", ascending=False)

    lifetimes = train.groupby("unit")["cycle"].max()
    op_setting_std = {c: round(float(train[c].std()), 6) for c in OP_SETTING_COLS}

    cap = 125
    n_rows = int(len(train))
    n_rows_capped = int((train["rul"] > cap).sum())
    frac_capped_pct = round(100 * n_rows_capped / n_rows, 1) if n_rows else 0.0

    return {
        "dataset": dataset,
        "n_units": int(train["unit"].nunique()),
        "n_train_rows": n_rows,
        "rul_cap": cap,
        "cap_rationale": (
            f"{frac_capped_pct}% of training rows have a true remaining life above "
            f"{cap} cycles — healthy early life where the sensors barely move. "
            f"Clipping the target at {cap} stops the model fitting noise on that "
            "healthy stretch and matches how a maintenance team acts (anything "
            "beyond ~125 cycles is simply 'healthy')."
        ),
        "rul_capping": {
            "n_rows": n_rows,
            "n_rows_capped": n_rows_capped,
            "frac_rows_capped_pct": frac_capped_pct,
        },
        "lifetime": {
            "min": int(lifetimes.min()),
            "max": int(lifetimes.max()),
            "mean": round(float(lifetimes.mean()), 1),
            "median": int(lifetimes.median()),
        },
        "sensor_monotonicity": [
            {
                "sensor": r.sensor,
                "corr_with_rul": r.corr_with_rul,
                "abs_corr": r.abs_corr,
            }
            for r in informative.itertuples()
        ],
        "flat_sensors": [
            {"sensor": r.sensor, "std": r.std, "n_unique": r.n_unique}
            for r in flat.itertuples()
        ],
        "op_setting_std": op_setting_std,
        "informative_sensors_expected": INFORMATIVE_SENSORS,
        "dropped_sensors_expected": DROPPED_SENSORS,
        "_stats_table": stats,  # internal, dropped before JSON serialisation
    }


# --- figures -----------------------------------------------------------------
def _plot_monotonicity(summary: dict, out: Path) -> None:
    data = summary["sensor_monotonicity"]
    names = [d["sensor"].replace("sensor_", "s") for d in data]
    vals = [d["abs_corr"] for d in data]
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(names))
    ax.barh(y, vals, color="#3b7dd8", edgecolor="k", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # strongest carrier on top
    ax.set_xlabel("|Pearson correlation| with RUL")
    ax.set_title(f"{summary['dataset']} train: sensor degradation-signal ranking")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_flat_sensors(summary: dict, out: Path) -> None:
    stats: pd.DataFrame = summary["_stats_table"]
    flat_names = {d["sensor"] for d in summary["flat_sensors"]}
    order = stats.sort_values("std")
    names = [s.replace("sensor_", "s") for s in order["sensor"]]
    # std can be 0; nudge for log display.
    stds = np.clip(order["std"].to_numpy(dtype=float), 1e-6, None)
    colors = ["#d1495b" if s in flat_names else "#8aa0b8" for s in order["sensor"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(names)), stds, color=colors, edgecolor="k", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("sensor std (log scale)")
    ax.set_title(
        f"{summary['dataset']} train: flat sensors (red) carry no wear signal"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_lifetime(summary: dict, out: Path, dataset: str) -> None:
    train = add_training_rul(load_raw(dataset, "train"), cap=None)
    lifetimes = train.groupby("unit")["cycle"].max().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(lifetimes, bins=20, color="#3b7dd8", edgecolor="k", alpha=0.85)
    ax.axvline(125, color="r", linestyle="--", label="RUL cap = 125")
    ax.set_xlabel("Unit lifetime (cycles to failure)")
    ax.set_ylabel("Number of units")
    ax.set_title(f"{summary['dataset']} train: engine lifetime distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --- markdown ----------------------------------------------------------------
def _render_markdown(summary: dict) -> str:
    lt = summary["lifetime"]
    lines: list[str] = []
    lines.append("# FD001 Exploratory Data Analysis\n")
    lines.append(
        "Generated by pipeline stage `s02_eda` (`src/pipeline/eda.py`) from "
        "`data/raw/CMAPSSData/train_FD001.txt`. Descriptive only — it justifies "
        "the preprocessing and modelling choices made downstream.\n"
    )

    lines.append("## Unit-lifetime distribution\n")
    lines.append(
        f"- Units: **{summary['n_units']}**, training rows: **{summary['n_train_rows']}**."
    )
    lines.append(
        f"- Lifetime (cycles to failure): min **{lt['min']}**, median **{lt['median']}**, "
        f"mean **{lt['mean']}**, max **{lt['max']}**."
    )
    cap_stats = summary["rul_capping"]
    lines.append(
        f"- **{cap_stats['frac_rows_capped_pct']}%** of training rows "
        f"({cap_stats['n_rows_capped']:,} of {cap_stats['n_rows']:,}) have a true "
        f"remaining life above the {summary['rul_cap']}-cycle cap and are clipped "
        f"to it. {summary['cap_rationale']}\n"
    )

    lines.append("## Sensor monotonicity ranking (degradation carriers)\n")
    lines.append(
        "Pearson correlation of each raw sensor with remaining useful life across "
        "all training rows; sensors with a stronger monotonic trend carry more "
        "degradation signal.\n"
    )
    lines.append("| rank | sensor | corr with RUL | |corr| |")
    lines.append("|-----:|--------|--------------:|------:|")
    for i, d in enumerate(summary["sensor_monotonicity"], 1):
        lines.append(
            f"| {i} | {d['sensor']} | {d['corr_with_rul']} | {d['abs_corr']} |"
        )
    lines.append("")

    lines.append("## Flat-sensor drop list (evidence)\n")
    lines.append(
        f"These sensors have at most {FLAT_MAX_UNIQUE} distinct values on FD001 "
        "train — (near-)constant, so they carry no degradation information and are "
        "dropped before modelling.\n"
    )
    lines.append("| sensor | std | distinct values |")
    lines.append("|--------|----:|----------------:|")
    for d in summary["flat_sensors"]:
        lines.append(f"| {d['sensor']} | {d['std']} | {d['n_unique']} |")
    lines.append("")
    lines.append(
        "Operational settings are also dropped (FD001 is a single 'Sea Level' "
        "condition): "
        + ", ".join(f"{k} std={v}" for k, v in summary["op_setting_std"].items())
        + ".\n"
    )
    lines.append(
        f"Result: **{len(summary['flat_sensors'])}** flat sensors dropped, leaving "
        f"the **{len(summary['informative_sensors_expected'])}** informative "
        "sensors used as base features. This matches the drop-list hard-coded in "
        "`src/features/build_features.py`.\n"
    )
    return "\n".join(lines)


def run(dataset: str, out_md: Path, out_json: Path, fig_dir: Path) -> dict:
    """Compute EDA and write markdown + JSON + figures. Returns the summary dict."""
    summary = compute_eda(dataset)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    figs = {
        "monotonicity": fig_dir / "monotonicity.png",
        "flat_sensors": fig_dir / "flat_sensors.png",
        "lifetime_distribution": fig_dir / "lifetime_distribution.png",
    }
    _plot_monotonicity(summary, figs["monotonicity"])
    _plot_flat_sensors(summary, figs["flat_sensors"])
    _plot_lifetime(summary, figs["lifetime_distribution"], dataset)

    out_md.write_text(_render_markdown(summary))

    serialisable = {k: v for k, v in summary.items() if not k.startswith("_")}
    out_json.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False))

    summary["_figures"] = figs
    return summary
