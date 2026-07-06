"""Streamlit demo for the GenAI-assisted condition-monitoring prototype.

Written for a non-expert reader (a hiring manager, or a quick skim): a one-line
"what is this", a plain-language walk-through of the pipeline, an engine picker,
a labelled sensor-trend plot, the model's Remaining-Useful-Life estimate with a
colour-coded risk badge, and a formatted (never raw-JSON) diagnostic report whose
every claim is quoted from a cited reference document.

Run:  .venv/bin/streamlit run src/app/streamlit_app.py

The UI body is guarded under ``if __name__ == "__main__"`` so the module imports
cleanly (no Streamlit side effects) for testing; Streamlit executes the file
with ``__name__ == "__main__"``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- make `src` importable when run via `streamlit run` -------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import build_evidence as be  # noqa: E402
from src.rag.assistant import diagnose  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402

LIMITATIONS_BANNER = (
    "Independent R&D prototype on public NASA C-MAPSS turbofan data. "
    "Not production-validated, not affiliated with any equipment manufacturer, "
    "and not a source of safety-critical decisions. Every output is advisory and "
    "requires review by a qualified human."
)

WHAT_IS_THIS = (
    "This demo reads an aircraft engine's recent sensor history, estimates how "
    "much longer it can safely run, and writes a plain-English maintenance note "
    "in which every stated cause and next step is quoted from a cited reference "
    "document — it never invents a diagnosis."
)

# Risk band -> (badge colour, label, plain-English meaning). Same thresholds the
# model uses: High <= 30 cycles, Medium 30-80, Low > 80.
RISK_BADGE = {
    "high": ("#c0392b", "HIGH", "schedule inspection soon — near end-of-life"),
    "medium": ("#e67e22", "MEDIUM", "degrading — monitor closely"),
    "low": ("#27ae60", "LOW", "healthy — keep monitoring"),
}

# Physical meaning of C-MAPSS sensor columns. Source: Saxena et al., "Damage
# Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," PHM08
# (Table 1) — the paper cited in data/raw/CMAPSSData/readme.txt. The readme
# itself labels these columns only as "sensor measurement 1-21"; the physical
# names below are the community-standard interpretation from that reference, not
# printed in the readme. (short label used on plots, longer description in table)
SENSOR_META = {
    "sensor_1": ("T2", "Fan inlet total temperature (°R)"),
    "sensor_2": ("T24", "LPC outlet total temperature (°R)"),
    "sensor_3": ("T30", "HPC outlet total temperature (°R)"),
    "sensor_4": ("T50", "LPT outlet total temperature (°R)"),
    "sensor_5": ("P2", "Fan inlet pressure (psia)"),
    "sensor_6": ("P15", "Bypass-duct total pressure (psia)"),
    "sensor_7": ("P30", "HPC outlet total pressure (psia)"),
    "sensor_8": ("Nf", "Physical fan speed (rpm)"),
    "sensor_9": ("Nc", "Physical core speed (rpm)"),
    "sensor_10": ("epr", "Engine pressure ratio P50/P2"),
    "sensor_11": ("Ps30", "HPC outlet static pressure (psia)"),
    "sensor_12": ("phi", "Fuel flow to Ps30 ratio (pps/psi)"),
    "sensor_13": ("NRf", "Corrected fan speed (rpm)"),
    "sensor_14": ("NRc", "Corrected core speed (rpm)"),
    "sensor_15": ("BPR", "Bypass ratio"),
    "sensor_16": ("farB", "Burner fuel-air ratio"),
    "sensor_17": ("htBleed", "Bleed enthalpy"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed (rpm)"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed (rpm)"),
    "sensor_20": ("W31", "HPT coolant bleed (lbm/s) — interpretation varies across sources"),
    "sensor_21": ("W32", "LPT coolant bleed (lbm/s) — interpretation varies across sources"),
}

_TREND_ARROW = {"increasing": "↑", "decreasing": "↓", "flat": "→"}


def _short_label(sensor_col: str) -> str:
    meta = SENSOR_META.get(sensor_col)
    return f"{meta[0]}" if meta else sensor_col


def _load_evidence(unit_id: int) -> dict | None:
    """Return the evidence record for a unit, building all records on demand if
    the DS artifacts exist but evidence hasn't been generated yet."""
    path = be.EVIDENCE_DIR / f"unit_{unit_id}.json"
    if not path.exists():
        if be.PRED_PATH.exists() and be.FI_PATH.exists() and be.RAW_TEST_PATH.exists():
            be.run()
        if not path.exists():
            return None
    return json.loads(path.read_text())


def _run() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="GenAI Condition Monitoring — Prototype", layout="wide"
    )

    @st.cache_resource
    def get_retriever() -> Retriever:
        return Retriever(be.KB_DIR)

    @st.cache_data
    def get_predictions() -> "pd.DataFrame":
        return pd.read_csv(be.PRED_PATH)

    @st.cache_data
    def get_raw() -> "pd.DataFrame":
        return be.load_raw_test()

    # ======================================================================
    # Header: what this is, the fixed limitations banner, how to read it
    # ======================================================================
    st.title("Predicting When an Engine Needs Maintenance")
    st.markdown(f"#### {WHAT_IS_THIS}")
    st.warning(LIMITATIONS_BANNER, icon="⚠️")

    with st.expander("How to read this page (30-second guide)", expanded=False):
        st.markdown(
            "This prototype runs a **4-step pipeline**. You are looking at the "
            "output of all four for one engine:\n\n"
            "1. **Sensors** — we take the engine's recent readings (temperatures, "
            "pressures, speeds) from its last flight cycles.\n"
            "2. **Prediction** — a simple machine-learning model estimates the "
            "engine's *Remaining Useful Life*: how many more flight cycles it can "
            "run before maintenance-critical wear.\n"
            "3. **Retrieved guidance** — the system searches a small library of "
            "maintenance and engineering notes for passages relevant to this "
            "engine's condition.\n"
            "4. **Cited report** — it writes a short diagnosis, and **every** "
            "possible cause or next step it lists is quoted from those notes with "
            "a citation. If it finds nothing relevant, it says so rather than "
            "guessing. A qualified human always makes the final call."
        )

    if not be.PRED_PATH.exists():
        st.error(
            f"Predictions artifact not found at {be.PRED_PATH}. Run the "
            "data/model pipeline (Phases 2-3) first."
        )
        st.stop()

    preds = get_predictions()
    raw = get_raw()
    retriever = get_retriever()

    # ======================================================================
    # Sidebar controls
    # ======================================================================
    st.sidebar.header("Pick an engine")
    st.sidebar.caption(
        "Each engine is one test unit from the NASA turbofan fleet. Pick one to "
        "see its sensors, its predicted remaining life, and its diagnosis."
    )
    unit_ids = sorted(preds["unit_id"].astype(int).tolist())
    unit_id = st.sidebar.selectbox("Engine (test unit) id", unit_ids)
    window = st.sidebar.slider(
        "How many recent cycles to plot", 10, 60, be.LAST_WINDOW
    )
    if len(retriever) == 0:
        st.sidebar.info("Knowledge base is empty — diagnostic guidance unavailable.")

    prow = preds[preds["unit_id"] == int(unit_id)].iloc[0]
    evidence = _load_evidence(int(unit_id))
    band = str(prow["risk_band"]).lower()

    # ======================================================================
    # Model output: RUL + risk badge
    # ======================================================================
    st.header(f"Engine {unit_id} — health at a glance")
    c1, c2 = st.columns(2)
    c1.metric("Flight cycles flown so far", int(prow["last_cycle"]))
    c2.metric("Predicted Remaining Useful Life", f"{round(float(prow['pred_rul']))} cycles")
    st.caption(
        "**Remaining Useful Life (RUL)** — roughly how many flight cycles this "
        "engine has left before maintenance-critical degradation. Higher is "
        "healthier. (The true RUL is held out for scoring only and is never shown "
        "to the model as an input.)"
    )

    color, label, meaning = RISK_BADGE.get(
        band, ("#7f8c8d", "UNKNOWN", "review required")
    )
    st.markdown(
        f"<div style='margin:0.4rem 0 0.2rem 0'>"
        f"<span style='background:{color};color:#fff;padding:4px 12px;"
        f"border-radius:6px;font-weight:700;font-size:0.95rem'>{label} RISK</span>"
        f"&nbsp;&nbsp;{meaning}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Risk band comes straight from the predicted RUL: High ≤ 30 cycles · "
        "Medium 30–80 · Low > 80. It is an advisory triage signal, not a control "
        "setpoint."
    )

    # ======================================================================
    # Sensor trends
    # ======================================================================
    st.header("What the sensors are doing")
    unit_raw = raw[raw["unit"] == int(unit_id)].sort_values("cycle").tail(window)
    if evidence and evidence.get("sensor_summary"):
        sensors = list(evidence["sensor_summary"].keys())
        trends = {k: v.get("trend") for k, v in evidence["sensor_summary"].items()}
    else:
        sensors = be.rank_sensors_from_importances(
            be.load_feature_importances(), be.TOP_K_SENSORS
        )
        trends = {}
    sensors = [s for s in sensors if s in unit_raw.columns][:6]

    if sensors and not unit_raw.empty:
        ncol = 2
        nrow = (len(sensors) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.3 * nrow), squeeze=False)
        for i, s in enumerate(sensors):
            ax = axes[i // ncol][i % ncol]
            ax.plot(unit_raw["cycle"], unit_raw[s], marker=".", linewidth=1)
            arrow = _TREND_ARROW.get(trends.get(s), "")
            ax.set_title(f"{_short_label(s)}  {arrow}", fontsize=9)
            ax.set_xlabel("flight cycle", fontsize=7)
            ax.tick_params(labelsize=7)
        for j in range(len(sensors), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Each panel is one of the model's most-informative sensors over this "
            "engine's most recent cycles; the arrow shows its overall direction "
            "(↑ rising, ↓ falling, → flat). A **sustained drift that several "
            "physically-related sensors share** is the fingerprint of real "
            "degradation as the engine nears end-of-life; a single lone jump is "
            "more likely sensor noise."
        )

        with st.expander("What do these sensors measure?"):
            table = pd.DataFrame(
                [
                    {
                        "Sensor": s,
                        "Symbol": SENSOR_META.get(s, ("?", ""))[0],
                        "What it measures": SENSOR_META.get(s, ("", "—"))[1],
                    }
                    for s in sensors
                ]
            )
            st.table(table)
            st.caption(
                "Physical meanings are from Saxena et al., *Damage Propagation "
                "Modeling for Aircraft Engine Run-to-Failure Simulation* (PHM08, "
                "Table 1) — the paper cited in the dataset readme. The dataset's "
                "own `readme.txt` labels these columns only as 'sensor measurement "
                "1–21'; the names here are the community-standard reading of that "
                "reference, not printed in the readme."
            )
    else:
        st.info("No sensor data available for this engine.")

    # ======================================================================
    # Diagnostic report (formatted, never raw JSON)
    # ======================================================================
    st.header("Diagnostic report")
    if not evidence:
        st.info(
            "No evidence record for this engine. Ensure the DS artifacts exist and "
            "run `src/diagnostics/build_evidence.py`."
        )
        return

    report = diagnose(evidence, retriever)

    st.markdown("**In plain English**")
    st.write(report["summary"])

    st.markdown("**What the data shows**")
    for item in report["supporting_evidence"]:
        st.markdown(f"- {item}")

    st.markdown("**Possible failure modes** (quoted from the reference library)")
    for fm in report["possible_failure_modes"]:
        if fm.get("source_file"):
            st.markdown(f"> **{fm['failure_mode']}.** {fm['evidence']}")
            st.caption(f"📄 {fm['source_file']} · {fm['section']}")
        else:
            st.info(fm["evidence"])

    st.markdown("**Recommended next steps**")
    for stp in report["recommended_next_steps"]:
        if stp.get("source_file"):
            st.markdown(f"☐ **{stp['step']}** — {stp['detail']}")
            st.caption(f"📄 {stp['source_file']} · {stp['section']}")
        else:
            st.info(stp["detail"])

    st.info(f"**Uncertainty.** {report['uncertainty']}", icon="ℹ️")

    if report["human_review_required"]:
        st.warning(
            "Human review required — a qualified engineer must confirm this before "
            "any maintenance action. This tool focuses attention; it does not "
            "decide.",
            icon="🧑‍🔧",
        )

    with st.expander("Sources cited in this report"):
        if report["citations"]:
            for cite in report["citations"]:
                st.markdown(f"- `{cite['source_file']}` · **{cite['section']}**")
        else:
            st.markdown("_No relevant sources were retrieved for this engine._")
        st.caption(
            "Every failure mode and next step above is a verbatim quote from one "
            "of these knowledge-base sections — the assistant composes, it does "
            "not invent."
        )

    st.caption(report["safety_note"])


if __name__ == "__main__":
    _run()
