"""Streamlit demo for the GenAI-assisted condition-monitoring prototype.

Select a test unit and see: a sensor trend plot for the top signals, the
model's RUL / risk-band output, the retrieved knowledge-base snippets, the
grounded diagnostic report, and a fixed limitations banner.

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

    st.title("GenAI-Assisted Condition Monitoring — Diagnostic Prototype")
    st.warning(LIMITATIONS_BANNER, icon="⚠️")

    if not be.PRED_PATH.exists():
        st.error(
            f"Predictions artifact not found at {be.PRED_PATH}. Run the "
            "data/model pipeline (Phases 2-3) first."
        )
        st.stop()

    preds = get_predictions()
    raw = get_raw()
    retriever = get_retriever()

    # --- Sidebar controls --------------------------------------------------
    st.sidebar.header("Select asset")
    unit_ids = sorted(preds["unit_id"].astype(int).tolist())
    unit_id = st.sidebar.selectbox("Test unit (asset) id", unit_ids)
    window = st.sidebar.slider("Trend window (cycles)", 10, 60, be.LAST_WINDOW)
    if len(retriever) == 0:
        st.sidebar.info("Knowledge base is empty — diagnostic guidance unavailable.")

    prow = preds[preds["unit_id"] == unit_id].iloc[0]
    evidence = _load_evidence(int(unit_id))

    # --- Model output ------------------------------------------------------
    st.subheader(f"Model output — asset {unit_id}")
    band = str(prow["risk_band"])
    color = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(band, "⚪")
    c1, c2, c3 = st.columns(3)
    c1.metric("Last cycle", int(prow["last_cycle"]))
    c2.metric("Predicted RUL (cycles)", round(float(prow["pred_rul"]), 2))
    c3.metric("Risk band", f"{color} {band}")
    st.caption(
        "true RUL is held out for evaluation only and is not shown here as a "
        "model input."
    )

    # --- Sensor trend plot -------------------------------------------------
    st.subheader("Sensor trends (top contributing signals)")
    unit_raw = raw[raw["unit"] == int(unit_id)].sort_values("cycle").tail(window)
    if evidence and evidence.get("sensor_summary"):
        sensors = list(evidence["sensor_summary"].keys())
    else:
        sensors = be.rank_sensors_from_importances(
            be.load_feature_importances(), be.TOP_K_SENSORS
        )
    sensors = [s for s in sensors if s in unit_raw.columns][:6]
    if sensors and not unit_raw.empty:
        ncol = 2
        nrow = (len(sensors) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.2 * nrow), squeeze=False)
        for i, s in enumerate(sensors):
            ax = axes[i // ncol][i % ncol]
            ax.plot(unit_raw["cycle"], unit_raw[s], marker=".", linewidth=1)
            ax.set_title(s, fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(len(sensors), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No sensor data available for this unit.")

    # --- Diagnostic report -------------------------------------------------
    st.subheader("Diagnostic assistant report")
    if not evidence:
        st.info(
            "No evidence record for this unit. Ensure the DS artifacts exist and "
            "run `src/diagnostics/build_evidence.py`."
        )
        return

    report = diagnose(evidence, retriever)
    st.markdown(f"**Summary.** {report['summary']}")

    st.markdown("**Supporting evidence (from the model + sensors):**")
    for item in report["supporting_evidence"]:
        st.markdown(f"- {item}")

    st.markdown("**Possible failure modes (retrieved, cited):**")
    for fm in report["possible_failure_modes"]:
        src = (
            f"  \n  ↳ _source: {fm['source_file']} → {fm['section']}_"
            if fm.get("source_file")
            else ""
        )
        st.markdown(f"- **{fm['failure_mode']}** — {fm['evidence']}{src}")

    st.markdown("**Recommended next steps (retrieved, cited):**")
    for stp in report["recommended_next_steps"]:
        src = (
            f"  \n  ↳ _source: {stp['source_file']} → {stp['section']}_"
            if stp.get("source_file")
            else ""
        )
        st.markdown(f"- **{stp['step']}** — {stp['detail']}{src}")

    st.markdown(f"**Uncertainty.** {report['uncertainty']}")

    with st.expander("Retrieved knowledge-base snippets (citations)"):
        if report["citations"]:
            for cite in report["citations"]:
                st.markdown(f"- `{cite['source_file']}` → **{cite['section']}**")
        else:
            st.markdown("_No relevant snippets retrieved._")

    if report["human_review_required"]:
        st.error("Human review required before any maintenance action.", icon="🧑‍🔧")
    st.caption(report["safety_note"])


if __name__ == "__main__":
    _run()
