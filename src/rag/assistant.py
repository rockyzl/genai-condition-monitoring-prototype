"""Deterministic, grounded diagnostic assistant.

``diagnose(evidence, retriever)`` composes a maintenance-style diagnostic report
by *template*, not by generation: there is no LLM and no API call. Every
possible-failure-mode and next-step claim is a literal excerpt lifted from a
chunk the retriever surfaced, tagged with its ``source_file`` + ``section``
citation. If retrieval surfaces nothing relevant, the report says so instead of
inventing a cause.

The report always carries: the model's own numbers (predicted RUL, risk band),
supporting sensor/importance facts from the evidence record, an explicit
uncertainty statement, citations, a hard ``human_review_required`` flag, and a
fixed safety note framing the system as an advisory prototype on public data.
"""

from __future__ import annotations

import re

SAFETY_NOTE = (
    "Independent R&D prototype built on public NASA C-MAPSS simulation data. "
    "It does not make safety-critical decisions, does not command any action, "
    "and is not affiliated with any equipment manufacturer. Every output is "
    "advisory and must be reviewed by a qualified human before any maintenance "
    "action."
)

_BAND_READING = {
    "high": "near end-of-life for this model's convention; prioritize review",
    "medium": "degrading; monitor closely and plan a check",
    "low": "healthy under routine monitoring",
}

# Deterministic query vocabulary (expanded from the evidence record, not free
# text). These terms bias TF-IDF retrieval toward the relevant guidance without
# hardcoding which files or sections exist.
_FM_TERMS = (
    "failure mode fault wear degradation compressor turbine blade bearing "
    "efficiency signature"
)
_STEPS_TERMS = (
    "maintenance review checklist inspection schedule verify data quality "
    "cross-check sensors corroborate escalate human review next steps"
)


def normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces (for substring checks)."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _first_sentence(chunk_text: str, max_len: int = 300) -> str:
    """First sentence of a chunk's body (heading line dropped), whitespace-normalized.

    The result is a substring of the whitespace-normalized chunk text, which is
    what lets the eval harness verify the claim traces to retrieved content.
    """
    lines = chunk_text.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    body = normalize_ws(" ".join(lines))
    m = re.search(r"(.+?[.!?])(\s|$)", body)
    sentence = m.group(1) if m else body
    if len(sentence) > max_len:
        sentence = sentence[:max_len].rsplit(" ", 1)[0] + "…"
    return sentence.strip()


def _trend_phrases(sensor_summary: dict) -> tuple[list[str], list[str]]:
    increasing, decreasing = [], []
    for name, s in sensor_summary.items():
        if s.get("trend") == "increasing":
            increasing.append(name)
        elif s.get("trend") == "decreasing":
            decreasing.append(name)
    return increasing, decreasing


def _build_queries(evidence: dict) -> tuple[str, str]:
    risk = str(evidence.get("risk_band", "")).strip()
    inc, dec = _trend_phrases(evidence.get("sensor_summary", {}))
    dir_terms = []
    if inc:
        dir_terms.append("rising increasing trend")
    if dec:
        dir_terms.append("declining decreasing trend")
    dir_str = " ".join(dir_terms)
    fm_query = f"{risk} risk {dir_str} {_FM_TERMS}".strip()
    steps_query = f"{risk} risk {_STEPS_TERMS}".strip()
    return fm_query, steps_query


def _supporting_evidence(evidence: dict) -> list[str]:
    facts: list[str] = []
    for name, s in evidence.get("sensor_summary", {}).items():
        facts.append(
            f"{name}: mean {s['mean']}, std {s['std']}, {s['trend']} trend over "
            f"the last {s.get('window_cycles', '?')} cycles."
        )
    signals = evidence.get("top_contributing_signals", [])
    if signals:
        rendered = ", ".join(
            f"{sig['feature']} ({sig['importance']})" for sig in signals
        )
        facts.append(f"Top model signals by importance: {rendered}.")
    return facts


def diagnose(evidence: dict, retriever) -> dict:
    """Produce a grounded, cited diagnostic report for one evidence record."""
    asset_id = evidence.get("asset_id")
    last_cycle = evidence.get("last_cycle")
    pred_rul = evidence.get("predicted_rul")
    risk_band = str(evidence.get("risk_band", "unknown"))
    model = evidence.get("model", "baseline model")

    fm_query, steps_query = _build_queries(evidence)
    fm_hits = retriever.retrieve(fm_query, k=4)
    steps_hits = retriever.retrieve(steps_query, k=4)

    # --- Summary (must reference the actual predicted RUL number) ----------
    band_reading = _BAND_READING.get(risk_band, "review required")
    summary = (
        f"Asset {asset_id} at cycle {last_cycle}: the {model} predicts a "
        f"remaining useful life of about {pred_rul} cycles, placing this unit in "
        f"the {risk_band}-risk band ({band_reading}). This is an advisory "
        f"estimate; the sections below trace the reading to retrieved guidance "
        f"and flag its uncertainty."
    )

    # --- Possible failure modes (ONLY from retrieval) ----------------------
    possible_failure_modes: list[dict] = []
    for hit in fm_hits[:3]:
        possible_failure_modes.append(
            {
                "failure_mode": hit["section"],
                "evidence": _first_sentence(hit["text"]),
                "source_file": hit["source_file"],
                "section": hit["section"],
                "retrieval_score": hit["score"],
            }
        )
    if not possible_failure_modes:
        possible_failure_modes.append(
            {
                "failure_mode": "none retrieved",
                "evidence": (
                    "No matching failure-mode guidance was retrieved from the "
                    "knowledge base; a qualified reviewer must diagnose this unit "
                    "manually."
                ),
                "source_file": None,
                "section": None,
            }
        )

    # --- Recommended next steps (ONLY from retrieval) ----------------------
    recommended_next_steps: list[dict] = []
    for hit in steps_hits[:3]:
        recommended_next_steps.append(
            {
                "step": hit["section"],
                "detail": _first_sentence(hit["text"]),
                "source_file": hit["source_file"],
                "section": hit["section"],
                "retrieval_score": hit["score"],
            }
        )
    if not recommended_next_steps:
        recommended_next_steps.append(
            {
                "step": "none retrieved",
                "detail": (
                    "No matching maintenance-checklist guidance was retrieved; "
                    "defer to standard review procedure and human judgment."
                ),
                "source_file": None,
                "section": None,
            }
        )

    # --- Citations (deduped union of everything actually used) -------------
    citations: list[dict] = []
    seen = set()
    for item in possible_failure_modes + recommended_next_steps:
        sf, sec = item.get("source_file"), item.get("section")
        if sf and (sf, sec) not in seen:
            seen.add((sf, sec))
            citations.append({"source_file": sf, "section": sec})

    return {
        "asset_id": asset_id,
        "summary": summary,
        "supporting_evidence": _supporting_evidence(evidence),
        "possible_failure_modes": possible_failure_modes,
        "recommended_next_steps": recommended_next_steps,
        "uncertainty": evidence.get(
            "uncertainty_note",
            "This is a point estimate without a calibrated confidence interval; "
            "treat it as decision support only.",
        ),
        "citations": citations,
        "human_review_required": True,
        "safety_note": SAFETY_NOTE,
    }
