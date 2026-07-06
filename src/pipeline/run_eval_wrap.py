"""Thin wrapper exposing the eval harness (stage s10) as a ``run() -> dict``.

Calls the existing three-section harness in :mod:`src.eval.run_eval` (model
metrics, retrieval hit@k, diagnostic-output governance), writes
``reports/evaluation_summary.md`` exactly as the standalone CLI does, and returns
a compact status/metrics dict for the pipeline journal and manifest. No
behaviour change — the harness itself is untouched.
"""

from __future__ import annotations

from src.eval import run_eval


def run() -> dict:
    metrics = run_eval.eval_model_metrics()
    retrieval = run_eval.eval_retrieval()
    diagnostics = run_eval.eval_diagnostics()
    run_eval.write_summary(metrics, retrieval, diagnostics)

    out = {
        "model_status": metrics["status"],
        "retrieval_status": retrieval["status"],
        "diagnostics_status": diagnostics["status"],
    }
    if metrics["status"] == "ok":
        out["n_units"] = metrics["n_units"]
        out["rmse"] = metrics["rmse"]
        out["mae"] = metrics["mae"]
    if retrieval["status"] == "ok":
        out["hit_at_k"] = retrieval["hit_at_k"]
    if diagnostics["status"] == "ok":
        out["n_violations"] = len(diagnostics["violations"])
    return out
