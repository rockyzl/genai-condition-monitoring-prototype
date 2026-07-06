"""The 10 pipeline stages — thin wrappers over the existing ``src/`` functions.

Each ``sNN_*`` function takes a :class:`PipelineContext`, declares its input and
output file contracts, and delegates the real work to the modules that already
exist (loader, feature builder, trainer, evidence builder, RAG assistant, eval
harness). The wrappers never reimplement behaviour — s02 (EDA) and s07 (predict,
split out of training) are the only stages with genuinely new orchestration, and
even those call existing functions. All model artifacts stay byte-identical.

Multi-file stages (evidence, diagnostics) anchor their provenance on a small
``_*_manifest.json`` so the skip decision does not need a sidecar per unit file.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.diagnostics import build_evidence as be
from src.models import train_baseline as tb
from src.pipeline import eda as eda_mod
from src.pipeline import run_eval_wrap  # local wrapper, see bottom of file
from src.pipeline.context import PipelineContext, StageResult, StageWork
from src.rag.assistant import diagnose
from src.rag.retriever import Retriever


# --- small IO helpers --------------------------------------------------------
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _raw_file(ctx: PipelineContext, split: str) -> Path:
    return ctx.cfg.path("data_raw") / f"{split}_{ctx.cfg.dataset}.txt"


def _rul_file(ctx: PipelineContext) -> Path:
    return ctx.cfg.path("data_raw") / f"RUL_{ctx.cfg.dataset}.txt"


def _kb_files(ctx: PipelineContext) -> list[Path]:
    return sorted(ctx.cfg.path("knowledge_base").glob("*.md"))


# --- s01 ingest --------------------------------------------------------------
def s01_ingest(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    inputs = [_raw_file(ctx, "train"), _raw_file(ctx, "test"), _rul_file(ctx)]
    out = cfg.path("data_processed") / "ingest_manifest.json"
    params = {"dataset": cfg.dataset, "rul_cap": cfg.rul_cap}

    def work() -> StageWork:
        from src.data.load_cmapss import add_training_rul, load_raw, load_test_rul

        train = add_training_rul(load_raw(cfg.dataset, "train"), cap=cfg.rul_cap)
        test = load_raw(cfg.dataset, "test")
        rul = load_test_rul(cfg.dataset)
        manifest = {
            "dataset": cfg.dataset,
            "rul_cap": cfg.rul_cap,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "n_train_units": int(train["unit"].nunique()),
            "n_test_units": int(test["unit"].nunique()),
            "n_official_rul": int(len(rul)),
        }
        _write_json(out, manifest)
        ctx.journal.stage_progress(
            "s01_ingest",
            f"loaded {manifest['n_train_units']} train + "
            f"{manifest['n_test_units']} test units",
        )
        return StageWork(rows=manifest["train_rows"], key_metrics=manifest,
                         artifacts={str(out): {"train_rows": manifest["train_rows"]}})

    return ctx.run_stage("s01_ingest", inputs, [out], params, work)


# --- s02 eda (new logic) -----------------------------------------------------
def s02_eda(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    inputs = [_raw_file(ctx, "train")]
    out_md = cfg.path("reports") / "eda_summary.md"
    out_json = cfg.path("eda") / "eda_summary.json"
    fig_dir = cfg.path("eda")
    outputs = [
        out_md,
        out_json,
        fig_dir / "monotonicity.png",
        fig_dir / "flat_sensors.png",
        fig_dir / "lifetime_distribution.png",
    ]
    params = {"dataset": cfg.dataset, "flat_max_unique": eda_mod.FLAT_MAX_UNIQUE}

    def work() -> StageWork:
        summary = eda_mod.run(cfg.dataset, out_md, out_json, fig_dir)
        top = summary["sensor_monotonicity"][0] if summary["sensor_monotonicity"] else {}
        n_flat = len(summary["flat_sensors"])
        ctx.journal.stage_progress(
            "s02_eda",
            f"ranked {len(summary['sensor_monotonicity'])} sensors; "
            f"flagged {n_flat} flat sensors to drop",
        )
        km = {
            "top_monotonic_sensor": top.get("sensor"),
            "top_abs_corr": top.get("abs_corr"),
            "n_flat_sensors": n_flat,
            "n_units": summary["n_units"],
        }
        return StageWork(
            rows=summary["n_train_rows"],
            key_metrics=km,
            artifacts={str(out_md): {"n_flat_sensors": n_flat}, str(out_json): km},
        )

    return ctx.run_stage("s02_eda", inputs, outputs, params, work)


# --- s03 preprocess ----------------------------------------------------------
def s03_preprocess(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    inputs = [cfg.path("data_processed") / "ingest_manifest.json"]
    out = cfg.path("data_processed") / "preprocess_summary.json"
    from src.features.build_features import (
        DROPPED_SENSORS,
        INFORMATIVE_SENSORS,
    )

    params = {
        "dropped_sensors": DROPPED_SENSORS,
        "informative_sensors": INFORMATIVE_SENSORS,
        "rolling_window": cfg.rolling_window,
    }

    def work() -> StageWork:
        summary = {
            "dropped_sensors": DROPPED_SENSORS,
            "informative_sensors": INFORMATIVE_SENSORS,
            "n_dropped": len(DROPPED_SENSORS),
            "n_informative": len(INFORMATIVE_SENSORS),
            "op_settings_dropped": True,
            "zscore": False,
            "zscore_reason": "tree model is scale-invariant; z-scoring intentionally omitted",
            "leakage_guards": [
                "group-by-unit CV (no engine spans train and validation folds)",
                "cap applied to the target only, never to features",
                "rolling windows computed within each unit (no cross-engine leakage)",
                "no future-window information used at any cycle",
            ],
            "rolling_window": cfg.rolling_window,
        }
        _write_json(out, summary)
        ctx.journal.stage_progress(
            "s03_preprocess",
            f"dropped {summary['n_dropped']} flat sensors + op-settings; "
            f"kept {summary['n_informative']} informative",
        )
        return StageWork(rows=summary["n_informative"], key_metrics=summary,
                         artifacts={str(out): {"n_informative": summary["n_informative"]}})

    return ctx.run_stage("s03_preprocess", inputs, [out], params, work)


# --- s04 features ------------------------------------------------------------
def s04_features(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    inputs = [
        cfg.path("data_processed") / "preprocess_summary.json",
        _raw_file(ctx, "train"),
    ]
    out = cfg.path("data_processed") / "feature_spec.json"
    params = {"rolling_window": cfg.rolling_window, "dataset": cfg.dataset}

    def work() -> StageWork:
        from src.data.load_cmapss import load_raw
        from src.features.build_features import build_features

        train = load_raw(cfg.dataset, "train")
        feat, feature_cols = build_features(train, window=cfg.rolling_window)
        spec = {
            "n_features": len(feature_cols),
            "n_train_rows": int(len(feat)),
            "rolling_window": cfg.rolling_window,
            "feature_cols": feature_cols,
        }
        _write_json(out, spec)
        ctx.journal.stage_progress(
            "s04_features",
            f"built {spec['n_features']} features over {spec['n_train_rows']} rows",
        )
        return StageWork(rows=spec["n_train_rows"],
                         key_metrics={"n_features": spec["n_features"]},
                         artifacts={str(out): {"n_features": spec["n_features"]}})

    return ctx.run_stage("s04_features", inputs, [out], params, work)


# --- s05 model ---------------------------------------------------------------
def s05_model(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    inputs = [
        cfg.path("data_processed") / "feature_spec.json",
        _raw_file(ctx, "train"),
    ]
    model_path = cfg.path("models") / "rul_baseline.joblib"
    fi_path = cfg.path("data_processed") / "feature_importances.csv"
    meta_path = cfg.path("data_processed") / "model_meta.json"
    outputs = [model_path, fi_path, meta_path]
    rf_kwargs = cfg.rf_params.sklearn_kwargs(cfg.seed)
    params = {"rf_params": rf_kwargs, "rul_cap": cfg.rul_cap, "dataset": cfg.dataset}

    def work() -> StageWork:
        fitted = tb.fit_model(cfg.dataset, cfg.rul_cap, rf_kwargs)
        saved = tb.save_model_artifacts(
            fitted, cfg.rul_cap, model_path, fi_path, meta_path
        )
        top = saved["fi_df"].iloc[0]
        ctx.journal.stage_progress(
            "s05_model",
            f"trained RandomForest on {fitted['n_train_rows']} rows; "
            f"top feature {top['feature']}",
        )
        km = {
            "n_train_rows": fitted["n_train_rows"],
            "n_features": len(fitted["feature_cols"]),
            "top_feature": str(top["feature"]),
        }
        return StageWork(
            rows=fitted["n_train_rows"],
            key_metrics=km,
            artifacts={
                str(model_path): {"n_features": km["n_features"]},
                str(fi_path): {"top_feature": km["top_feature"]},
                str(meta_path): {"n_train_rows": km["n_train_rows"]},
            },
        )

    return ctx.run_stage("s05_model", inputs, outputs, params, work)


# --- s06 select --------------------------------------------------------------
def s06_select(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    meta_path = cfg.path("data_processed") / "model_meta.json"
    inputs = [meta_path]
    out = cfg.path("data_processed") / "champion.json"
    params = {"phase": "A", "candidates": ["RandomForestRegressor"]}

    def work() -> StageWork:
        meta = json.loads(meta_path.read_text())
        champion = {
            "champion": meta["model"],
            "candidates": ["RandomForestRegressor"],
            "selection_basis": "grouped-CV RMSE with a simplicity tiebreak",
            "rationale": (
                "Phase A has a single candidate, so selection is a governed "
                "pass-through recording the RandomForest as champion. Phase B adds "
                "the Ridge floor and the HistGBM challenger over identical "
                "GroupKFold folds and can raise a CARD/HALT if the champion only "
                "marginally beats — or loses to — the Ridge floor."
            ),
            "model_params": meta["model_params"],
            "n_train_rows": meta["n_train_rows"],
            "source": "data/processed/model_meta.json",
        }
        _write_json(out, champion)
        ctx.journal.stage_progress(
            "s06_select", f"champion = {champion['champion']} (1 candidate, Phase A)"
        )
        return StageWork(rows=1, key_metrics={"champion": champion["champion"]},
                         artifacts={str(out): {"champion": champion["champion"]}})

    return ctx.run_stage("s06_select", inputs, [out], params, work)


# --- s07 predict (split out of training) -------------------------------------
def s07_predict(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    model_path = cfg.path("models") / "rul_baseline.joblib"
    fi_path = cfg.path("data_processed") / "feature_importances.csv"
    meta_path = cfg.path("data_processed") / "model_meta.json"
    inputs = [model_path, fi_path, meta_path, _raw_file(ctx, "test"), _rul_file(ctx)]

    preds_path = cfg.path("data_processed") / "test_predictions.csv"
    metrics_path = cfg.path("reports") / "metrics_model.json"
    fig_dir = cfg.path("figures")
    outputs = [
        preds_path,
        metrics_path,
        fig_dir / "pred_vs_true.png",
        fig_dir / "error_hist.png",
        fig_dir / "degradation_units.png",
    ]
    params = {
        "rul_cap": cfg.rul_cap,
        "dataset": cfg.dataset,
        "high_max": cfg.risk_thresholds.high_max,
        "medium_max": cfg.risk_thresholds.medium_max,
    }

    def work() -> StageWork:
        import pandas as pd

        from src.data.load_cmapss import load_raw

        bundle = tb.load_model_bundle(model_path)
        model, feature_cols = bundle["model"], bundle["feature_cols"]
        cap = int(bundle["rul_cap"])
        fi_df = pd.read_csv(fi_path)
        meta = json.loads(meta_path.read_text())

        scored = tb.score_test_units(
            model,
            feature_cols,
            cap,
            cfg.dataset,
            cfg.risk_thresholds.high_max,
            cfg.risk_thresholds.medium_max,
        )
        train_raw = load_raw(cfg.dataset, "train")
        # figures + metrics + predictions all use the shared trainer writers.
        tb.write_predictions_and_metrics(
            scored, fi_df, meta, train_raw, preds_path, metrics_path
        )
        mc = scored["metrics_capped"]
        mu = scored["metrics_uncapped"]
        ctx.journal.stage_progress(
            "s07_predict",
            f"scored {len(scored['preds_df'])} units; "
            f"capped RMSE {mc['rmse']} / uncapped {mu['rmse']}",
        )
        bands = scored["preds_df"]["risk_band"].value_counts().to_dict()
        km = {
            "n_test_units": int(len(scored["preds_df"])),
            "rmse_capped": mc["rmse"],
            "rmse_uncapped": mu["rmse"],
            "risk_band_counts": {k: int(v) for k, v in bands.items()},
        }
        return StageWork(
            rows=km["n_test_units"],
            key_metrics=km,
            artifacts={
                str(preds_path): {"n_test_units": km["n_test_units"]},
                str(metrics_path): {"rmse_capped": mc["rmse"], "rmse_uncapped": mu["rmse"]},
            },
        )

    return ctx.run_stage("s07_predict", inputs, outputs, params, work)


# --- s08 evidence ------------------------------------------------------------
def s08_evidence(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    preds_path = cfg.path("data_processed") / "test_predictions.csv"
    fi_path = cfg.path("data_processed") / "feature_importances.csv"
    raw_test = _raw_file(ctx, "test")
    inputs = [preds_path, fi_path, raw_test]

    evidence_dir = cfg.path("evidence")
    manifest = evidence_dir / "_evidence_manifest.json"
    outputs = [manifest]
    params = {"window": be.LAST_WINDOW, "dataset": cfg.dataset}

    def work() -> StageWork:
        written = be.run(preds_path, fi_path, raw_test, evidence_dir, be.LAST_WINDOW)
        files = sorted(ctx.rel(p) for p in written)
        _write_json(manifest, {"n_records": len(written), "files": files})
        ctx.journal.stage_progress(
            "s08_evidence", f"built {len(written)} per-unit evidence records"
        )
        return StageWork(rows=len(written), key_metrics={"n_records": len(written)},
                         artifacts={str(manifest): {"n_records": len(written)}})

    return ctx.run_stage("s08_evidence", inputs, outputs, params, work)


# --- s09 diagnose ------------------------------------------------------------
def s09_diagnose(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    evidence_dir = cfg.path("evidence")
    evidence_manifest = evidence_dir / "_evidence_manifest.json"
    inputs = [evidence_manifest, *_kb_files(ctx)]

    diag_dir = cfg.path("diagnostics")
    manifest = diag_dir / "_diagnostics_manifest.json"
    outputs = [manifest]
    params = {"retrieval_k": cfg.retrieval_k}

    def work() -> StageWork:
        retriever = Retriever(cfg.path("knowledge_base"))
        diag_dir.mkdir(parents=True, exist_ok=True)
        evidence_files = sorted(evidence_dir.glob("unit_*.json"))
        written: list[str] = []
        grounded = 0
        for ef in evidence_files:
            evidence = json.loads(ef.read_text())
            report = diagnose(evidence, retriever)
            if report["citations"]:
                grounded += 1
            out = diag_dir / ef.name
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            written.append(ctx.rel(out))
        _write_json(
            manifest,
            {"n_reports": len(written), "n_with_citations": grounded, "files": sorted(written)},
        )
        ctx.journal.stage_progress(
            "s09_diagnose",
            f"wrote {len(written)} grounded reports ({grounded} cited)",
        )
        return StageWork(
            rows=len(written),
            key_metrics={"n_reports": len(written), "n_with_citations": grounded},
            artifacts={str(manifest): {"n_reports": len(written)}},
        )

    return ctx.run_stage("s09_diagnose", inputs, outputs, params, work)


# --- s10 eval ----------------------------------------------------------------
def s10_eval(ctx: PipelineContext) -> StageResult:
    cfg = ctx.cfg
    preds_path = cfg.path("data_processed") / "test_predictions.csv"
    metrics_path = cfg.path("reports") / "metrics_model.json"
    evidence_manifest = cfg.path("evidence") / "_evidence_manifest.json"
    inputs = [preds_path, metrics_path, evidence_manifest, *_kb_files(ctx)]
    out = cfg.path("reports") / "evaluation_summary.md"
    params = {"retrieval_k": cfg.retrieval_k}

    def work() -> StageWork:
        result = run_eval_wrap.run()
        ctx.journal.stage_progress(
            "s10_eval",
            f"model={result['model_status']} retrieval={result['retrieval_status']} "
            f"diagnostics={result['diagnostics_status']}",
        )
        return StageWork(
            rows=result.get("n_units"),
            key_metrics=result,
            artifacts={str(out): {"rmse": result.get("rmse")}},
        )

    return ctx.run_stage("s10_eval", inputs, [out], params, work)


#: Stage functions in canonical DAG order (name -> callable).
STAGE_FUNCS = {
    "s01_ingest": s01_ingest,
    "s02_eda": s02_eda,
    "s03_preprocess": s03_preprocess,
    "s04_features": s04_features,
    "s05_model": s05_model,
    "s06_select": s06_select,
    "s07_predict": s07_predict,
    "s08_evidence": s08_evidence,
    "s09_diagnose": s09_diagnose,
    "s10_eval": s10_eval,
}
