"""Train and evaluate the RandomForest RUL baseline on C-MAPSS FD001.

Run directly:

    .venv/bin/python src/models/train_baseline.py

Produces:
    models/rul_baseline.joblib
    data/processed/test_predictions.csv        (contract columns, see below)
    data/processed/feature_importances.csv
    reports/metrics_model.json
    reports/figures/{pred_vs_true.png, error_hist.png, degradation_units.png}

Evaluation protocol
-------------------
We train on the full FD001 training set with a capped RUL target (cap 125,
see :mod:`src.data.load_cmapss`). We then predict RUL at each test unit's LAST
recorded cycle and compare to the official ``RUL_FD001.txt`` vector.

Two truth references are reported, clearly labelled:

* **capped** — the official truth clipped to the same 125 cap the model was
  trained on. This is the metric the model is actually optimised for and the
  fair headline number.
* **uncapped** — the raw official truth. A handful of test units have true RUL
  well above 125; against them the capped model must under-predict by
  construction, so uncapped error is always somewhat worse. Reporting both
  avoids cherry-picking.
"""

from __future__ import annotations

import json
import pathlib
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make ``src`` importable when run as a plain script.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_cmapss import (  # noqa: E402
    RUL_CAP,
    add_training_rul,
    last_cycle_per_unit,
    load_raw,
    load_test_rul,
)
from src.features.build_features import build_features  # noqa: E402

RANDOM_STATE = 42
DATASET = "FD001"

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "rul_baseline.joblib"
PREDS_PATH = PROCESSED_DIR / "test_predictions.csv"
FI_PATH = PROCESSED_DIR / "feature_importances.csv"
META_PATH = PROCESSED_DIR / "model_meta.json"
METRICS_PATH = REPORTS_DIR / "metrics_model.json"

#: RandomForest hyper-parameters. Kept as a module constant so the pipeline
#: config layer and this script's ``__main__`` share one source of truth; the
#: values are exactly those used before the pipeline refactor (byte-identical).
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

#: Risk-band cutoffs (predicted RUL). high <= HIGH_MAX < medium <= MEDIUM_MAX.
HIGH_MAX = 30
MEDIUM_MAX = 80


def risk_band(pred_rul: float, high_max: float = HIGH_MAX, medium_max: float = MEDIUM_MAX) -> str:
    """Map a predicted RUL to a maintenance risk band.

    high:   pred_rul <= 30       (act soon)
    medium: 30 < pred_rul <= 80  (watch / schedule)
    low:    pred_rul > 80        (healthy)
    """
    if pred_rul <= high_max:
        return "high"
    if pred_rul <= medium_max:
        return "medium"
    return "low"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _ensure_dirs() -> None:
    for d in (MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _feature_importance_frame(model, feature_cols: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .assign(importance=lambda d: d["importance"].round(6))
    )


def fit_model(
    dataset: str = DATASET,
    cap: int = RUL_CAP,
    rf_params: dict | None = None,
) -> dict:
    """Fit the RandomForest RUL model on the training split.

    Returns a dict with the fitted ``model``, ``feature_cols``, the training
    feature frame, and ``n_train_rows`` — the single training step, so that the
    pipeline's model stage and the ``__main__`` orchestrator share one code path.
    """
    rf_params = dict(rf_params if rf_params is not None else RF_PARAMS)
    train_raw = load_raw(dataset, "train")
    train_raw = add_training_rul(train_raw, cap=cap)
    train_feat, feature_cols = build_features(train_raw)
    X_train = train_feat[feature_cols].to_numpy()
    y_train = train_feat["rul"].to_numpy()

    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "n_train_rows": int(len(train_feat)),
        "train_raw": train_raw,
    }


def save_model_artifacts(
    fitted: dict,
    cap: int = RUL_CAP,
    model_path=MODEL_PATH,
    fi_path=FI_PATH,
    meta_path=META_PATH,
) -> dict:
    """Persist the model joblib, feature importances CSV, and model-meta JSON.

    ``model_meta.json`` carries the small facts the downstream predict stage
    needs to rebuild ``metrics_model.json`` (n_train_rows / n_features) without
    retraining, keeping the joblib contract (model / feature_cols / rul_cap)
    unchanged.
    """
    _ensure_dirs()
    model = fitted["model"]
    feature_cols = fitted["feature_cols"]

    fi_df = _feature_importance_frame(model, feature_cols)
    fi_df.to_csv(fi_path, index=False)

    joblib.dump({"model": model, "feature_cols": feature_cols, "rul_cap": cap}, model_path)

    meta = {
        "dataset": DATASET,
        "n_train_rows": int(fitted["n_train_rows"]),
        "n_features": len(feature_cols),
        "rul_cap": int(cap),
        "model": "RandomForestRegressor",
        "model_params": {
            "n_estimators": RF_PARAMS["n_estimators"],
            "min_samples_leaf": RF_PARAMS["min_samples_leaf"],
            "max_features": RF_PARAMS["max_features"],
            "random_state": RF_PARAMS["random_state"],
        },
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"fi_df": fi_df, "meta": meta}


def load_model_bundle(model_path=MODEL_PATH) -> dict:
    """Load the persisted joblib bundle (model + feature_cols + rul_cap)."""
    return joblib.load(model_path)


def score_test_units(
    model,
    feature_cols: list[str],
    cap: int,
    dataset: str = DATASET,
    high_max: float = HIGH_MAX,
    medium_max: float = MEDIUM_MAX,
) -> dict:
    """Score each test unit at its last cycle. Returns predictions frame + arrays."""
    test_raw = load_raw(dataset, "test")
    test_feat, _ = build_features(test_raw)
    test_last = last_cycle_per_unit(test_feat)
    X_test = test_last[feature_cols].to_numpy()
    pred = model.predict(X_test)
    pred = np.clip(pred, 0, None)  # RUL cannot be negative

    true_uncapped = load_test_rul(dataset).to_numpy()  # official, in unit order
    true_capped = np.clip(true_uncapped, 0, cap)

    preds_df = pd.DataFrame(
        {
            "unit_id": test_last["unit"].to_numpy(),
            "last_cycle": test_last["cycle"].to_numpy(),
            "true_rul": true_uncapped,  # official uncapped ground truth
            "pred_rul": np.round(pred, 2),
            "risk_band": [risk_band(p, high_max, medium_max) for p in pred],
        }
    )
    return {
        "preds_df": preds_df,
        "pred": pred,
        "true_capped": true_capped,
        "true_uncapped": true_uncapped,
        "metrics_capped": _metrics(true_capped, pred),
        "metrics_uncapped": _metrics(true_uncapped, pred),
    }


def build_metrics(scored: dict, fi_df: pd.DataFrame, meta: dict) -> dict:
    """Assemble the ``metrics_model.json`` payload (identical layout to before)."""
    preds_df = scored["preds_df"]
    band_counts = preds_df["risk_band"].value_counts().to_dict()
    return {
        "dataset": meta["dataset"],
        "model": "RandomForestRegressor",
        "model_params": meta["model_params"],
        "rul_cap": meta["rul_cap"],
        "n_test_units": int(len(preds_df)),
        "n_train_rows": int(meta["n_train_rows"]),
        "n_features": meta["n_features"],
        "metrics_vs_capped_truth": scored["metrics_capped"],
        "metrics_vs_uncapped_truth": scored["metrics_uncapped"],
        "risk_band_counts": {k: int(v) for k, v in band_counts.items()},
        "top_features": fi_df.head(8).to_dict(orient="records"),
        "notes": (
            "Headline metrics are vs capped truth (cap=125), the target the model "
            "was trained on. Uncapped metrics are worse because some test units have "
            "true RUL > 125, which a capped model cannot reach by construction. "
            "Baseline only; no hyperparameter search, no sequence model."
        ),
    }


def write_predictions_and_metrics(
    scored: dict,
    fi_df: pd.DataFrame,
    meta: dict,
    train_raw: pd.DataFrame,
    preds_path=PREDS_PATH,
    metrics_path=METRICS_PATH,
) -> dict:
    """Write test_predictions.csv, metrics_model.json, and the three figures."""
    _ensure_dirs()
    scored["preds_df"].to_csv(preds_path, index=False)

    _plot_pred_vs_true(scored["true_capped"], scored["pred"])
    _plot_error_hist(scored["pred"] - scored["true_capped"])
    _plot_degradation(train_raw)

    metrics = build_metrics(scored, fi_df, meta)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def train_and_evaluate() -> dict:
    # --- Train ---------------------------------------------------------------
    fitted = fit_model(DATASET, RUL_CAP, RF_PARAMS)
    saved = save_model_artifacts(fitted, RUL_CAP)
    fi_df = saved["fi_df"]
    meta = saved["meta"]

    # --- Predict at each test unit's last cycle ------------------------------
    scored = score_test_units(fitted["model"], fitted["feature_cols"], RUL_CAP, DATASET)

    # --- Persist predictions, figures, metrics -------------------------------
    metrics = write_predictions_and_metrics(scored, fi_df, meta, fitted["train_raw"])

    # --- Console summary -----------------------------------------------------
    metrics_capped = scored["metrics_capped"]
    metrics_uncapped = scored["metrics_uncapped"]
    band_counts = scored["preds_df"]["risk_band"].value_counts().to_dict()
    print(f"[train] rows={fitted['n_train_rows']}  features={len(fitted['feature_cols'])}")
    print(f"[test ] units={len(scored['preds_df'])}")
    print(
        "[capped   truth] "
        f"RMSE={metrics_capped['rmse']:.2f}  MAE={metrics_capped['mae']:.2f}  R2={metrics_capped['r2']:.3f}"
    )
    print(
        "[uncapped truth] "
        f"RMSE={metrics_uncapped['rmse']:.2f}  MAE={metrics_uncapped['mae']:.2f}  R2={metrics_uncapped['r2']:.3f}"
    )
    print(f"[bands] {band_counts}")
    print(f"[write] {PREDS_PATH}")
    print(f"[write] {FI_PATH}")
    print(f"[write] {MODEL_PATH}")
    print(f"[write] {METRICS_PATH}")
    print(f"[write] figures -> {FIGURES_DIR}")
    return metrics


def _plot_pred_vs_true(true_capped: np.ndarray, pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_capped, pred, alpha=0.7, edgecolor="k", linewidth=0.3)
    lim = max(true_capped.max(), pred.max()) + 5
    ax.plot([0, lim], [0, lim], "r--", label="perfect")
    ax.set_xlabel("True RUL (capped at 125)")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("FD001 test: predicted vs true RUL (last cycle)")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pred_vs_true.png", dpi=120)
    plt.close(fig)


def _plot_error_hist(errors: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(errors, bins=25, edgecolor="k", alpha=0.8)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Prediction error (pred - true, capped)")
    ax.set_ylabel("Number of test units")
    ax.set_title("FD001 test: RUL prediction error distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_hist.png", dpi=120)
    plt.close(fig)


def _plot_degradation(train_raw: pd.DataFrame, units: tuple[int, ...] = (1, 2, 3)) -> None:
    """Show a few informative sensors trending as a unit approaches failure."""
    sensors = ["sensor_4", "sensor_11", "sensor_15"]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(8, 8), sharex=False)
    for ax, sensor in zip(axes, sensors):
        for unit in units:
            u = train_raw[train_raw["unit"] == unit]
            # x-axis: cycles remaining until failure (0 == failure)
            rem = u["cycle"].max() - u["cycle"]
            ax.plot(rem, u[sensor], label=f"unit {unit}", linewidth=1)
        ax.set_ylabel(sensor)
        ax.invert_xaxis()  # failure (0) on the right
        ax.grid(alpha=0.3)
    axes[0].set_title("FD001 train: sensor trends vs cycles-to-failure")
    axes[-1].set_xlabel("Cycles remaining until failure")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "degradation_units.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    train_and_evaluate()
