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


def risk_band(pred_rul: float) -> str:
    """Map a predicted RUL to a maintenance risk band.

    high:   pred_rul <= 30       (act soon)
    medium: 30 < pred_rul <= 80  (watch / schedule)
    low:    pred_rul > 80        (healthy)
    """
    if pred_rul <= 30:
        return "high"
    if pred_rul <= 80:
        return "medium"
    return "low"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train_and_evaluate() -> dict:
    for d in (MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- Train ---------------------------------------------------------------
    train_raw = load_raw(DATASET, "train")
    train_raw = add_training_rul(train_raw, cap=RUL_CAP)
    train_feat, feature_cols = build_features(train_raw)
    X_train = train_feat[feature_cols].to_numpy()
    y_train = train_feat["rul"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Predict at each test unit's last cycle ------------------------------
    test_raw = load_raw(DATASET, "test")
    test_feat, _ = build_features(test_raw)
    test_last = last_cycle_per_unit(test_feat)
    X_test = test_last[feature_cols].to_numpy()
    pred = model.predict(X_test)
    pred = np.clip(pred, 0, None)  # RUL cannot be negative

    true_uncapped = load_test_rul(DATASET).to_numpy()  # official, in unit order
    true_capped = np.clip(true_uncapped, 0, RUL_CAP)

    metrics_capped = _metrics(true_capped, pred)
    metrics_uncapped = _metrics(true_uncapped, pred)

    # --- Predictions CSV (interface contract) --------------------------------
    preds_df = pd.DataFrame(
        {
            "unit_id": test_last["unit"].to_numpy(),
            "last_cycle": test_last["cycle"].to_numpy(),
            "true_rul": true_uncapped,  # official uncapped ground truth
            "pred_rul": np.round(pred, 2),
            "risk_band": [risk_band(p) for p in pred],
        }
    )
    preds_path = PROCESSED_DIR / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # --- Feature importances -------------------------------------------------
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["importance"] = fi_df["importance"].round(6)
    fi_path = PROCESSED_DIR / "feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)

    # --- Save model ----------------------------------------------------------
    model_path = MODELS_DIR / "rul_baseline.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols, "rul_cap": RUL_CAP}, model_path)

    # --- Figures -------------------------------------------------------------
    _plot_pred_vs_true(true_capped, pred)
    _plot_error_hist(pred - true_capped)
    _plot_degradation(train_raw)

    # --- Metrics JSON --------------------------------------------------------
    band_counts = preds_df["risk_band"].value_counts().to_dict()
    metrics = {
        "dataset": DATASET,
        "model": "RandomForestRegressor",
        "model_params": {
            "n_estimators": 200,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "random_state": RANDOM_STATE,
        },
        "rul_cap": RUL_CAP,
        "n_test_units": int(len(preds_df)),
        "n_train_rows": int(len(train_feat)),
        "n_features": len(feature_cols),
        "metrics_vs_capped_truth": metrics_capped,
        "metrics_vs_uncapped_truth": metrics_uncapped,
        "risk_band_counts": {k: int(v) for k, v in band_counts.items()},
        "top_features": fi_df.head(8).to_dict(orient="records"),
        "notes": (
            "Headline metrics are vs capped truth (cap=125), the target the model "
            "was trained on. Uncapped metrics are worse because some test units have "
            "true RUL > 125, which a capped model cannot reach by construction. "
            "Baseline only; no hyperparameter search, no sequence model."
        ),
    }
    with open(REPORTS_DIR / "metrics_model.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # --- Console summary -----------------------------------------------------
    print(f"[train] rows={len(train_feat)}  features={len(feature_cols)}")
    print(f"[test ] units={len(preds_df)}")
    print(
        "[capped   truth] "
        f"RMSE={metrics_capped['rmse']:.2f}  MAE={metrics_capped['mae']:.2f}  R2={metrics_capped['r2']:.3f}"
    )
    print(
        "[uncapped truth] "
        f"RMSE={metrics_uncapped['rmse']:.2f}  MAE={metrics_uncapped['mae']:.2f}  R2={metrics_uncapped['r2']:.3f}"
    )
    print(f"[bands] {band_counts}")
    print(f"[write] {preds_path}")
    print(f"[write] {fi_path}")
    print(f"[write] {model_path}")
    print(f"[write] {REPORTS_DIR / 'metrics_model.json'}")
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
