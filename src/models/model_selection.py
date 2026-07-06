"""Model-selection bake-off (Phase B) — the governed champion decision.

s05 trains one RandomForest. This module answers a different question: *is the
RandomForest actually the right model?* It runs a small, honest bake-off over
three candidates on identical cross-validation folds and records why the winner
won, as both a lay-reader report and a machine-readable JSON.

Candidates
----------
* **Ridge** — the interpretable *floor*. A plain linear model (one weight per
  feature, standardised inside each fold). The champion must beat it or the
  whole stage HALTs: if a straight line does as well as the ensemble, the
  ensemble is not earning its complexity.
* **RandomForestRegressor** — the incumbent champion (exactly s05's params).
* **HistGradientBoostingRegressor** — the challenger (gradient boosting).

Protocol (leakage-safe, deterministic)
--------------------------------------
* **GroupKFold(5) by ``unit`` on the training split only.** No engine ever spans
  a train and a validation fold, so the CV score reflects generalisation to
  *unseen engines*, not memorised ones. The test set is never touched here.
* **Identical folds for every candidate** — the fold indices are built once and
  reused, so score differences are the model, not the split.
* **All preprocessing fits inside the fold** — Ridge's ``StandardScaler`` is fit
  on the training fold only; nothing is fit on validation rows.
* Same seed as the rest of the pipeline → byte-stable numbers across reruns.

Selection criteria, in priority order
--------------------------------------
1. **Grouped-CV RMSE (mean ± std)** — the headline "typical miss".
2. **End-of-life calibration** — RMSE on the low-RUL rows (true RUL < 50) plus
   the *optimistic-error fraction* (share of those rows where the model predicts
   MORE life than remains — the operationally dangerous direction).
3. **Simplicity tiebreak** — when nothing separates the models, keep the simpler
   / incumbent one.

Champion contract
-----------------
The default outcome is the RandomForest. A challenger replaces it only on a
*clear* win — better than RF by more than ``champion_margin_rmse`` cycles on
BOTH criterion 1 AND criterion 2 — because determinism and the downstream
prediction contract matter more than a fractional-cycle improvement. Whatever
wins, it must beat the Ridge floor; otherwise :func:`choose_champion` raises
:class:`ChampionBelowFloorError` and the stage fails loudly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

# Candidate names — kept identical to the sklearn class names so ``model_meta``
# / ``metrics_model.json`` and the selection report speak the same vocabulary.
RIDGE = "Ridge"
RANDOM_FOREST = "RandomForestRegressor"
HIST_GBM = "HistGradientBoostingRegressor"

#: Rows with a (capped) true RUL below this count as "near end-of-life" — the
#: region where an optimistic miss is operationally dangerous (criterion 2).
LOW_RUL_THRESHOLD = 50

#: GroupKFold fold count (identical folds shared by every candidate).
N_SPLITS = 5

#: Ridge regularisation strength (features are standardised inside each fold, so
#: alpha is on a comparable scale). Fixed, not searched — the floor is a
#: reference, not a tuned contender.
RIDGE_ALPHA = 1.0

#: Default clear-win margin (cycles of RMSE) a challenger must exceed on BOTH
#: criterion 1 and criterion 2 to unseat the incumbent. Overridable via
#: ``config.gate_thresholds.champion_margin_rmse``.
DEFAULT_CHAMPION_MARGIN = 1.0


class ChampionBelowFloorError(RuntimeError):
    """Raised when the selected champion fails to beat the Ridge floor (HALT)."""


@dataclass(frozen=True)
class CandidateSpec:
    """Static description of one candidate (how it works, for the lay report)."""

    name: str
    role: str  # "floor" | "incumbent champion" | "challenger"
    make: Callable[[], object]  # fresh, unfitted estimator
    how_it_works: str  # one plain-language line
    explains_itself: str  # lay answer to "can it show its reasoning?"
    complexity_rank: int  # 1 = simplest; used only for the tiebreak
    # Cap on native (OpenMP/BLAS) threads while fitting this candidate. HGB's
    # OpenMP backend deadlocks under some WSL2/container setups with the default
    # thread count; pinning it to 1 is deterministic and costs ~0.5s/fold. None
    # leaves threading untouched (RandomForest keeps its joblib parallelism).
    thread_limit: int | None = None


# --- data + folds ------------------------------------------------------------
def load_training_xy(cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y, groups, feature_cols)`` for the FD001 training split.

    ``y`` is the capped RUL target (cap from config) — the exact target s05
    trains on — and ``groups`` is the per-row ``unit`` id used by GroupKFold.
    """
    from src.data.load_cmapss import add_training_rul, load_raw
    from src.features.build_features import build_features

    train = add_training_rul(load_raw(cfg.dataset, "train"), cap=cfg.rul_cap)
    feat, feature_cols = build_features(train, window=cfg.rolling_window)
    X = feat[feature_cols].to_numpy()
    y = feat["rul"].to_numpy()
    groups = feat["unit"].to_numpy()
    return X, y, groups, feature_cols


def build_folds(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = N_SPLITS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the shared GroupKFold folds once. Deterministic (GroupKFold does not
    shuffle), so the same data always yields the same folds."""
    gkf = GroupKFold(n_splits=n_splits)
    return [(tr, va) for tr, va in gkf.split(X, y, groups)]


def fold_signature(folds: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Stable 16-hex digest of a fold set's validation indices.

    Two candidates evaluated on identical folds produce the identical signature;
    this is what the "identical folds across candidates" test checks.
    """
    h = hashlib.sha256()
    for _, val_idx in folds:
        h.update(np.asarray(val_idx, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def units_cross_folds(
    folds: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> bool:
    """True if any unit appears in both train and validation of some fold.

    The whole point of GroupKFold-by-unit is that this is always False; the
    leakage test asserts on it.
    """
    for train_idx, val_idx in folds:
        if set(groups[train_idx]) & set(groups[val_idx]):
            return True
    return False


# --- metrics -----------------------------------------------------------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_candidate(
    make_estimator: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    low_rul_threshold: int = LOW_RUL_THRESHOLD,
    thread_limit: int | None = None,
) -> dict:
    """Cross-validate one candidate on the shared folds.

    Returns criterion-1 (per-fold + mean±std CV-RMSE) and criterion-2
    (low-RUL RMSE + optimistic fraction, pooled over out-of-fold predictions).
    A fresh estimator is fit per fold; predictions are clipped at 0 exactly like
    the deployed scorer, so the CV number is comparable to test-time behaviour.
    ``thread_limit`` caps native threads for the fits (``None`` = untouched).
    """
    oof_pred = np.full(len(y), np.nan)
    fold_rmses: list[float] = []
    used_val: list[np.ndarray] = []
    # threadpool_limits(limits=None) is a no-op, so a limit-free candidate keeps
    # full parallelism while HGB is pinned single-threaded (WSL2 OpenMP guard).
    with threadpool_limits(limits=thread_limit):
        for train_idx, val_idx in folds:
            est = make_estimator()
            est.fit(X[train_idx], y[train_idx])
            pred = np.clip(est.predict(X[val_idx]), 0, None)
            oof_pred[val_idx] = pred
            fold_rmses.append(_rmse(y[val_idx], pred))
            used_val.append(val_idx)

    low_mask = y < low_rul_threshold
    low_true = y[low_mask]
    low_pred = oof_pred[low_mask]
    low_rmse = _rmse(low_true, low_pred)
    optimistic_fraction = float(np.mean(low_pred > low_true)) if low_true.size else 0.0

    return {
        "cv_rmse_mean": round(float(np.mean(fold_rmses)), 4),
        "cv_rmse_std": round(float(np.std(fold_rmses)), 4),
        "cv_rmse_folds": [round(v, 4) for v in fold_rmses],
        "low_rul_rmse": round(low_rmse, 4),
        "low_rul_optimistic_fraction": round(optimistic_fraction, 4),
        "n_low_rul_rows": int(low_mask.sum()),
        "fold_signature": fold_signature([(None, v) for v in used_val]),
    }


# --- candidate roster --------------------------------------------------------
def candidate_specs(cfg) -> list[CandidateSpec]:
    """The three bake-off candidates, in report order (floor, incumbent, challenger)."""
    seed = cfg.seed
    rf_kwargs = cfg.rf_params.sklearn_kwargs(seed)

    def make_ridge():
        # Standardise inside the pipeline so scaling is fit per-fold (no leakage)
        # and Ridge's penalty is on a comparable scale across features.
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))

    return [
        CandidateSpec(
            name=RIDGE,
            role="floor",
            make=make_ridge,
            how_it_works="Fits one straight-line weight per sensor feature (a linear model).",
            explains_itself="Yes — one signed weight per signal.",
            complexity_rank=1,
        ),
        CandidateSpec(
            name=RANDOM_FOREST,
            role="incumbent champion",
            make=lambda: RandomForestRegressor(**rf_kwargs),
            how_it_works="Averages 200 decision trees that each vote on remaining life.",
            explains_itself="Yes — ranks which sensors drove the estimate.",
            complexity_rank=2,
        ),
        CandidateSpec(
            name=HIST_GBM,
            role="challenger",
            make=lambda: HistGradientBoostingRegressor(random_state=seed),
            how_it_works="Builds trees in sequence, each correcting the previous one's misses.",
            explains_itself="Indirectly — needs a follow-up permutation test.",
            complexity_rank=3,
            thread_limit=1,
        ),
    ]


# --- the decision ------------------------------------------------------------
def choose_champion(
    metrics: dict[str, dict],
    margin_rmse: float = DEFAULT_CHAMPION_MARGIN,
    floor_rmse: float | None = None,
) -> dict:
    """Pure selection logic over already-computed candidate metrics.

    ``metrics`` maps candidate name -> its :func:`evaluate_candidate` dict.
    The incumbent (:data:`RANDOM_FOREST`) stays champion unless the challenger
    (:data:`HIST_GBM`) beats it by more than ``margin_rmse`` on BOTH criterion 1
    (CV-RMSE) and criterion 2 (low-RUL RMSE). The final champion must beat the
    Ridge floor or :class:`ChampionBelowFloorError` is raised.

    ``floor_rmse`` overrides the measured Ridge CV-RMSE when set (config).
    Returns a verdict dict (champion, whether it swapped, the numeric gaps, and
    the human rationale parts) — it does not touch any files.
    """
    rf = metrics[RANDOM_FOREST]
    ridge = metrics[RIDGE]
    challenger = metrics.get(HIST_GBM)

    floor = float(floor_rmse) if floor_rmse is not None else ridge["cv_rmse_mean"]

    champion = RANDOM_FOREST
    swapped = False
    cv_gap = None
    lowrul_gap = None
    if challenger is not None:
        cv_gap = round(rf["cv_rmse_mean"] - challenger["cv_rmse_mean"], 4)
        lowrul_gap = round(rf["low_rul_rmse"] - challenger["low_rul_rmse"], 4)
        if cv_gap > margin_rmse and lowrul_gap > margin_rmse:
            champion = HIST_GBM
            swapped = True

    champion_cv = metrics[champion]["cv_rmse_mean"]
    floor_gap = round(floor - champion_cv, 4)
    beats_floor = champion_cv < floor

    verdict = {
        "champion": champion,
        "swapped_from_default": swapped,
        "default_champion": RANDOM_FOREST,
        "beats_floor": beats_floor,
        "floor_rmse": round(float(floor), 4),
        "floor_gap_cycles": floor_gap,
        "champion_margin_rmse": float(margin_rmse),
        "challenger_cv_gap_cycles": cv_gap,
        "challenger_lowrul_gap_cycles": lowrul_gap,
    }

    if not beats_floor:
        raise ChampionBelowFloorError(
            f"champion {champion} CV-RMSE {champion_cv:.4f} does not beat the "
            f"Ridge floor {floor:.4f} — a linear model matches the ensemble, so "
            f"the stage HALTs rather than shipping an unjustified model."
        )
    return verdict


# --- champion swap (dormant unless a challenger wins) ------------------------
def _retrain_and_save_champion(
    cfg,
    name: str,
    make_estimator: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    model_path: Path,
    fi_path: Path,
    meta_path: Path,
    thread_limit: int | None = None,
) -> None:
    """Retrain a winning challenger on ALL training rows and overwrite the RF
    artifacts behind the SAME joblib path + feature_importances schema.

    Only reached when a challenger clearly wins. Importances come from a
    permutation test (the challenger is not self-explaining), mapped into the
    existing ``[feature, importance]`` CSV schema. After this, s07_predict must
    be rerun to refresh predictions/metrics.
    """
    import joblib
    from sklearn.inspection import permutation_importance

    with threadpool_limits(limits=thread_limit):
        est = make_estimator()
        est.fit(X, y)
        perm = permutation_importance(
            est, X, y, n_repeats=5, random_state=cfg.seed, n_jobs=1
        )
    joblib.dump(
        {"model": est, "feature_cols": feature_cols, "rul_cap": int(cfg.rul_cap)},
        model_path,
    )

    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .assign(importance=lambda d: d["importance"].round(6))
    )
    fi_df.to_csv(fi_path, index=False)

    meta = json.loads(Path(meta_path).read_text())
    meta["model"] = name
    meta["model_params"] = est.get_params() if not hasattr(est, "steps") else {"pipeline": name}
    Path(meta_path).write_text(json.dumps(meta, indent=2))


# --- orchestration -----------------------------------------------------------
def run_selection(
    cfg,
    md_path: Path,
    json_path: Path,
    champion_path: Path,
    model_path: Path | None = None,
    fi_path: Path | None = None,
    meta_path: Path | None = None,
) -> dict:
    """Run the full bake-off, write all three artifacts, and return a summary.

    Writes ``reports/model_selection.md`` (lay + technical tables + rationale),
    ``reports/model_selection.json`` (machine-readable), and refreshes
    ``data/processed/champion.json``. Retrains/saves the champion behind the RF
    joblib path only if a challenger wins (paths must be supplied for that).
    Raises :class:`ChampionBelowFloorError` (after writing evidence) if the
    champion cannot beat the Ridge floor.
    """
    X, y, groups, feature_cols = load_training_xy(cfg)
    folds = build_folds(X, y, groups, N_SPLITS)
    specs = candidate_specs(cfg)

    metrics = {
        spec.name: evaluate_candidate(
            spec.make, X, y, folds, LOW_RUL_THRESHOLD, spec.thread_limit
        )
        for spec in specs
    }

    margin = cfg.gate_thresholds.champion_margin_rmse
    margin = float(margin) if margin is not None else DEFAULT_CHAMPION_MARGIN
    floor_override = cfg.gate_thresholds.ridge_floor_rmse

    # Assemble the machine-readable payload BEFORE the floor gate so a HALT still
    # leaves auditable evidence on disk.
    protocol = {
        "cv": "GroupKFold",
        "n_splits": N_SPLITS,
        "group_by": "unit_id",
        "n_train_rows": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "n_features": len(feature_cols),
        "target": f"capped RUL (cap={cfg.rul_cap})",
        "low_rul_threshold": LOW_RUL_THRESHOLD,
        "identical_folds": len({m["fold_signature"] for m in metrics.values()}) == 1,
        "fold_signature": fold_signature(folds),
    }
    candidates_payload = [
        {
            "name": spec.name,
            "role": spec.role,
            "complexity_rank": spec.complexity_rank,
            "how_it_works": spec.how_it_works,
            "explains_itself": spec.explains_itself,
            **metrics[spec.name],
        }
        for spec in specs
    ]

    try:
        verdict = choose_champion(metrics, margin_rmse=margin, floor_rmse=floor_override)
        halted = False
        error = None
    except ChampionBelowFloorError as exc:
        verdict = {
            "champion": None,
            "beats_floor": False,
            "floor_rmse": round(
                float(floor_override) if floor_override is not None
                else metrics[RIDGE]["cv_rmse_mean"],
                4,
            ),
            "default_champion": RANDOM_FOREST,
        }
        halted = True
        error = str(exc)

    rationale = _champion_rationale(verdict, metrics, halted, error)
    payload = {
        "champion": verdict.get("champion"),
        "seed": cfg.seed,
        "protocol": protocol,
        "criteria": [
            "grouped-CV RMSE (mean±std)",
            "end-of-life calibration: low-RUL RMSE + optimistic-error fraction",
            "simplicity tiebreak",
        ],
        "candidates": candidates_payload,
        "verdict": verdict,
        "champion_rationale": rationale,
        "halted": halted,
        "error": error,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(payload, specs, metrics))

    if halted:
        # Evidence is on disk; now fail the stage loudly.
        raise ChampionBelowFloorError(error)

    # Refresh the (backward-compatible) champion.json contract.
    champion_name = verdict["champion"]
    champion_json = {
        "champion": champion_name,
        "candidates": [spec.name for spec in specs],
        "selection_basis": "grouped-CV RMSE, end-of-life calibration, simplicity tiebreak",
        "swapped_from_default": verdict["swapped_from_default"],
        "beats_floor": verdict["beats_floor"],
        "floor_rmse": verdict["floor_rmse"],
        "cv_rmse_mean": metrics[champion_name]["cv_rmse_mean"],
        "low_rul_rmse": metrics[champion_name]["low_rul_rmse"],
        "rationale": rationale,
        "source": "reports/model_selection.json",
    }
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    champion_path.write_text(json.dumps(champion_json, indent=2, ensure_ascii=False))

    # Only when a challenger unseats the RandomForest do we touch the model
    # artifacts. The default RF path leaves s05's outputs byte-identical.
    if verdict["swapped_from_default"] and model_path is not None:
        winner = next(s for s in specs if s.name == champion_name)
        _retrain_and_save_champion(
            cfg, champion_name, winner.make, X, y, feature_cols,
            model_path, fi_path, meta_path, winner.thread_limit,
        )

    return {
        "champion": champion_name,
        "swapped": verdict["swapped_from_default"],
        "beats_floor": verdict["beats_floor"],
        "floor_gap_cycles": verdict["floor_gap_cycles"],
        "cv_rmse_mean": metrics[champion_name]["cv_rmse_mean"],
        "cv_rmse_std": metrics[champion_name]["cv_rmse_std"],
        "low_rul_rmse": metrics[champion_name]["low_rul_rmse"],
        "candidates": [c["name"] for c in candidates_payload],
        "n_candidates": len(specs),
        "seed": cfg.seed,
        "identical_folds": protocol["identical_folds"],
    }


# --- rendering ---------------------------------------------------------------
def _champion_rationale(
    verdict: dict, metrics: dict[str, dict], halted: bool, error: str | None
) -> str:
    if halted:
        return (
            "HALT — no model was selected. " + (error or "")
            + " Re-examine features or hyper-parameters before shipping."
        )
    champ = verdict["champion"]
    m = metrics[champ]
    floor_gap = verdict["floor_gap_cycles"]
    parts = [
        f"Champion: **{champ}**. It posts a grouped-CV typical miss of "
        f"{m['cv_rmse_mean']:.2f} ± {m['cv_rmse_std']:.2f} cycles (5 folds, split "
        f"by engine so no unit is scored on itself), and beats the Ridge floor by "
        f"{floor_gap:.2f} cycles — the ensemble earns its complexity."
    ]
    if verdict["swapped_from_default"]:
        parts.append(
            f"The {HIST_GBM} challenger cleared the clear-win bar "
            f"(>{verdict['champion_margin_rmse']:.1f} cycles better than the "
            f"RandomForest on BOTH overall CV-RMSE ({verdict['challenger_cv_gap_cycles']:.2f}) "
            f"and end-of-life RMSE ({verdict['challenger_lowrul_gap_cycles']:.2f})), so it "
            f"replaces the incumbent; the model artifacts were retrained behind the "
            f"same prediction interface."
        )
    else:
        cv_gap = verdict.get("challenger_cv_gap_cycles")
        low_gap = verdict.get("challenger_lowrul_gap_cycles")
        parts.append(
            f"The {HIST_GBM} challenger did NOT clear the clear-win bar "
            f"(needs >{verdict['champion_margin_rmse']:.1f} cycles better than the "
            f"RandomForest on BOTH criteria; observed gaps: CV-RMSE "
            f"{cv_gap:+.2f}, end-of-life RMSE {low_gap:+.2f} — positive means RF is "
            f"already ahead). Determinism and the unchanged downstream contract "
            f"outweigh a fractional-cycle change, so the RandomForest stays champion "
            f"(simplicity tiebreak favours the incumbent)."
        )
    parts.append(
        f"End-of-life calibration (true RUL < {LOW_RUL_THRESHOLD}): the champion's "
        f"low-RUL RMSE is {m['low_rul_rmse']:.2f} cycles and it guesses too healthy "
        f"on {m['low_rul_optimistic_fraction'] * 100:.1f}% of near-failure rows — the "
        f"honest, watch-this caveat that the uncertainty note carries downstream."
    )
    return " ".join(parts)


def _render_markdown(
    payload: dict, specs: list[CandidateSpec], metrics: dict[str, dict]
) -> str:
    champ = payload["champion"]
    proto = payload["protocol"]
    lines: list[str] = []
    lines.append("# Model selection — bake-off report\n")
    lines.append(
        f"Champion: **{champ}**  ·  seed **{payload['seed']}**  ·  "
        f"protocol: GroupKFold({proto['n_splits']}) by `unit_id` on "
        f"{proto['n_train_rows']:,} training rows / {proto['n_groups']} engines. "
        f"The test set is never touched during selection.\n"
    )

    # --- lay-reader table ----------------------------------------------------
    lines.append("## How to read this (plain language)\n")
    lines.append(
        "Three models competed on the same engines, judged on how many cycles "
        "they typically miss by (lower is better), how they behave close to "
        "failure, and whether they can explain themselves.\n"
    )
    lines.append(
        "| Model | How it works | Typical miss (± cycles) | Explains itself? | Why (not) picked |"
    )
    lines.append("|---|---|---|---|---|")
    for spec in specs:
        m = metrics[spec.name]
        lines.append(
            f"| {spec.name} | {spec.how_it_works} | ±{m['cv_rmse_mean']:.1f} | "
            f"{spec.explains_itself} | {_why_cell(spec, payload, metrics)} |"
        )
    lines.append("")

    # --- technical table -----------------------------------------------------
    lines.append("## Technical comparison\n")
    lines.append(
        "| Model | CV-RMSE (mean ± std) | Low-RUL RMSE (true RUL<50) | "
        "Optimistic % (pred>true, low-RUL) | Role |"
    )
    lines.append("|---|---|---|---|---|")
    for spec in specs:
        m = metrics[spec.name]
        star = " ⬅ champion" if spec.name == champ else ""
        lines.append(
            f"| {spec.name}{star} | {m['cv_rmse_mean']:.2f} ± {m['cv_rmse_std']:.2f} | "
            f"{m['low_rul_rmse']:.2f} | {m['low_rul_optimistic_fraction'] * 100:.1f}% | "
            f"{spec.role} |"
        )
    lines.append("")
    lines.append(
        f"Criteria in priority order: (1) grouped-CV RMSE, (2) end-of-life "
        f"calibration (low-RUL RMSE + optimistic fraction), (3) simplicity "
        f"tiebreak. Identical folds across candidates: "
        f"**{proto['identical_folds']}** (fold signature `{proto['fold_signature']}`).\n"
    )

    lines.append("## Champion rationale\n")
    lines.append(payload["champion_rationale"] + "\n")

    lines.append("## Guardrails\n")
    lines.append(
        "- **Floor gate:** the champion must beat the Ridge linear floor on "
        "grouped-CV RMSE, or the stage HALTs. A straight line matching the "
        "ensemble would mean the ensemble is unjustified.\n"
        "- **Incumbent bias by design:** the RandomForest stays champion unless a "
        f"challenger is clearly better (> {payload['verdict'].get('champion_margin_rmse', DEFAULT_CHAMPION_MARGIN):.1f} "
        "cycles on BOTH overall and end-of-life RMSE). Determinism and the fixed "
        "downstream prediction contract outrank fractional-cycle wins.\n"
        "- **No leakage:** folds are grouped by engine; every preprocessing step "
        "(e.g. Ridge's standardiser) is fit inside the training fold only.\n"
    )
    return "\n".join(lines)


def _why_cell(spec: CandidateSpec, payload: dict, metrics: dict[str, dict]) -> str:
    champ = payload["champion"]
    verdict = payload["verdict"]
    if spec.role == "floor":
        gap = verdict.get("floor_gap_cycles")
        gap_txt = f" (champion beats it by {gap:.1f} cycles)" if gap is not None else ""
        return f"Reference floor the champion must beat{gap_txt}."
    if spec.name == champ:
        if verdict.get("swapped_from_default"):
            return "Picked: clearly better than the incumbent on accuracy and end-of-life calibration."
        return "Picked: most accurate self-explaining model; clears the floor and holds the incumbent contract."
    # a non-champion, non-floor candidate (the challenger that did not win)
    return "Not picked: not clearly better than the incumbent on both criteria; keeps determinism."
