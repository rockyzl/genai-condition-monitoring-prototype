"""Typed configuration for the condition-monitoring pipeline.

``config/pipeline.yaml`` is the single source of truth; this module loads it into
a :class:`PipelineConfig` dataclass whose **defaults reproduce the exact values
that used to be hard-coded across ``src/``** (seed 42, RUL cap 125, rolling
window 5, the RandomForest grid, the 30/80 risk cutoffs). A config-free
``PipelineConfig()`` therefore behaves identically to the pre-pipeline scripts,
and the YAML only has to override what differs.

Round-trip guarantee: ``PipelineConfig.from_dict(cfg.to_dict()) == cfg``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Repository root, derived from this file (src/pipeline/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default location of the pipeline config file.
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"


@dataclass
class Paths:
    """Project-relative paths. Resolved against the repo root by :meth:`resolve`."""

    data_raw: str = "data/raw/CMAPSSData"
    data_processed: str = "data/processed"
    reports: str = "reports"
    figures: str = "reports/figures"
    eda: str = "reports/eda"
    evidence: str = "data/processed/evidence"
    diagnostics: str = "data/processed/diagnostics"
    models: str = "models"
    knowledge_base: str = "docs/knowledge_base"
    journal: str = "reports/pipeline_journal.jsonl"
    manifest: str = "reports/pipeline_manifest.md"

    def resolve(self, name: str, root: Path = REPO_ROOT) -> Path:
        """Return an absolute :class:`Path` for the named path attribute."""
        return (root / getattr(self, name)).resolve()


@dataclass
class RFParams:
    """RandomForestRegressor hyper-parameters (n_jobs/random_state added at use)."""

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 3
    max_features: str = "sqrt"
    n_jobs: int = -1

    def sklearn_kwargs(self, random_state: int) -> dict:
        """Full kwargs for ``RandomForestRegressor(**kwargs)``."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass
class RiskThresholds:
    """Predicted-RUL cutoffs: high <= high_max < medium <= medium_max < low."""

    high_max: float = 30
    medium_max: float = 80


@dataclass
class GateThresholds:
    """Reliability-gate thresholds. Placeholder for Phase C (see pipeline.yaml).

    Fixed now so gate outcomes can be hashed into the trace later without a
    config migration. Phase A does not read these.
    """

    ridge_floor_rmse: float | None = None
    champion_margin_rmse: float | None = None
    min_history_cycles: int = 20
    max_optimistic_fraction: float | None = None


@dataclass
class PipelineConfig:
    """Root configuration object threaded through every pipeline stage."""

    seed: int = 42
    dataset: str = "FD001"
    rul_cap: int = 125
    rolling_window: int = 5
    retrieval_k: int = 4
    paths: Paths = field(default_factory=Paths)
    rf_params: RFParams = field(default_factory=RFParams)
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    gate_thresholds: GateThresholds = field(default_factory=GateThresholds)
    root: Path = REPO_ROOT

    # --- (de)serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        """Plain-dict view (nested dataclasses expanded). ``root`` is excluded so
        the mapping is portable and matches ``pipeline.yaml``'s shape."""
        return {
            "seed": self.seed,
            "dataset": self.dataset,
            "rul_cap": self.rul_cap,
            "rolling_window": self.rolling_window,
            "retrieval_k": self.retrieval_k,
            "paths": dataclasses.asdict(self.paths),
            "rf_params": dataclasses.asdict(self.rf_params),
            "risk_thresholds": dataclasses.asdict(self.risk_thresholds),
            "gate_thresholds": dataclasses.asdict(self.gate_thresholds),
        }

    @classmethod
    def from_dict(cls, data: dict | None, root: Path = REPO_ROOT) -> "PipelineConfig":
        """Build a config from a mapping, filling any missing key with the
        dataclass default. Unknown keys are ignored."""
        data = dict(data or {})

        def _sub(kls, key):
            fields = {f.name for f in dataclasses.fields(kls)}
            given = {k: v for k, v in (data.get(key) or {}).items() if k in fields}
            return kls(**given)

        scalar = {
            k: data[k]
            for k in ("seed", "dataset", "rul_cap", "rolling_window", "retrieval_k")
            if k in data
        }
        return cls(
            paths=_sub(Paths, "paths"),
            rf_params=_sub(RFParams, "rf_params"),
            risk_thresholds=_sub(RiskThresholds, "risk_thresholds"),
            gate_thresholds=_sub(GateThresholds, "gate_thresholds"),
            root=root,
            **scalar,
        )

    @classmethod
    def load(cls, path: Path | str | None = None, root: Path = REPO_ROOT) -> "PipelineConfig":
        """Load config from YAML (defaults if the file is absent)."""
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls(root=root)
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(data, root=root)

    # --- convenience ----------------------------------------------------------
    def path(self, name: str) -> Path:
        """Absolute path for a named entry in :class:`Paths`."""
        return self.paths.resolve(name, self.root)
