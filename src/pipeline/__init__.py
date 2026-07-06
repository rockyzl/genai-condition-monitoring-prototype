"""Explicit, deterministic 10-stage condition-monitoring pipeline.

A thin orchestration layer over the existing ``src/`` modules: every stage
declares a human-readable :class:`~src.pipeline.specs.StageSpec`, records
provenance for skip/idempotency, and streams a step-event journal. See
``docs/expansion-plan.md`` (Phase A) and ``config/pipeline.yaml``.

    python -m src.pipeline run --all
"""

from __future__ import annotations

from src.pipeline.config import PipelineConfig

__all__ = ["PipelineConfig"]
