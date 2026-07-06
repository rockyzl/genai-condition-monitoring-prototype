"""Run context and per-stage result objects shared by all stages.

:class:`PipelineContext` threads the config, journal, run id, and the resolved
repo root through every stage, and centralises the execute/skip/journal/
provenance boilerplate in :meth:`PipelineContext.run_stage` so each stage
function stays a thin wrapper over existing ``src/`` code. :class:`StageResult`
is the observed outcome (rows, artifacts, seconds, skipped) the manifest renders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.pipeline import provenance
from src.pipeline.config import PipelineConfig
from src.pipeline.journal import Journal
from src.pipeline.specs import STAGE_SPECS, StageSpec


@dataclass
class StageResult:
    """Observed outcome of one stage execution (feeds the manifest)."""

    name: str
    skipped: bool
    seconds: float
    rows: int | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    key_metrics: dict = field(default_factory=dict)

    @property
    def spec(self) -> StageSpec:
        return STAGE_SPECS[self.name]


@dataclass
class StageWork:
    """What a stage's work function returns to the context after doing its job.

    ``rows`` and ``key_metrics`` are observed (for the manifest/journal);
    ``artifacts`` maps each written output path to the small metrics dict that
    the ``artifact`` journal event should carry (may be empty).
    """

    rows: int | None = None
    key_metrics: dict = field(default_factory=dict)
    artifacts: dict[str, dict] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Everything a stage needs: config, journal, run id, repo root, force flag."""

    cfg: PipelineConfig
    journal: Journal
    run_id: str
    force: bool = False
    results: list[StageResult] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.cfg.root

    def rel(self, path: Path) -> str:
        """Repo-relative string for a path (for readable manifest/journal output)."""
        path = Path(path)
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def run_stage(
        self,
        name: str,
        input_paths: list[Path],
        output_paths: list[Path],
        params: dict,
        work: Callable[[], StageWork],
    ) -> StageResult:
        """Execute one stage with skip-check, journaling, provenance, and timing.

        ``work`` does the real job (calling into existing ``src/`` code) and
        returns a :class:`StageWork`. It is only invoked when the stage is not
        current (or ``--force`` is set).
        """
        spec = STAGE_SPECS[name]
        sig = provenance.build_signature(name, input_paths, params, self.cfg.seed, self.root)
        t0 = time.perf_counter()

        if not self.force and provenance.is_stage_current(output_paths, sig):
            self.journal.stage_started(name, spec.what, spec.why)
            self.journal.stage_progress(name, "inputs unchanged — skipping (cached)")
            seconds = time.perf_counter() - t0
            self.journal.stage_done(name, seconds, rows=None, skipped=True)
            result = StageResult(
                name=name,
                skipped=True,
                seconds=seconds,
                inputs=[self.rel(p) for p in input_paths],
                outputs=[self.rel(p) for p in output_paths],
            )
            self.results.append(result)
            return result

        self.journal.stage_started(name, spec.what, spec.why)
        outcome = work()
        for out in output_paths:
            provenance.write_provenance(out, sig)
        for art_path, metrics in outcome.artifacts.items():
            self.journal.artifact(name, self.rel(Path(art_path)), metrics)
        seconds = time.perf_counter() - t0
        self.journal.stage_done(name, seconds, rows=outcome.rows, skipped=False)

        result = StageResult(
            name=name,
            skipped=False,
            seconds=seconds,
            rows=outcome.rows,
            inputs=[self.rel(p) for p in input_paths],
            outputs=[self.rel(p) for p in output_paths],
            key_metrics=outcome.key_metrics,
        )
        self.results.append(result)
        return result
