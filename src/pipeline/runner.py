"""Pipeline runner + CLI.

    python -m src.pipeline run --all
    python -m src.pipeline run --stage s05_model
    python -m src.pipeline run --from s07_predict
    python -m src.pipeline run --all --force        # ignore provenance, rerun all

Executes the selected stages in canonical order, streaming a step-event journal
(``reports/pipeline_journal.jsonl``) and regenerating a per-run
``reports/pipeline_manifest.md`` (declared what/why + observed rows/sizes/seconds
per stage + a mermaid DAG). Unchanged stages skip via provenance unless
``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline import provenance
from src.pipeline.config import PipelineConfig
from src.pipeline.context import PipelineContext
from src.pipeline.journal import Journal
from src.pipeline.specs import STAGE_ORDER, STAGE_SPECS
from src.pipeline.stages import STAGE_FUNCS

#: True dependency edges of the 10-stage DAG (for the manifest's mermaid render).
DAG_EDGES = [
    ("s01_ingest", "s02_eda"),
    ("s01_ingest", "s03_preprocess"),
    ("s03_preprocess", "s04_features"),
    ("s04_features", "s05_model"),
    ("s05_model", "s06_select"),
    ("s05_model", "s07_predict"),
    ("s07_predict", "s08_evidence"),
    ("s08_evidence", "s09_diagnose"),
    ("s09_diagnose", "s10_eval"),
]


def _new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def select_stages(all_: bool, stage: str | None, from_: str | None) -> list[str]:
    """Resolve the CLI selection into an ordered list of stage names."""
    if stage is not None:
        if stage not in STAGE_SPECS:
            raise SystemExit(f"unknown stage {stage!r}; valid: {', '.join(STAGE_ORDER)}")
        return [stage]
    if from_ is not None:
        if from_ not in STAGE_SPECS:
            raise SystemExit(f"unknown stage {from_!r}; valid: {', '.join(STAGE_ORDER)}")
        return STAGE_ORDER[STAGE_ORDER.index(from_):]
    # default / --all
    return list(STAGE_ORDER)


def run_pipeline(
    cfg: PipelineConfig,
    stages: list[str],
    force: bool = False,
    run_id: str | None = None,
) -> PipelineContext:
    """Execute ``stages`` in canonical order, returning the populated context."""
    run_id = run_id or _new_run_id()
    journal = Journal(cfg.path("journal"), run_id)
    ctx = PipelineContext(cfg=cfg, journal=journal, run_id=run_id, force=force)

    t0 = datetime.now(timezone.utc)
    journal.run_started(stages)
    for name in stages:
        STAGE_FUNCS[name](ctx)
    seconds = (datetime.now(timezone.utc) - t0).total_seconds()

    n_skipped = sum(1 for r in ctx.results if r.skipped)
    journal.run_done(
        stages_run=len(ctx.results) - n_skipped,
        stages_skipped=n_skipped,
        seconds=seconds,
    )
    write_manifest(ctx)
    return ctx


# --- manifest ----------------------------------------------------------------
def _bytes_of(paths: list[str], root: Path) -> int:
    total = 0
    for rel in paths:
        p = root / rel
        if p.exists() and p.is_file():
            total += p.stat().st_size
    return total


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _mermaid(selected: set[str]) -> list[str]:
    lines = ["```mermaid", "flowchart TD"]
    for name in STAGE_ORDER:
        label = name.replace("_", " ", 1)
        mark = "" if name in selected else " · skipped-not-selected"
        lines.append(f'  {name}["{label}{mark}"]')
    for a, b in DAG_EDGES:
        lines.append(f"  {a} --> {b}")
    lines.append("```")
    return lines


def build_manifest(ctx: PipelineContext) -> str:
    """Render the per-run manifest markdown from the context's stage results."""
    root = ctx.root
    ran = [r.name for r in ctx.results]
    lines: list[str] = []
    lines.append("# Pipeline Run Manifest\n")
    lines.append(
        f"- Run id: `{ctx.run_id}`  ·  stages executed: **{len(ctx.results)}** "
        f"(skipped: **{sum(1 for r in ctx.results if r.skipped)}**)"
    )
    lines.append(
        f"- Seed: **{ctx.cfg.seed}**  ·  dataset: **{ctx.cfg.dataset}**  ·  "
        f"RUL cap: **{ctx.cfg.rul_cap}**  ·  git: `{provenance.git_sha(root) or 'n/a'}`"
    )
    lines.append(
        "- Journal: `" + ctx.rel(ctx.cfg.path("journal")) + "` (append-only NDJSON, "
        "one line per step event)\n"
    )

    lines.append("## Stage DAG\n")
    lines.extend(_mermaid(set(ran)))
    lines.append("")

    lines.append("## Stage cards\n")
    for r in ctx.results:
        spec = STAGE_SPECS[r.name]
        status = "⏭ skipped (cached)" if r.skipped else "✓ ran"
        out_bytes = _bytes_of(r.outputs, root)
        lines.append(f"### {r.name} — {status}\n")
        lines.append(f"**What.** {spec.what}\n")
        lines.append(f"**Why.** {spec.why}\n")
        lines.append(f"**功能.** {spec.zh_what}\n")
        lines.append(f"**目的.** {spec.zh_why}\n")
        rows = "n/a" if r.rows is None else f"{r.rows:,}"
        lines.append(
            f"- Observed: rows **{rows}**, outputs **{_fmt_size(out_bytes)}**, "
            f"time **{r.seconds:.3f}s**"
        )
        if r.key_metrics:
            shown = {
                k: v
                for k, v in r.key_metrics.items()
                if not isinstance(v, (list, dict)) or k == "risk_band_counts"
            }
            if shown:
                kv = ", ".join(f"{k}={v}" for k, v in shown.items())
                lines.append(f"- Key metrics: {kv}")
        lines.append(f"- Inputs: {', '.join(f'`{p}`' for p in r.inputs) or '—'}")
        lines.append(f"- Outputs: {', '.join(f'`{p}`' for p in r.outputs) or '—'}")
        lines.append("- Assumptions:")
        for a in spec.assumptions:
            lines.append(f"  - {a}")
        lines.append("")
    return "\n".join(lines)


def write_manifest(ctx: PipelineContext) -> Path:
    path = ctx.cfg.path("manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_manifest(ctx))
    return path


# --- CLI ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Deterministic condition-monitoring pipeline runner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run pipeline stages")
    sel = run_p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="run all 10 stages (default)")
    sel.add_argument("--stage", metavar="sXX", help="run a single stage")
    sel.add_argument("--from", dest="from_", metavar="sXX", help="run from a stage to the end")
    run_p.add_argument("--force", action="store_true", help="ignore provenance; rerun")
    run_p.add_argument("--config", metavar="PATH", help="path to pipeline.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        return 2

    cfg = PipelineConfig.load(args.config)
    stages = select_stages(args.all, args.stage, args.from_)
    ctx = run_pipeline(cfg, stages, force=args.force)

    n_skipped = sum(1 for r in ctx.results if r.skipped)
    print(
        f"[pipeline] run {ctx.run_id}: {len(ctx.results)} stages "
        f"({len(ctx.results) - n_skipped} ran, {n_skipped} skipped)"
    )
    for r in ctx.results:
        tag = "skip" if r.skipped else "ran "
        rows = "" if r.rows is None else f"rows={r.rows}"
        print(f"  [{tag}] {r.name:<14} {r.seconds:6.3f}s  {rows}")
    print(f"[pipeline] manifest -> {ctx.rel(cfg.path('manifest'))}")
    print(f"[pipeline] journal  -> {ctx.rel(cfg.path('journal'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
