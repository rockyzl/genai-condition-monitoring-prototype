"""Autopilot supervisor — the deterministic pipeline agent with HITL gates.

The supervisor walks the fixed 10-stage DAG (``s01_ingest`` → ``s10_eval``) with a
per-stage state machine:

    EXECUTE (run the stage via the Phase-A runner)
        → VALIDATE (a reliability gate)
            → PASS   : log and continue
            → CARD   : raise a decision card for the human
            → HALT   : stop the run (leakage / governance / schema violation)

It reuses :class:`src.pipeline.context.PipelineContext` and ``STAGE_FUNCS`` for
execution — the same skip/provenance/journal machinery Phase A shipped — and
interleaves gates between stages, streaming stage *and* gate events through one
:class:`src.pipeline.journal.Journal` (so a single monotonic sequence renders in
the app). Decisions are grounded in artifacts and reproducible.

Bounded autonomy (``--autonomy``):

* ``dry-run`` — walk everything, raise **no** cards, but report what *would* raise.
* ``gated`` (default) — raise blocking cards to ``autopilot_inbox/pending/`` and
  stop; a human answer in ``autopilot_inbox/answered/`` resumes the walk (earlier
  stages skip via provenance). ``--yes-safe-defaults`` accepts a card's safe
  default at the CLI, except for cards in ``NEVER_SAFE_DEFAULT_KINDS``.
* ``auto`` — auto-apply safe defaults, **except** triage and sign-off cards,
  which never auto-pass even here.

Checkpoint/resume is provenance-native: re-invoking the supervisor re-walks from
``s01`` with unchanged stages skipping, and any card that now has an answer file
resolves instead of blocking. State lives in ``autopilot_state.json``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.agent import cards as cards_mod
from src.agent.cards import Card
from src.agent.registry import Registry
from src.agent.trace import Trace, config_hash
from src.pipeline.config import PipelineConfig
from src.pipeline.context import PipelineContext, StageResult
from src.pipeline.journal import Journal
from src.pipeline.specs import STAGE_ORDER, STAGE_SPECS
from src.pipeline.stages import STAGE_FUNCS

# Repo-relative artifact contracts the gates read (resolved against cfg.root).
PREDS_REL = "data/processed/test_predictions.csv"
METRICS_REL = "reports/metrics_model.json"
EVAL_SUMMARY_REL = "reports/evaluation_summary.md"

# Dispositions
PASS, CARD, HALT, SKIP = "PASS", "CARD", "HALT", "SKIP"


@dataclass(frozen=True)
class AgentGateConfig:
    """Reliability-gate thresholds. Hashed into the trace (anti-silent-weakening).

    ``min_history_cycles`` is sourced from the pipeline config's
    ``gate_thresholds`` so the two layers agree; the rest are agent-owned and
    live here rather than mutating the shared pipeline config.
    """

    inspect_first_max: float = 10.0
    min_history_cycles: int = 20
    ridge_floor_rmse: float | None = None
    champion_margin_rmse: float = 1.0
    max_band_fraction: float = 0.95
    min_distinct_bands: int = 2
    recompute_tol: float = 0.10
    expected_test_units: int | None = None

    @classmethod
    def from_pipeline(cls, cfg: PipelineConfig) -> "AgentGateConfig":
        gt = cfg.gate_thresholds
        return cls(
            min_history_cycles=gt.min_history_cycles,
            ridge_floor_rmse=gt.ridge_floor_rmse,
            champion_margin_rmse=(
                gt.champion_margin_rmse if gt.champion_margin_rmse is not None else 1.0
            ),
        )

    def hash(self) -> str:
        return config_hash(dataclasses.asdict(self))


@dataclass
class GateOutcome:
    """One gate's verdict for one stage."""

    stage: str
    gate: str
    disposition: str
    card: Card | None = None
    detail: str = ""


@dataclass
class RunReport:
    """The full result of an autopilot run — what the CLI/app renders."""

    run_id: str
    autonomy: str
    status: str  # "done" | "awaiting_input" | "halted"
    thresholds_hash: str
    decisions: list[tuple] = field(default_factory=list)  # (stage, gate, disp, kind)
    stages: list[dict] = field(default_factory=list)
    cards_pending: list[dict] = field(default_factory=list)
    cards_resolved: list[dict] = field(default_factory=list)
    would_raise: list[dict] = field(default_factory=list)
    halt: dict | None = None
    digest: dict | None = None
    trace_path: str | None = None
    journal_path: str | None = None
    state_path: str | None = None
    inbox_pending_dir: str | None = None


class Autopilot:
    """The supervisor. One instance per run; :meth:`run` walks the DAG."""

    def __init__(
        self,
        cfg: PipelineConfig,
        autonomy: str = "gated",
        yes_safe_defaults: bool = False,
        force: bool = False,
        out_dir: Path | str | None = None,
        gcfg: AgentGateConfig | None = None,
    ):
        if autonomy not in ("gated", "auto", "dry-run"):
            raise ValueError(
                f"unknown autonomy {autonomy!r}; valid: gated, auto, dry-run"
            )
        self.cfg = cfg
        self.autonomy = autonomy
        self.yes_safe_defaults = yes_safe_defaults
        self.force = force
        self.gcfg = gcfg or AgentGateConfig.from_pipeline(cfg)
        self.out_dir = Path(out_dir) if out_dir else cfg.path("reports")
        self.reg = Registry(cfg)

    # --- paths ---------------------------------------------------------------
    @property
    def journal_path(self) -> Path:
        return self.out_dir / "autopilot_journal.jsonl"

    @property
    def inbox_dir(self) -> Path:
        return self.out_dir / "autopilot_inbox"

    @property
    def pending_dir(self) -> Path:
        return self.inbox_dir / "pending"

    @property
    def answered_dir(self) -> Path:
        return self.inbox_dir / "answered"

    @property
    def state_path(self) -> Path:
        return self.out_dir / "autopilot_state.json"

    # --- run -----------------------------------------------------------------
    def run(self) -> RunReport:
        run_id = "auto_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.answered_dir.mkdir(parents=True, exist_ok=True)

        journal = Journal(self.journal_path, run_id)
        ctx = PipelineContext(cfg=self.cfg, journal=journal, run_id=run_id, force=self.force)
        trace = Trace(
            run_id=run_id,
            kind="autopilot",
            autonomy=self.autonomy,
            thresholds_hash=self.gcfg.hash(),
        )
        report = RunReport(
            run_id=run_id,
            autonomy=self.autonomy,
            status="done",
            thresholds_hash=self.gcfg.hash(),
            journal_path=str(self.journal_path),
            state_path=str(self.state_path),
            inbox_pending_dir=str(self.pending_dir),
        )

        journal.run_started(STAGE_ORDER)
        completed: list[str] = []

        for stage in STAGE_ORDER:
            result = STAGE_FUNCS[stage](ctx)  # EXECUTE (skips via provenance)
            self._record_stage_exec(trace, result)
            stage_row = {
                "stage": stage,
                "skipped": result.skipped,
                "seconds": round(result.seconds, 4),
                "gates": [],
            }
            outcomes = self._validate(stage, ctx, result, journal, trace)
            for outcome in outcomes:
                trace.record_gate(
                    outcome.stage,
                    outcome.gate,
                    outcome.disposition,
                    card_id=outcome.card.id if outcome.card else None,
                    detail=outcome.detail,
                )
                stage_row["gates"].append(
                    {
                        "gate": outcome.gate,
                        "disposition": outcome.disposition,
                        "card_id": outcome.card.id if outcome.card else None,
                        "detail": outcome.detail,
                    }
                )
                kind = outcome.card.kind if outcome.card else None
                report.decisions.append(
                    (outcome.stage, outcome.gate, outcome.disposition, kind)
                )

                if outcome.disposition in (PASS, SKIP):
                    journal.stage_progress(
                        stage, f"gate {outcome.gate}: {outcome.disposition.lower()}"
                    )
                    continue

                if outcome.disposition == HALT:
                    journal.halt(stage, outcome.gate, outcome.detail)
                    report.status = "halted"
                    report.halt = {
                        "stage": stage,
                        "gate": outcome.gate,
                        "reason": outcome.detail,
                    }
                    stage_row["gates"][-1]["halted"] = True
                    report.stages.append(stage_row)
                    self._finalize(report, trace, ctx, completed, journal, done=False)
                    return report

                # disposition == CARD
                decision = self._handle_card(outcome.card, stage, journal, trace, report)
                if decision == "pending":
                    report.status = "awaiting_input"
                    report.stages.append(stage_row)
                    self._save_state(report, completed, stage, ctx)
                    trace.write(self.out_dir)
                    report.trace_path = str(
                        self.out_dir / f"agent_trace_{run_id}.json"
                    )
                    return report

            completed.append(stage)
            report.stages.append(stage_row)

        # reached the end: everything passed or was resolved
        self._finalize(report, trace, ctx, completed, journal, done=True)
        return report

    # --- stage execution recording ------------------------------------------
    def _record_stage_exec(self, trace: Trace, result: StageResult) -> None:
        # Represent each stage EXECUTE as a run_stage-shaped record so the trace
        # is a single, uniform tool/gate stream (grounding for run_stage claims).
        from src.agent.registry import ToolResult, digest

        out = {
            "stage": result.name,
            "skipped": result.skipped,
            "rows": result.rows,
            "seconds": round(result.seconds, 4),
            "outputs": result.outputs,
            "key_metrics": result.key_metrics,
        }
        trace.record_tool(
            ToolResult(
                tool="run_stage",
                args={"stage": result.name},
                ok=True,
                output=out,
                digest=digest(out),
                seconds=round(result.seconds, 4),
            )
        )

    # --- validation gates ----------------------------------------------------
    def _validate(
        self,
        stage: str,
        ctx: PipelineContext,
        result: StageResult,
        journal: Journal,
        trace: Trace,
    ) -> list[GateOutcome]:
        fn = getattr(self, f"_gate_{stage}", None)
        if fn is None:
            return [GateOutcome(stage, "none", PASS, detail="no gate declared")]
        return fn(ctx, result, journal, trace)

    def _read_json(self, rel_or_path) -> dict | None:
        p = Path(rel_or_path)
        if not p.is_absolute():
            p = (self.cfg.root / p).resolve()
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (ValueError, OSError):
            return None

    def _n_test_units(self) -> int | None:
        m = self._read_json(METRICS_REL)
        if m and "n_test_units" in m:
            return int(m["n_test_units"])
        return None

    def _gate_s01_ingest(self, ctx, result, journal, trace) -> list[GateOutcome]:
        man = self._read_json(self.cfg.path("data_processed") / "ingest_manifest.json")
        if not man:
            return [GateOutcome("s01_ingest", "ingest_schema", HALT,
                                detail="ingest manifest missing — cannot validate schema")]
        if man.get("train_rows", 0) <= 0 or man.get("n_test_units", 0) <= 0:
            return [GateOutcome("s01_ingest", "ingest_schema", HALT,
                                detail="ingest produced zero train/test rows")]
        if (self.gcfg.expected_test_units is not None
                and man.get("n_test_units") != self.gcfg.expected_test_units):
            return [GateOutcome("s01_ingest", "ingest_schema", HALT,
                                detail=f"expected {self.gcfg.expected_test_units} test "
                                f"units, got {man.get('n_test_units')}")]
        return [GateOutcome("s01_ingest", "ingest_schema", PASS,
                            detail=f"{man['n_train_units']} train + "
                            f"{man['n_test_units']} test units")]

    def _gate_s02_eda(self, ctx, result, journal, trace) -> list[GateOutcome]:
        p = self.cfg.path("eda") / "eda_summary.json"
        if not p.exists():
            return [GateOutcome("s02_eda", "eda_sanity", HALT, detail="EDA summary missing")]
        return [GateOutcome("s02_eda", "eda_sanity", PASS)]

    def _gate_s03_preprocess(self, ctx, result, journal, trace) -> list[GateOutcome]:
        s = self._read_json(self.cfg.path("data_processed") / "preprocess_summary.json")
        if not s:
            return [GateOutcome("s03_preprocess", "leakage_canary", HALT,
                                detail="preprocess summary missing")]
        guards = s.get("leakage_guards") or []
        joined = " ".join(guards).lower()
        if not guards or "group" not in joined:
            return [GateOutcome("s03_preprocess", "leakage_canary", HALT,
                                detail="declared leakage guards missing (group-by-unit CV)")]
        return [GateOutcome("s03_preprocess", "leakage_canary", PASS,
                            detail=f"{len(guards)} leakage guards declared and intact")]

    def _gate_s04_features(self, ctx, result, journal, trace) -> list[GateOutcome]:
        spec = self._read_json(self.cfg.path("data_processed") / "feature_spec.json")
        if not spec or not spec.get("feature_cols"):
            return [GateOutcome("s04_features", "feature_provenance", HALT,
                                detail="feature spec missing or empty")]
        return [GateOutcome("s04_features", "feature_provenance", PASS,
                            detail=f"{spec['n_features']} features")]

    def _gate_s05_model(self, ctx, result, journal, trace) -> list[GateOutcome]:
        meta = self._read_json(self.cfg.path("data_processed") / "model_meta.json")
        spec = self._read_json(self.cfg.path("data_processed") / "feature_spec.json")
        model_p = self.cfg.path("models") / "rul_baseline.joblib"
        if not meta or not model_p.exists():
            return [GateOutcome("s05_model", "model_provenance", HALT,
                                detail="model artifact or meta missing")]
        if spec and meta.get("n_features") != spec.get("n_features"):
            return [GateOutcome("s05_model", "model_provenance", HALT,
                                detail=f"feature-count drift: model {meta.get('n_features')} "
                                f"vs spec {spec.get('n_features')} (possible leakage)")]
        from src.pipeline import provenance
        if provenance.read_provenance(model_p) is None:
            return [GateOutcome("s05_model", "model_provenance", HALT,
                                detail="model has no provenance sidecar")]
        return [GateOutcome("s05_model", "model_provenance", PASS,
                            detail=f"{meta['n_features']} features, provenance stamped")]

    def _gate_s06_select(self, ctx, result, journal, trace) -> list[GateOutcome]:
        sel = None
        sel_rel = None
        for rel in ("reports/model_selection.json", "data/processed/model_selection.json"):
            data = self._read_json(rel)
            if data:
                sel, sel_rel = data, rel
                break
        if not sel:
            return [GateOutcome("s06_select", "champion_beats_floor", SKIP,
                                detail="model_selection.json absent — champion-beats-floor "
                                "gate skipped, recorded")]
        verdict = sel.get("verdict") or {}
        # Bake-off schema: prefer the explicit verdict fields Phase B records.
        beats = verdict.get("beats_floor")
        margin = verdict.get("floor_gap_cycles")
        bar = verdict.get("champion_margin_rmse", self.gcfg.champion_margin_rmse)
        floor_rmse = verdict.get("floor_rmse")
        if beats is None or margin is None:
            return [GateOutcome("s06_select", "champion_beats_floor", SKIP,
                                detail="selection file lacks a verdict (beats_floor / "
                                "floor_gap_cycles) — gate skipped, recorded")]
        champ_rmse = (floor_rmse - margin) if floor_rmse is not None else None
        if beats is False or margin < 0:
            return [GateOutcome("s06_select", "champion_beats_floor", HALT,
                                detail=f"champion does not beat the Ridge floor "
                                f"(gap {margin:.2f} cycles)")]
        if margin < bar:
            card = cards_mod.build_champion_card(
                self.cfg, sel_rel, champ_rmse or 0.0, floor_rmse or 0.0, margin, bar
            )
            return [GateOutcome("s06_select", "champion_beats_floor", CARD, card=card,
                                detail=f"marginal win: {margin:.2f} < bar {bar:.2f} cycles")]
        return [GateOutcome("s06_select", "champion_beats_floor", PASS,
                            detail=f"champion beats floor by {margin:.2f} cycles "
                            f"(bar {bar:.2f})")]

    def _gate_s07_predict(self, ctx, result, journal, trace) -> list[GateOutcome]:
        import pandas as pd

        preds_p = (self.cfg.root / PREDS_REL).resolve()
        if not preds_p.exists():
            return [GateOutcome("s07_predict", "prediction_distribution", HALT,
                                detail="predictions artifact missing")]
        df = pd.read_csv(preds_p)
        counts = df["risk_band"].value_counts().to_dict()
        n = len(df)
        outcomes: list[GateOutcome] = []

        # Gate 1: distribution sanity (not degenerate).
        distinct = len([b for b, c in counts.items() if c > 0])
        top_frac = (max(counts.values()) / n) if n else 1.0
        if distinct < self.gcfg.min_distinct_bands or top_frac > self.gcfg.max_band_fraction:
            card = cards_mod.build_data_quality_card(
                self.cfg, PREDS_REL, "count:risk_band=low",
                "Risk distribution looks degenerate (nearly all one band)",
                "风险分布看起来退化了（几乎全在同一档）",
            )
            outcomes.append(GateOutcome("s07_predict", "prediction_distribution", CARD,
                                        card=card, detail=f"distinct bands={distinct}, "
                                        f"top fraction={top_frac:.2f}"))
        else:
            outcomes.append(GateOutcome("s07_predict", "prediction_distribution", PASS,
                                        detail=f"bands={counts}"))

        # Gate 2: high-risk triage (the flagship card). Ground via a real tool call.
        n_high = int((df["risk_band"] == "high").sum())
        if n_high > 0:
            self._call_and_trace(trace, "list_units_by_risk", {"band": "high"})
            card = cards_mod.build_triage_card(
                self.cfg, PREDS_REL, METRICS_REL, self.gcfg.inspect_first_max
            )
            outcomes.append(GateOutcome("s07_predict", "high_risk_triage", CARD,
                                        card=card, detail=f"{n_high} high-risk units"))
        else:
            outcomes.append(GateOutcome("s07_predict", "high_risk_triage", PASS,
                                        detail="no high-risk units"))
        return outcomes

    def _gate_s08_evidence(self, ctx, result, journal, trace) -> list[GateOutcome]:
        man = self._read_json(self.cfg.path("evidence") / "_evidence_manifest.json")
        n_units = self._n_test_units()
        if not man:
            return [GateOutcome("s08_evidence", "evidence_completeness", HALT,
                                detail="evidence manifest missing")]
        n_rec = man.get("n_records", 0)
        if n_units is not None and n_rec != n_units:
            card = cards_mod.build_data_quality_card(
                self.cfg, "reports/metrics_model.json", "n_test_units",
                f"Evidence built for {n_rec} units but {n_units} were scored",
                f"证据只覆盖 {n_rec} 台，但打分的是 {n_units} 台",
            )
            return [GateOutcome("s08_evidence", "evidence_completeness", CARD, card=card,
                                detail=f"{n_rec}/{n_units} evidence records")]
        return [GateOutcome("s08_evidence", "evidence_completeness", PASS,
                            detail=f"{n_rec} evidence records")]

    def _gate_s09_diagnose(self, ctx, result, journal, trace) -> list[GateOutcome]:
        man = self._read_json(self.cfg.path("diagnostics") / "_diagnostics_manifest.json")
        if not man:
            return [GateOutcome("s09_diagnose", "diagnosis_governance", HALT,
                                detail="diagnostics manifest missing")]
        n_rep = man.get("n_reports", 0)
        n_cit = man.get("n_with_citations", 0)
        if n_rep == 0 or n_cit < n_rep:
            return [GateOutcome("s09_diagnose", "diagnosis_governance", HALT,
                                detail=f"governance violation: {n_cit}/{n_rep} reports cited "
                                "(a diagnosis without citations may not ship)")]
        # Spot-check one report enforces the human-review flag.
        diag_dir = self.cfg.path("diagnostics")
        sample = sorted(diag_dir.glob("unit_*.json"))
        if sample:
            rep = self._read_json(sample[0])
            if rep is not None and rep.get("human_review_required") is not True:
                return [GateOutcome("s09_diagnose", "diagnosis_governance", HALT,
                                    detail="a diagnosis has human_review_required != true")]
        return [GateOutcome("s09_diagnose", "diagnosis_governance", PASS,
                            detail=f"{n_cit}/{n_rep} reports cited, human review enforced")]

    def _gate_s10_eval(self, ctx, result, journal, trace) -> list[GateOutcome]:
        import math

        import pandas as pd

        outcomes: list[GateOutcome] = []
        preds_p = (self.cfg.root / PREDS_REL).resolve()
        metrics = self._read_json(METRICS_REL)
        if not preds_p.exists() or not metrics:
            outcomes.append(GateOutcome("s10_eval", "eval_recompute_match", HALT,
                                        detail="predictions or metrics missing for recompute"))
            return outcomes
        df = pd.read_csv(preds_p)
        err = df["pred_rul"] - df["true_rul"]
        rmse = float(math.sqrt((err ** 2).mean()))
        ref = (metrics.get("metrics_vs_uncapped_truth") or {}).get("rmse")
        if ref is None or abs(rmse - float(ref)) > self.gcfg.recompute_tol:
            outcomes.append(GateOutcome("s10_eval", "eval_recompute_match", HALT,
                                        detail=f"recomputed RMSE {rmse:.3f} != recorded "
                                        f"{ref} (artifact drift)"))
            return outcomes
        outcomes.append(GateOutcome("s10_eval", "eval_recompute_match", PASS,
                                    detail=f"recomputed RMSE {rmse:.2f} matches recorded"))

        # Report sign-off card — always raised; a human must sign.
        n_units = self._n_test_units() or int(len(df))
        card = cards_mod.build_signoff_card(self.cfg, METRICS_REL, EVAL_SUMMARY_REL, n_units)
        outcomes.append(GateOutcome("s10_eval", "report_signoff", CARD, card=card,
                                    detail="report complete; needs human signature"))
        return outcomes

    def _call_and_trace(self, trace: Trace, tool: str, args: dict) -> None:
        """Run a registry tool for card grounding and record it in the trace."""
        res = self.reg.call(tool, args)
        trace.record_tool(res)

    # --- card handling (autonomy policy) ------------------------------------
    def _handle_card(
        self, card: Card, stage: str, journal: Journal, trace: Trace, report: RunReport
    ) -> str:
        """Apply the autonomy policy to a raised card.

        Returns ``"resolved"`` (continue the walk) or ``"pending"`` (block).
        """
        trace.add_card(card.to_dict())

        if self.autonomy == "dry-run":
            report.would_raise.append(
                {"card_id": card.id, "kind": card.kind, "priority": card.priority,
                 "verdict_en": card.verdict_en,
                 "would_block": not card.auto_passable()}
            )
            journal.stage_progress(
                stage, f"[dry-run] would raise {card.kind} ({card.id})"
            )
            return "resolved"  # dry-run never blocks

        journal.gate_raised(card.id, card.kind, stage, card.payload_summary())

        # An explicit human answer always wins (this is what makes resume work).
        answer = self._read_answer(card)
        if answer is not None:
            self._resolve(card, answer, stage, journal, trace, report, source="answered")
            return "resolved"

        # auto: apply safe default only for auto-passable cards.
        if self.autonomy == "auto" and card.auto_passable():
            self._resolve(card, card.safe_action().id, stage, journal, trace, report,
                          source="auto")
            return "resolved"

        # gated + --yes-safe-defaults: apply safe default unless the kind forbids it.
        if (self.autonomy == "gated" and self.yes_safe_defaults
                and card.safe_default_applicable()):
            self._resolve(card, card.safe_action().id, stage, journal, trace, report,
                          source="yes-safe-defaults")
            return "resolved"

        # otherwise block: write the pending card and stop.
        self._write_pending(card)
        report.cards_pending.append(card.to_dict())
        return "pending"

    def _resolve(self, card, action_id, stage, journal, trace, report, source) -> None:
        journal.gate_resolved(card.id, action_id, stage)
        report.cards_resolved.append(
            {"card_id": card.id, "kind": card.kind, "action": action_id, "source": source}
        )
        # move any pending copy to answered for auditability
        pend = self.pending_dir / f"{card.id}.json"
        if pend.exists():
            pend.unlink()

    def _write_pending(self, card: Card) -> Path:
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        p = self.pending_dir / f"{card.id}.json"
        p.write_text(json.dumps(card.to_dict(), indent=2, ensure_ascii=False))
        return p

    def _read_answer(self, card: Card) -> str | None:
        """Return a valid answered action id for this card, or None."""
        p = self.answered_dir / f"{card.id}.json"
        if not p.exists():
            return None
        try:
            ans = json.loads(p.read_text())
        except (ValueError, OSError):
            return None
        action_id = ans.get("action")
        valid = {a.id for a in card.actions}
        if action_id in valid:
            return action_id
        return None

    # --- finalize ------------------------------------------------------------
    def _finalize(self, report, trace, ctx, completed, journal, done: bool) -> None:
        if done:
            n_flagged = None
            m = self._read_json(METRICS_REL)
            if m:
                bands = m.get("risk_band_counts", {})
                n_flagged = int(bands.get("high", 0)) + int(bands.get("medium", 0))
            preds_p = (self.cfg.root / PREDS_REL).resolve()
            if preds_p.exists() and n_flagged is not None:
                report.digest = cards_mod.build_healthy_digest(self.cfg, PREDS_REL, n_flagged)
                trace.add_card(report.digest)
        n_skipped = sum(1 for r in ctx.results if r.skipped)
        journal.run_done(
            stages_run=len(ctx.results) - n_skipped,
            stages_skipped=n_skipped,
            seconds=sum(r.seconds for r in ctx.results),
        )
        self._save_state(report, completed, None, ctx)
        trace.write(self.out_dir)
        report.trace_path = str(self.out_dir / f"agent_trace_{report.run_id}.json")

    def _save_state(self, report, completed, cursor, ctx) -> None:
        state = {
            "run_id": report.run_id,
            "autonomy": self.autonomy,
            "status": report.status,
            "thresholds_hash": report.thresholds_hash,
            "cursor": cursor,
            "completed_stages": list(completed),
            "decisions": [list(d) for d in report.decisions],
            "cards_pending": [c["id"] for c in report.cards_pending],
            "cards_resolved": [c["card_id"] for c in report.cards_resolved],
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def card_to_stage(card: Card) -> str:
    """Map a card kind back to the stage that raises it (for journal events)."""
    return {
        "data_quality_exception": "s01_ingest",
        "champion_confirmation": "s06_select",
        "high_risk_triage": "s07_predict",
        "report_signoff": "s10_eval",
    }.get(card.kind, "autopilot")


# Re-export for callers that only import the module-level helpers.
__all__ = ["Autopilot", "AgentGateConfig", "RunReport", "GateOutcome"]
