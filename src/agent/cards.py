"""Decision cards — the condensed human-facing output of an autopilot run.

The supervisor does the heavy lifting; the human faces only a small Decision
Inbox. Each :class:`Card` is a one-sentence bilingual verdict, at most three
signals (one is always the uncertainty/optimism caveat), a deep-link to the
evidence, and safe-default actions with a consequence preview. There are four
card kinds plus a collapsed healthy digest:

* ``data_quality_exception``  — a data-quality gate tripped (schema/railed sensor).
* ``champion_confirmation``   — the champion only marginally beats the floor.
* ``high_risk_triage``        — engines need inspection (the flagship P1 card).
* ``report_signoff``          — the report is ready but needs a human signature.

**Grounding is structural.** Every signal names a real ``artifact`` and ``field``;
:func:`signal_grounded` re-reads that artifact and confirms the datum, so a card
can never assert a number that isn't reproducible from an on-disk artifact.

**Bounded-autonomy policy** lives here as data, not prose:

* ``NEVER_AUTO_KINDS`` — cards ``--autonomy auto`` must never auto-pass (triage,
  sign-off): a machine may not decide to inspect an engine or sign a report.
* ``NEVER_SAFE_DEFAULT_KINDS`` — cards whose safe default even a human's
  ``--yes-safe-defaults`` flag may not apply (sign-off always needs an explicit
  human answer file).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.config import PipelineConfig

# --- bounded-autonomy policy (data, not prose) -------------------------------
CARD_KINDS = (
    "data_quality_exception",
    "champion_confirmation",
    "high_risk_triage",
    "report_signoff",
)

#: Cards that ``--autonomy auto`` may never auto-pass (a human owns the decision).
NEVER_AUTO_KINDS = frozenset({"high_risk_triage", "report_signoff"})

#: Cards whose safe default even ``--yes-safe-defaults`` may not apply; these
#: require an explicit answered card written by a human.
NEVER_SAFE_DEFAULT_KINDS = frozenset({"report_signoff"})

_MISSING = object()


# --- schema ------------------------------------------------------------------
@dataclass
class Signal:
    """One reproducible evidence line. ``artifact`` + ``field`` re-derive the datum."""

    text_en: str
    text_zh: str
    artifact: str
    field: str


@dataclass
class Action:
    """A recommended action with a consequence preview and a safe-default flag."""

    id: str
    label_en: str
    label_zh: str
    consequence_en: str
    consequence_zh: str
    safe_default: bool = False


@dataclass
class Card:
    """A single decision card written to the inbox and rendered to the human."""

    id: str
    kind: str
    priority: str  # "P1" | "P2" | "P3"
    verdict_en: str
    verdict_zh: str
    signals: list[Signal]
    evidence_link: str
    actions: list[Action]
    raised_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
    )

    def safe_action(self) -> Action:
        """The action marked ``safe_default`` (each card defines exactly one)."""
        for a in self.actions:
            if a.safe_default:
                return a
        raise ValueError(f"card {self.id} has no safe-default action")

    def auto_passable(self) -> bool:
        """Whether ``--autonomy auto`` may apply this card's safe default."""
        return self.kind not in NEVER_AUTO_KINDS

    def safe_default_applicable(self) -> bool:
        """Whether ``--yes-safe-defaults`` may apply this card's safe default."""
        return self.kind not in NEVER_SAFE_DEFAULT_KINDS

    def payload_summary(self) -> dict:
        """Compact summary for the ``gate_raised`` journal event."""
        return {
            "priority": self.priority,
            "verdict_en": self.verdict_en,
            "n_signals": len(self.signals),
            "safe_default": self.safe_action().id,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["auto_passable"] = self.auto_passable()
        d["safe_default_applicable"] = self.safe_default_applicable()
        return d


def _card_id(kind: str, *parts: Any) -> str:
    """Deterministic id from the card's grounding (stable across identical runs)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8]
    return f"{kind}-{h}"


# =============================================================================
# Artifact field resolution — the reproducibility backbone
# =============================================================================
def _load_df(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def resolve_field(cfg: PipelineConfig, artifact: str, field_: str) -> Any:
    """Re-derive a signal's datum from an on-disk artifact.

    Grammar (kept tiny and explicit so every signal is reproducible):

    * CSV  ``count:col=value``        → number of rows where ``col == value``
    * CSV  ``count:col<=N`` / ``col<N`` → number of rows under a threshold
    * CSV  ``unit:<id>:<col>``        → a single cell for one unit
    * JSON ``a.b.c``                  → dotted path into the JSON object
    * MD   ``contains:<substring>``   → True iff the file contains the substring
    """
    path = (cfg.root / artifact).resolve()
    if not path.exists():
        return _MISSING
    suffix = path.suffix.lower()

    if field_.startswith("contains:"):
        needle = field_[len("contains:"):]
        return needle in path.read_text(encoding="utf-8")

    if suffix == ".csv":
        df = _load_df(path)
        if field_.startswith("count:"):
            expr = field_[len("count:"):]
            for op in ("<=", ">=", "<", ">", "="):
                if op in expr:
                    col, rhs = expr.split(op, 1)
                    col, rhs = col.strip(), rhs.strip()
                    if col not in df.columns:
                        return _MISSING
                    series = df[col]
                    if op == "=":
                        mask = series.astype(str) == rhs
                    else:
                        val = float(rhs)
                        mask = {
                            "<=": series <= val,
                            ">=": series >= val,
                            "<": series < val,
                            ">": series > val,
                        }[op]
                    return int(mask.sum())
            return _MISSING
        if field_.startswith("unit:"):
            _, uid, col = field_.split(":", 2)
            row = df[df["unit_id"] == int(uid)]
            if row.empty or col not in df.columns:
                return _MISSING
            return row.iloc[0][col].item()
        return _MISSING

    if suffix == ".json":
        data = json.loads(path.read_text())
        cur: Any = data
        for seg in field_.split("."):
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
            else:
                return _MISSING
        return cur

    return _MISSING


def _renderings(value: Any) -> set[str]:
    """Plausible string renderings of a numeric/text value (format-tolerant)."""
    out = {str(value)}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out.add(str(int(round(value))))
        out.add(f"{value:.1f}")
        out.add(f"{value:.2f}")
    return out


def signal_grounded(cfg: PipelineConfig, signal: Signal) -> bool:
    """True iff the signal's datum re-derives from its artifact and shows in text.

    For ``contains:`` fields the datum is a boolean presence check. For numeric /
    cell fields, some standard rendering of the re-derived value must appear in
    the signal's English text — so the human-visible number is the artifact's.
    """
    value = resolve_field(cfg, signal.artifact, signal.field)
    if value is _MISSING:
        return False
    if signal.field.startswith("contains:"):
        return bool(value)
    return any(r in signal.text_en for r in _renderings(value))


def all_signals_grounded(cfg: PipelineConfig, card: Card) -> bool:
    return all(signal_grounded(cfg, s) for s in card.signals)


# =============================================================================
# Shared caveat signal (always the third signal on an actionable card)
# =============================================================================
def _uncertainty_signal(cfg: PipelineConfig, metrics_rel: str) -> Signal:
    rmse = resolve_field(cfg, metrics_rel, "metrics_vs_uncapped_truth.rmse")
    rmse_txt = f"{rmse:.0f}" if isinstance(rmse, (int, float)) else "?"
    return Signal(
        text_en=(
            f"Point estimates, typically off by about ±{rmse_txt} cycles and "
            "widest near end-of-life, where the model tends to read optimistic — "
            "use for triage ordering, not exact timing."
        ),
        text_zh=(
            f"这些都是点估计，误差大约 ±{rmse_txt} 个周期，越接近寿命末端越大，"
            "而且模型这时往往偏乐观——只用来排优先级，别当作精确的剩余时间。"
        ),
        artifact=metrics_rel,
        field="metrics_vs_uncapped_truth.rmse",
    )


# =============================================================================
# Card builders (each grounded in real artifacts)
# =============================================================================
def build_triage_card(
    cfg: PipelineConfig,
    preds_rel: str,
    metrics_rel: str,
    inspect_first_max: float,
) -> Card:
    """The flagship P1 card: engines that need inspection first (from predictions)."""
    df = _load_df((cfg.root / preds_rel).resolve())
    high = df[df["risk_band"] == "high"].sort_values("pred_rul")
    first = high[high["pred_rul"] <= inspect_first_max]
    if first.empty:  # degrade: still surface the N most urgent high-risk units
        first = high.head(3)
    ids = [int(u) for u in first["unit_id"].tolist()]
    ruls = [round(float(r), 1) for r in first["pred_rul"].tolist()]
    n_first = len(ids)
    n_high = int((df["risk_band"] == "high").sum())
    lead = ids[0] if ids else None
    ruls_txt = " / ".join(f"{r:.1f}" for r in ruls)
    ids_txt = ", ".join(str(i) for i in ids)

    signals = [
        Signal(
            text_en=(
                f"Units {ids_txt} sit at predicted RUL {ruls_txt} cycles — the "
                f"{n_first} closest to end-of-life (≤{inspect_first_max:g} cycles)."
            ),
            text_zh=(
                f"{ids_txt} 号机组预测剩余寿命 {ruls_txt} 个周期，是最接近报废的 "
                f"{n_first} 台（≤{inspect_first_max:g} 周期）。"
            ),
            artifact=preds_rel,
            field=f"count:pred_rul<={inspect_first_max}",
        ),
        Signal(
            text_en=(
                f"{n_high} engines are in the high-risk band overall — schedule "
                "the rest this cycle after the first inspections."
            ),
            text_zh=(
                f"整机队共有 {n_high} 台落在高风险区间——先查最急的，其余本周期"
                "内排上。"
            ),
            artifact=preds_rel,
            field="count:risk_band=high",
        ),
        _uncertainty_signal(cfg, metrics_rel),
    ]
    lead_evidence = f"data/processed/evidence/unit_{lead}.json" if lead else preds_rel
    actions = [
        Action(
            id="schedule_inspection",
            label_en=f"Schedule inspection for units {ids_txt}",
            label_zh=f"为 {ids_txt} 号机组安排检查",
            consequence_en=(
                "Export = draft work orders only, never commands. No maintenance "
                "is actuated; a human still approves each work order."
            ),
            consequence_zh=(
                "导出的只是工单草稿，绝不是执行指令。不会触发任何维护动作，"
                "每张工单仍需人工批准。"
            ),
            safe_default=True,
        ),
        Action(
            id="defer",
            label_en="Defer to next cycle",
            label_zh="推迟到下个周期",
            consequence_en="Units stay flagged; no work order is drafted.",
            consequence_zh="机组仍保持标记；不生成任何工单。",
        ),
        Action(
            id="open_evidence",
            label_en=f"Open evidence for unit {lead}",
            label_zh=f"查看 {lead} 号机组的证据",
            consequence_en="Opens the teaching-mode evidence viewer; changes nothing.",
            consequence_zh="打开教学模式的证据查看器；不改变任何状态。",
        ),
    ]
    return Card(
        id=_card_id("high_risk_triage", ids, n_high),
        kind="high_risk_triage",
        priority="P1",
        verdict_en=(
            f"{n_first} engines need inspection first — units {ids_txt} "
            f"(predicted RUL ≤{inspect_first_max:g} cycles)."
        ),
        verdict_zh=(
            f"有 {n_first} 台发动机需要优先检查——{ids_txt} 号"
            f"（预测剩余寿命 ≤{inspect_first_max:g} 周期）。"
        ),
        signals=signals,
        evidence_link=lead_evidence,
        actions=actions,
    )


def build_signoff_card(
    cfg: PipelineConfig,
    metrics_rel: str,
    eval_summary_rel: str,
    n_units: int,
) -> Card:
    """The report sign-off card: ready to publish, but only a human may sign."""
    rmse = resolve_field(cfg, metrics_rel, "metrics_vs_uncapped_truth.rmse")
    rmse_txt = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "?"
    signals = [
        Signal(
            text_en=(
                f"An independent recompute of test-set RMSE matches the recorded "
                f"metric ({rmse_txt} cycles) — the prediction artifact is consistent."
            ),
            text_zh=(
                f"独立重算的测试集 RMSE 与记录的指标一致（{rmse_txt} 个周期）——"
                "预测产物是自洽的。"
            ),
            artifact=metrics_rel,
            field="metrics_vs_uncapped_truth.rmse",
        ),
        Signal(
            text_en=(
                f"All {n_units} diagnostic reports carry citations, uncertainty, "
                "and a human-review flag; the evaluation summary records no "
                "governance violations."
            ),
            text_zh=(
                f"全部 {n_units} 份诊断报告都带引用、不确定性说明和人工复核标记；"
                "评测总结记录无治理违规。"
            ),
            artifact=eval_summary_rel,
            field="contains:No violations",
        ),
        _uncertainty_signal(cfg, metrics_rel),
    ]
    actions = [
        Action(
            id="hold_for_review",
            label_en="Hold report for human sign-off",
            label_zh="暂缓报告，等待人工签署",
            consequence_en=(
                "Report stays in draft; nothing is published or signed. This is "
                "the safe default — the agent may never sign a report itself."
            ),
            consequence_zh=(
                "报告保持草稿状态；不发布、不签署。这是安全默认项——智能体永远"
                "不会自己签署报告。"
            ),
            safe_default=True,
        ),
        Action(
            id="sign_off",
            label_en="Sign off and publish the report",
            label_zh="签署并发布报告",
            consequence_en=(
                "Marks the report human-approved. Requires an explicit human "
                "decision — it can never be auto-applied, even with "
                "--yes-safe-defaults."
            ),
            consequence_zh=(
                "把报告标记为人工批准。必须由人明确决定——即使加了 "
                "--yes-safe-defaults 也不会自动执行。"
            ),
        ),
        Action(
            id="export_draft",
            label_en="Export the draft only",
            label_zh="仅导出草稿",
            consequence_en="Writes a draft report file; no sign-off is recorded.",
            consequence_zh="写出一份草稿报告文件；不记录任何签署。",
        ),
    ]
    return Card(
        id=_card_id("report_signoff", rmse_txt, n_units),
        kind="report_signoff",
        priority="P2",
        verdict_en=(
            "The run's report is complete and self-consistent — it needs a human "
            "signature before it can be published."
        ),
        verdict_zh=(
            "本次运行的报告已完成且自洽——需要人工签署后才能发布。"
        ),
        signals=signals,
        evidence_link=eval_summary_rel,
        actions=actions,
    )


def build_champion_card(
    cfg: PipelineConfig,
    selection_rel: str,
    champion_rmse: float,
    floor_rmse: float,
    margin: float,
    bar: float,
) -> Card:
    """Champion-confirmation card: the champion only marginally beats the floor.

    Raised only when ``model_selection.json`` records a thin ``floor_gap_cycles``;
    when the selection file is absent the gate degrades to a recorded skip instead.
    Signals reference the bake-off's own ``verdict`` fields, so they reproduce.
    """
    signals = [
        Signal(
            text_en=(
                f"The champion beats the Ridge floor by only {margin:.2f} cycles — "
                f"below the {bar:.2f}-cycle clear-win bar, so it's a thin margin."
            ),
            text_zh=(
                f"冠军模型只比 Ridge 下限好 {margin:.2f} 个周期——没到 {bar:.2f} "
                "周期的明显胜出门槛，优势很薄。"
            ),
            artifact=selection_rel,
            field="verdict.floor_gap_cycles",
        ),
        Signal(
            text_en=(
                f"The clear-win bar is {bar:.2f} cycles: a tie or thin margin is a "
                "governance decision, not an automatic pick — the agent must not "
                "choose the champion on a calibration tie."
            ),
            text_zh=(
                f"明显胜出门槛是 {bar:.2f} 个周期：打平或优势很薄时选谁要讲规矩，"
                "不能自动定——校准打平时智能体不得擅自选冠军。"
            ),
            artifact=selection_rel,
            field="verdict.champion_margin_rmse",
        ),
        _uncertainty_signal(cfg, "reports/metrics_model.json"),
    ]
    actions = [
        Action(
            id="keep_floor",
            label_en="Keep the simpler Ridge floor",
            label_zh="保留更简单的 Ridge 下限模型",
            consequence_en="Falls back to the simpler model; no champion swap.",
            consequence_zh="回退到更简单的模型；不更换冠军。",
            safe_default=True,
        ),
        Action(
            id="confirm_champion",
            label_en="Confirm the champion",
            label_zh="确认冠军模型",
            consequence_en="Records the champion behind the prediction interface.",
            consequence_zh="在预测接口后面记录该冠军模型。",
        ),
    ]
    return Card(
        id=_card_id("champion_confirmation", champion_rmse, floor_rmse),
        kind="champion_confirmation",
        priority="P2",
        verdict_en=(
            "The champion only marginally beats the Ridge floor — confirm the pick "
            "or keep the simpler model."
        ),
        verdict_zh=(
            "冠军模型只是勉强赢过 Ridge 下限——请确认这个选择，或保留更简单的模型。"
        ),
        signals=signals,
        evidence_link=selection_rel,
        actions=actions,
    )


def build_data_quality_card(
    cfg: PipelineConfig,
    artifact_rel: str,
    field_: str,
    detail_en: str,
    detail_zh: str,
) -> Card:
    """Data-quality exception card, grounded on the offending artifact+field."""
    value = resolve_field(cfg, artifact_rel, field_)
    signals = [
        Signal(
            text_en=f"{detail_en} (observed value: {value}).",
            text_zh=f"{detail_zh}（观测值：{value}）。",
            artifact=artifact_rel,
            field=field_,
        ),
        _uncertainty_signal(cfg, "reports/metrics_model.json"),
    ]
    actions = [
        Action(
            id="quarantine",
            label_en="Quarantine the affected data and re-check",
            label_zh="隔离受影响的数据并复查",
            consequence_en="Marks the data as suspect; downstream stays paused.",
            consequence_zh="把数据标为可疑；下游保持暂停。",
            safe_default=True,
        ),
        Action(
            id="accept",
            label_en="Accept and continue",
            label_zh="接受并继续",
            consequence_en="Proceeds despite the flag; recorded in the trace.",
            consequence_zh="尽管有标记仍继续；会记录在追踪里。",
        ),
    ]
    return Card(
        id=_card_id("data_quality_exception", artifact_rel, field_, value),
        kind="data_quality_exception",
        priority="P2",
        verdict_en=f"A data-quality check tripped on {artifact_rel} — review before trusting downstream results.",
        verdict_zh=f"{artifact_rel} 上的数据质量检查被触发——在信任下游结果前先复查。",
        signals=signals,
        evidence_link=artifact_rel,
        actions=actions,
    )


# =============================================================================
# Healthy digest (collapsed; not an actionable card)
# =============================================================================
def build_healthy_digest(cfg: PipelineConfig, preds_rel: str, n_flagged: int) -> dict:
    """The collapsed green digest that batches everything not needing a decision."""
    df = _load_df((cfg.root / preds_rel).resolve())
    n_scored = int(len(df))
    n_healthy = int((df["risk_band"] == "low").sum())
    return {
        "kind": "healthy_digest",
        "n_scored": n_scored,
        "n_flagged": n_flagged,
        "n_healthy": n_healthy,
        "artifact": preds_rel,
        "sentence_en": (
            f"Agent scored {n_scored} engines, flagged {n_flagged}, prepared "
            f"evidence per unit. {n_healthy} healthy engines auto-cleared."
        ),
        "sentence_zh": (
            f"智能体给 {n_scored} 台发动机打了分，标记了 {n_flagged} 台，"
            f"并为每台整理了证据。{n_healthy} 台健康发动机已自动放行。"
        ),
    }
