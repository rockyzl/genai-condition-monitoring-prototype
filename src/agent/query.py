"""Query orchestrator — the ``ask`` entry mode over the shared registry.

A query is planned (:mod:`src.agent.planner`), executed tool-by-tool through the
:class:`src.agent.registry.Registry`, and answered. The hard rule the plan calls
non-negotiable is enforced structurally here: **the answer is composed only from
tool outputs**, and every factual claim is bound to the trace seq + field it came
from, so :meth:`src.agent.trace.Trace.all_claims_grounded` can verify each one.

Two canonical intents are supported end-to-end:

* ``"diagnose unit 81"`` → a grounded, KB-cited diagnosis of that unit.
* ``"which engines need inspection?"`` → the high-risk list, with each flagged
  unit's own predicted RUL fanned out via ``get_prediction`` for per-unit grounding.

Anything the planner cannot map yields an honest "couldn't map that" answer — the
orchestrator never invents tool calls or numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.agent.planner import ToolCall, make_planner
from src.agent.registry import Registry, ToolResult
from src.agent.trace import Trace
from src.pipeline.config import PipelineConfig

#: How many of the lowest-RUL high-risk units to fan out get_prediction for.
FANOUT_LIMIT = 5
#: Predicted-RUL threshold (cycles) for the "inspect first" set in the answer.
INSPECT_FIRST_MAX = 10.0


@dataclass
class Step:
    call: ToolCall
    result: ToolResult
    seq: int
    fanned: list[tuple]  # (unit_id, tool, ToolResult, seq)


def answer_query(
    cfg: PipelineConfig,
    query: str,
    planner_kind: str = "rule",
    out_dir: Path | str | None = None,
) -> dict:
    """Plan → execute → compose a grounded answer; write and return the trace."""
    reg = Registry(cfg)
    planner = make_planner(planner_kind)
    plan = planner.plan(query)  # may raise PlannerNotConfigured for the llm stub

    run_id = "ask_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trace = Trace(run_id=run_id, kind="query", planner=planner_kind)

    steps: list[Step] = []
    for call in plan:
        res = reg.call(call.tool, call.args)
        seq = trace.record_tool(res)
        fanned: list[tuple] = []
        if call.fan_out and res.ok and isinstance(res.output, dict):
            for u in res.output.get("units", [])[:FANOUT_LIMIT]:
                for ft in call.fan_out:
                    fr = reg.call(ft, {"unit": u["unit_id"]})
                    fseq = trace.record_tool(fr)
                    fanned.append((u["unit_id"], ft, fr, fseq))
        steps.append(Step(call, res, seq, fanned))

    answer = _compose(cfg, query, steps, trace)
    trace.set_answer(answer)
    path = trace.write(out_dir or cfg.path("reports"))
    return {
        "answer": answer,
        "trace": trace,
        "trace_path": str(path),
        "grounded": trace.all_claims_grounded(),
    }


def _find(steps: list[Step], tool: str) -> Step | None:
    for s in steps:
        if s.call.tool == tool and s.result.ok:
            return s
    return None


def _compose(cfg: PipelineConfig, query: str, steps: list[Step], trace: Trace) -> dict:
    if not steps:
        return _unmapped(query)

    diag = _find(steps, "diagnose")
    if diag is not None:
        return _compose_diagnosis(query, steps, diag, trace)

    lst = _find(steps, "list_units_by_risk")
    if lst is not None:
        return _compose_inspection(query, steps, lst, trace)

    rep = _find(steps, "report")
    if rep is not None:
        return _compose_report(query, rep, trace)

    # A plan ran but produced only errors — report them, invent nothing.
    errs = [f"{s.call.tool}: {s.result.error}" for s in steps if not s.result.ok]
    return {
        "question": query,
        "intent": "error",
        "answer_en": "The tools needed to answer this did not return results: "
        + "; ".join(errs),
        "answer_zh": "回答这个问题所需的工具没有返回结果：" + "；".join(errs),
        "claims": [],
        "citations": [],
    }


def _unmapped(query: str) -> dict:
    return {
        "question": query,
        "intent": "unmapped",
        "answer_en": (
            "I could not map that to a supported tool. Try 'diagnose unit <N>' or "
            "'which engines need inspection?'."
        ),
        "answer_zh": (
            "我没法把这句话对应到支持的工具。可以试试 “diagnose unit <N>” 或 "
            "“which engines need inspection?”。"
        ),
        "claims": [],
        "citations": [],
    }


def _compose_inspection(query, steps, lst: Step, trace: Trace) -> dict:
    out = lst.result.output
    n_high = out["n"]
    band = out["band"]
    trace.add_claim(f"{n_high} engines in the {band}-risk band", lst.seq,
                    "list_units_by_risk", "n", n_high)

    # Per-unit predictions from the fan-out (grounded unit-by-unit).
    pred_by_unit: dict[int, tuple[int, float]] = {}
    for unit_id, tool, res, seq in lst.fanned:
        if tool == "get_prediction" and res.ok:
            pred_by_unit[unit_id] = (seq, res.output["pred_rul"])

    inspect_first = []
    for u in out["units"][:FANOUT_LIMIT]:
        uid = u["unit_id"]
        if uid in pred_by_unit:
            seq, pred = pred_by_unit[uid]
            if pred <= INSPECT_FIRST_MAX:
                inspect_first.append((uid, pred, seq))
                trace.add_claim(f"unit {uid} predicted RUL {pred}", seq,
                                "get_prediction", "pred_rul", pred)

    if inspect_first:
        listed = ", ".join(f"unit {uid} ({pred:g} cycles)" for uid, pred, _ in inspect_first)
        answer_en = (
            f"{n_high} engines are in the high-risk band. Inspect first — the "
            f"{len(inspect_first)} within {INSPECT_FIRST_MAX:g} predicted cycles of "
            f"end-of-life: {listed}. Then schedule the remaining high-risk units."
        )
        listed_zh = "、".join(f"{uid} 号（{pred:g} 周期）" for uid, pred, _ in inspect_first)
        answer_zh = (
            f"共有 {n_high} 台发动机处于高风险区间。优先检查——预测剩余寿命在 "
            f"{INSPECT_FIRST_MAX:g} 周期以内的 {len(inspect_first)} 台：{listed_zh}。"
            "随后再安排其余高风险机组。"
        )
    else:
        answer_en = (
            f"{n_high} engines are in the high-risk band; none are within "
            f"{INSPECT_FIRST_MAX:g} predicted cycles, so schedule them this cycle."
        )
        answer_zh = (
            f"共有 {n_high} 台发动机处于高风险区间；没有一台在 "
            f"{INSPECT_FIRST_MAX:g} 周期以内，本周期内安排检查即可。"
        )

    return {
        "question": query,
        "intent": "inspection",
        "answer_en": answer_en,
        "answer_zh": answer_zh,
        "claims": [c.__dict__ for c in trace.claims],
        "citations": [{"artifact": out.get("artifact"), "note": "predicted RUL per unit"}],
    }


def _compose_diagnosis(query, steps, diag: Step, trace: Trace) -> dict:
    report = diag.result.output
    ev_step = _find(steps, "get_evidence")

    pred_rul = None
    risk = None
    if ev_step is not None:
        ev = ev_step.result.output
        pred_rul = ev.get("predicted_rul")
        risk = ev.get("risk_band")
        trace.add_claim(f"predicted RUL {pred_rul}", ev_step.seq,
                        "get_evidence", "predicted_rul", pred_rul)
        trace.add_claim(f"risk band {risk}", ev_step.seq,
                        "get_evidence", "risk_band", risk)

    fms = report.get("possible_failure_modes", [])
    steps_list = report.get("recommended_next_steps", [])
    top_fm = fms[0] if fms else None
    top_step = steps_list[0] if steps_list else None
    if top_fm:
        trace.add_claim(
            f"possible failure mode: {top_fm.get('failure_mode')}", diag.seq,
            "diagnose", "possible_failure_modes.0.failure_mode",
            top_fm.get("failure_mode"),
        )
    if top_step:
        trace.add_claim(
            f"recommended next step: {top_step.get('step')}", diag.seq,
            "diagnose", "recommended_next_steps.0.step", top_step.get("step"),
        )

    asset = report.get("asset_id")
    parts = [report.get("summary", "")]
    if top_fm and top_fm.get("source_file"):
        parts.append(
            f"Most-likely failure mode from the knowledge base: "
            f"{top_fm['failure_mode']} ({top_fm['source_file']} › {top_fm['section']})."
        )
    if top_step and top_step.get("source_file"):
        parts.append(
            f"Recommended next step: {top_step['step']} "
            f"({top_step['source_file']} › {top_step['section']})."
        )
    parts.append(report.get("uncertainty", ""))
    answer_en = " ".join(p for p in parts if p)
    _band_zh = {"high": "高", "medium": "中", "low": "低"}.get(str(risk), str(risk))
    fm_zh = (
        f"最可能的失效模式（来自知识库）：{top_fm['failure_mode']}"
        f"（出处 {top_fm['source_file']} › {top_fm['section']}）。"
        if top_fm and top_fm.get("source_file") else ""
    )
    answer_zh = (
        f"关于 {asset} 号机组：预测剩余寿命约 {pred_rul} 周期，风险等级{_band_zh}。"
        f"{fm_zh}以下结论均来自知识库检索并附出处，必须由人工复核（见 citations）。"
    )

    return {
        "question": query,
        "intent": "diagnosis",
        "asset_id": asset,
        "answer_en": answer_en,
        "answer_zh": answer_zh,
        "claims": [c.__dict__ for c in trace.claims],
        "citations": report.get("citations", []),
        "human_review_required": report.get("human_review_required", True),
    }


def _compose_report(query, rep: Step, trace: Trace) -> dict:
    out = rep.result.output
    pred = out.get("prediction", {})
    trace.add_claim(f"predicted RUL {pred.get('pred_rul')}", rep.seq,
                    "report", "prediction.pred_rul", pred.get("pred_rul"))
    diag = out.get("diagnosis", {})
    return {
        "question": query,
        "intent": "report",
        "asset_id": out.get("unit_id"),
        "answer_en": diag.get("summary", ""),
        "answer_zh": "该机组的完整报告（预测＋证据＋诊断）已生成，结论均可溯源，"
        "需人工复核。",
        "claims": [c.__dict__ for c in trace.claims],
        "citations": diag.get("citations", []),
        "human_review_required": diag.get("human_review_required", True),
    }
