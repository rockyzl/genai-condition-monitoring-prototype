"""Streamlit demo — v2: Autopilot page + Decision Inbox + Teaching mode.

A top-level bilingual (English default / 中文) app with two modes:

* **🤖 Autopilot / 自动巡检** (landing default) — the flagship. It launches the
  deterministic pipeline agent as a subprocess (never in-process), tails its
  append-only journal to render a live step-by-step timeline, and condenses the
  run into a **Decision Inbox**: grounded, cited decision cards with safe-default
  actions, a fleet triage table, and the EDA story charts.
* **🎓 Teaching mode / 教学模式** — the existing button-driven wizard, reused as
  the per-unit evidence viewer that decision cards deep-link into.

Single source of truth: the RUL cap, the typical-miss (±cycles), and the risk
thresholds are read from ``reports/metrics_model.json`` + ``config/pipeline.yaml``
(via :class:`PipelineConfig`) — nothing about the model is hard-coded here.

Naming discipline (enforced by the plan): this is a *deterministic pipeline agent
with human-in-the-loop decision gates* — never "autonomous".

Test seams (both off in normal use):
* ``CM_APP_REPORTS_DIR`` — override the dir the app reads the journal/inbox from
  and writes answered cards to (lets AppTest point at a tmp fixture dir).
* ``CM_APP_NO_SUBPROCESS`` — skip actually launching the agent subprocess.

Run:  .venv/bin/streamlit run src/app/streamlit_app.py

The UI body is guarded under ``if __name__ == "__main__"`` so the module imports
cleanly (no Streamlit side effects) for testing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- make `src` importable when run via `streamlit run` -------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import build_evidence as be  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.journal import read_events  # noqa: E402
from src.pipeline.specs import STAGE_ORDER, STAGE_SPECS  # noqa: E402
from src.rag.assistant import diagnose  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402

# Top-level mode labels (bilingual, fixed — this is the mode switch itself).
MODE_CHAT = "💬 Agent Chat / 对话巡检"
MODE_DASH = "📊 Autopilot dashboard / 巡检看板"
MODE_TEACH = "🎓 Teaching mode / 教学模式"

# Chip label -> canonical planner-friendly intent text.
CHIPS = [
    ("chip_fleet", "run the whole fleet"),
    ("chip_diag", "diagnose unit 81"),
    ("chip_how", "how it works"),
]

_HOW_RE = re.compile(
    r"(how (do|does) (it|you|this) work|how it works|what (do|can) you do|"
    r"怎么工作|怎么运作|如何工作|你能做|能做什么|工作原理)",
    re.IGNORECASE,
)

# Free-text that means "run the full pipeline" (vs. the read-only query intents).
_RUN_RE = re.compile(
    r"(run\b|autopilot|whole fleet|entire fleet|full (analysis|run|pipeline|sweep)|"
    r"fleet (run|scan|sweep|inspection|analysis)|巡检|整个机队|全机队|"
    r"跑(一遍|全程|管线|流程|完|一下))",
    re.IGNORECASE,
)

_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫"
BAND_HEX = {"high": "#c0392b", "medium": "#e67e22", "low": "#27ae60"}

# Physical meaning of C-MAPSS sensor columns: (symbol, English, 中文). Source:
# Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
# Simulation," PHM08 (Table 1) — the paper cited in readme.txt. The dataset
# readme itself labels columns only as "sensor measurement 1-21"; these physical
# names are the community-standard reading of that reference, not in the readme.
SENSOR_META = {
    "sensor_1": ("T2", "Fan inlet temperature", "风扇进口温度"),
    "sensor_2": ("T24", "LPC outlet temperature", "低压压气机出口温度"),
    "sensor_3": ("T30", "HPC outlet temperature", "高压压气机出口温度"),
    "sensor_4": ("T50", "LPT outlet temperature", "低压涡轮出口温度"),
    "sensor_5": ("P2", "Fan inlet pressure", "风扇进口压力"),
    "sensor_6": ("P15", "Bypass-duct pressure", "外涵道压力"),
    "sensor_7": ("P30", "HPC outlet pressure", "高压压气机出口压力"),
    "sensor_8": ("Nf", "Physical fan speed", "风扇转速"),
    "sensor_9": ("Nc", "Physical core speed", "核心机转速"),
    "sensor_10": ("epr", "Engine pressure ratio", "发动机压比"),
    "sensor_11": ("Ps30", "HPC outlet static pressure", "高压压气机出口静压"),
    "sensor_12": ("phi", "Fuel-flow to Ps30 ratio", "燃油流量与静压之比"),
    "sensor_13": ("NRf", "Corrected fan speed", "换算风扇转速"),
    "sensor_14": ("NRc", "Corrected core speed", "换算核心机转速"),
    "sensor_15": ("BPR", "Bypass ratio", "涵道比"),
    "sensor_16": ("farB", "Burner fuel-air ratio", "燃烧室油气比"),
    "sensor_17": ("htBleed", "Bleed enthalpy", "引气焓值"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed", "指令风扇转速"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed", "指令换算风扇转速"),
    "sensor_20": ("W31", "HPT coolant bleed", "高压涡轮冷却引气"),
    "sensor_21": ("W32", "LPT coolant bleed", "低压涡轮冷却引气"),
}

_TREND_ARROW = {"increasing": "↑", "decreasing": "↓", "flat": "→"}
_TREND_ZH = {"increasing": "在上升", "decreasing": "在下降", "flat": "基本平稳"}
_BAND_ZH = {"high": "高风险", "medium": "中风险", "low": "低风险"}
RISK_WORD = {
    "English": {"high": "High", "medium": "Medium", "low": "Low"},
    "中文": {"high": "高风险", "medium": "中风险", "low": "低风险"},
}

# --- Translation dict: every piece of UI chrome, keyed by language ---------
T = {
    "English": {
        "app_title": "Condition Monitoring — Autopilot + Evidence Viewer",
        "limitations": (
            "Independent R&D prototype on public NASA C-MAPSS turbofan data. Not "
            "production-validated, not affiliated with any equipment "
            "manufacturer, and not a source of safety-critical decisions. Every "
            "output is advisory and requires review by a qualified human."
        ),
        "mode_label": "Mode",
        # ---------- Agent Chat ----------
        "chat_badge": "Deterministic · Grounded · No LLM",
        "chat_intro": (
            "This agent runs the condition-monitoring pipeline and answers fleet "
            "questions. Try: scan the fleet, diagnose one engine, or how it works."
        ),
        "greeting": (
            "Hi — I can scan the fleet, diagnose one engine, or explain how I "
            "work. What do you need?"
        ),
        "showcase": (
            "Example — ran list_units_by_risk: **{n} engines high-risk**; inspect "
            "first units **{ids}** (predicted RUL ≤{cap} cycles)."
        ),
        "showcase_src": "test_predictions.csv · count:risk_band=high",
        "chip_fleet": "🛰 Scan fleet",
        "chip_diag": "🩺 Diagnose one",
        "chip_how": "⚙️ How it works?",
        "chat_input_ph": "Type a request…  (e.g. diagnose unit 81)",
        "chat_reset": "🗑 Reset chat",
        "howitworks": (
            "Runs a fixed **10-stage** pipeline (load → clean → features → score "
            "→ select → predict → evidence → diagnose → eval), deterministic and "
            "seed-fixed. Every number traces to a named artifact; every "
            "recommendation cites a knowledge-base section. No LLM in the loop."
        ),
        "plan_preview": (
            "Plan: load 100 engines → clean sensors → score life → rank by risk → "
            "prep evidence for the ones that need you. ~10s, read-only, no work "
            "orders sent. Run it?"
        ),
        "chat_btn_start": "▶ Start",
        "chat_btn_dry": "Adjust scope → dry-run",
        "chat_starting": "Started the full analysis (gated, read-only).",
        "chat_dry_starting": "Started a dry-run — reports what would happen, raises no decisions.",
        "run_progress_running": "Running the pipeline…  current: {stage}",
        "run_cache_reuse": "{n} steps reused cache",
        "run_done_expander": "✓ Done · view steps",
        "chat_fallback": (
            "Out of scope for this agent. It can: **scan the fleet**, **diagnose "
            "an engine** (e.g. 'diagnose unit 81'), or **explain how it works**."
        ),
        "chat_tools_label": "🔧 tools",
        "chat_grounded": "grounded — trace",
        "chat_citations": "Citations",
        "decision_pending": "{n} decision(s) pending ↓",
        "why_expander": "why?",
        "chat_resolved": "✓ Confirmed · {act}",
        "skip_results": "⏭ Skip to results",
        "fail_step": "Step {stage} failed. Nothing was changed.",
        "fail_retry": "Retry",
        "fail_details": "View details",
        "queued_run": "Queued — will run after the current pass.",
        "chat_done_note": "Every card's signals are grounded in named artifacts; nothing is invented.",
        # ---------- Autopilot page ----------
        "ap_intro": (
            "The agent does the heavy lifting — ingest → model → predict → "
            "summarize — and hands you a short Decision Inbox with the strongest "
            "signals extracted. You make the calls."
        ),
        "ap_naming": (
            "Deterministic pipeline agent with human-in-the-loop decision gates. "
            "It automates the analysis; you own the decision. It is not autonomous "
            "and never closes a safety-critical loop."
        ),
        "ap_run_header": "1 · Run the agent",
        "btn_run_gated": "▶ Run autopilot (gated)",
        "btn_run_dry": "Dry-run (raise nothing)",
        "ap_running": "Agent running… watching the pipeline live (polling ~1s).",
        "ap_watch_header": "2 · Watch it work",
        "ap_no_run": "No run yet — press **Run autopilot** to start the pipeline.",
        "ap_run_line": "Run {rid} — walking {n} stages",
        "ap_skipped": "already current — skipped",
        "ap_ran": "ran",
        "ap_details": "details",
        "ap_why": "Why",
        "ap_rundone": "Done: {ran} stage(s) ran, {skipped} skipped, {sec:.2f}s.",
        "ap_halt": "HALTED",
        "inbox_header": "3 · Decision Inbox",
        "inbox_empty": (
            "No pending decisions right now. Run the agent (gated) to generate the "
            "inbox — the flagship triage card appears here."
        ),
        "done_banner": (
            "Agent scored {n} engines, flagged {h} high-risk, prepared evidence "
            "per unit. **You have {d} decision(s).**"
        ),
        "digest_header": "🟢 Healthy engines (auto-cleared)",
        "digest_line": (
            "{low} engines are low-risk and were auto-cleared — no action needed."
        ),
        "handled_header": "✅ Handled",
        "handled_empty": "Nothing handled yet.",
        "handled_line": "Card `{cid}` → action **{act}** ({actor})",
        "card_signals_h": "Why",
        "card_actions_h": "Your options",
        "card_grounded": "grounded in",
        "card_evidence_btn": "🔍 Open evidence for unit {u} (Teaching mode)",
        "card_evidence_link": "Evidence:",
        "action_done": "Recorded your decision — resuming the agent to continue the walk.",
        # ---------- Fleet + EDA ----------
        "fleet_header": "4 · Fleet view — all engines",
        "fleet_caption": (
            "Every test engine with its predicted remaining life and risk band, "
            "most urgent first. Click a column header to re-sort."
        ),
        "fleet_c_unit": "Engine",
        "fleet_c_last": "Cycles flown",
        "fleet_c_pred": "Predicted RUL",
        "fleet_c_band": "Risk",
        "eda_header": "📊 Explore the data / 看看数据",
        "eda_cap_mono": (
            "Which sensors track wear: each sensor's correlation with remaining "
            "life. The strong ones (Ps30, T50, phi…) are the degradation carriers "
            "the model leans on."
        ),
        "eda_cap_flat": (
            "The flat sensors we drop: several channels barely move on FD001, so "
            "they carry no wear signal and are excluded before modelling."
        ),
        "eda_cap_life": (
            "How long engines run before failure (median ~199 cycles). Because "
            "~39% of training rows sit above {cap} cycles of healthy life, the RUL "
            "target is capped at {cap}."
        ),
        "eda_cap_predvtrue": (
            "Predicted vs true remaining life on the test set — points on the "
            "diagonal are perfect; the model is tighter mid-life and looser near "
            "failure."
        ),
        "eda_cap_errhist": (
            "Distribution of prediction errors — centred near zero, with a tail "
            "where the model reads too healthy near end-of-life."
        ),
        "eda_cap_degrad": (
            "Example degradation trajectories: a few engines' key sensor drifting "
            "as they approach failure."
        ),
        # ---------- Teaching mode (the wizard) ----------
        "title": "Predicting When an Engine Needs Maintenance",
        "what_is_this": (
            "A 5-step guided tour: press each button to run the next stage of the "
            "pipeline. It reads an aircraft engine's sensors, estimates how much "
            "longer it can safely run, and writes a maintenance note in which "
            "every cause and next step is quoted from a cited reference — it never "
            "invents a diagnosis."
        ),
        "lbl_function": "Function:",
        "lbl_purpose": "Purpose:",
        "sb_controls": "Controls",
        "sb_cycles_label": "How many recent cycles to plot",
        "sb_kb_empty": "Knowledge base is empty — diagnostic guidance unavailable.",
        "step1_title": "① Step 1 — Pick an engine",
        "step1_desc": (
            "Sensors — we take the engine's recent readings (temperatures, "
            "pressures, speeds) from its last flight cycles."
        ),
        "step1_func": "Pick one engine from the NASA test fleet and load its raw sensor history.",
        "step1_purpose": (
            "Everything downstream is about this one engine — you choose which "
            "asset to inspect."
        ),
        "step1_engine_label": "Engine (test unit) id",
        "step2_title": "② Step 2 — Read the conclusion",
        "step2_desc": (
            "Prediction — a simple machine-learning model estimates the engine's "
            "Remaining Useful Life: how many more flight cycles it can run before "
            "maintenance-critical wear."
        ),
        "step2_func": (
            "The model reads this engine's sensor history and estimates how many "
            "flight cycles remain."
        ),
        "step2_purpose": "Turns 21 sensor streams into one number a maintenance planner can act on.",
        "step3_title": "③ Step 3 — See the evidence",
        "step3_desc": (
            "Sensors — we take the engine's recent readings (temperatures, "
            "pressures, speeds) from its last flight cycles."
        ),
        "step3_func": (
            "Plot the most informative sensors over recent cycles and label what "
            "each one physically measures."
        ),
        "step3_purpose": (
            "Lets a human check the prediction against the raw signals — is the "
            "engine really degrading, or is it sensor noise?"
        ),
        "step4_title": "④ Step 4 — Read the guidance",
        "step4_desc_retrieval": (
            "Retrieved guidance — the system searches a small library of "
            "maintenance and engineering notes for passages relevant to this "
            "engine's condition."
        ),
        "step4_desc_report": (
            "Cited report — it writes a short diagnosis, and every possible cause "
            "or next step it lists is quoted from those notes with a citation. If "
            "it finds nothing relevant, it says so rather than guessing."
        ),
        "step4_func": (
            "Runs TF-IDF retrieval over the knowledge base, then composes a cited "
            "report by template (no LLM guesswork)."
        ),
        "step4_purpose": "Every recommendation is cited and traceable instead of invented.",
        "step5_title": "⑤ Step 5 — Remember the limits",
        "step5_desc": "A qualified human always makes the final call.",
        "step5_func": "Surfaces the model's uncertainty and forces human review before any action.",
        "step5_purpose": (
            "This is decision support, not a decision — it focuses attention, it "
            "never closes a safety-critical loop."
        ),
        "btn_load": "① Load sensor history",
        "btn_predict": "② Predict remaining life",
        "btn_evidence": "③ Show the evidence",
        "btn_report": "④ Generate cited diagnostic report",
        "btn_runall": "▶ Run all steps",
        "concl_high": (
            "Engine #{u} has about {r} flight cycles left — high risk. Schedule "
            "an inspection soon."
        ),
        "concl_medium": (
            "Engine #{u} has about {r} flight cycles left — medium risk. It's "
            "degrading; monitor it closely."
        ),
        "concl_low": (
            "Engine #{u} has about {r} flight cycles left — low risk. It looks "
            "healthy; keep monitoring."
        ),
        "health_text": "Estimated remaining life: {r} of {cap} cycles (fuller = healthier)",
        "health_caption": (
            "The bar shows predicted remaining life against the model's "
            "{cap}-cycle ceiling. An almost-empty bar means the engine is near "
            "the end of its useful life."
        ),
        "cycles_unit": "cycles",
        "m_rul_label": "Remaining Useful Life",
        "m_rul_gloss": (
            "**Remaining Useful Life (RUL)** — roughly how many flight cycles "
            "this engine has left before maintenance-critical degradation. Higher "
            "is healthier. (True RUL is held out for scoring only and never shown "
            "to the model.)"
        ),
        "m_rul_miss": (
            "This estimate is typically off by about ±{miss} cycles, and more near "
            "end-of-life, where the model tends to read optimistic."
        ),
        "m_risk_label": "Risk",
        "m_lastcycle_label": "Flight cycles flown so far",
        "risk_high": "schedule inspection soon — near end-of-life",
        "risk_medium": "degrading — monitor closely",
        "risk_low": "healthy — keep monitoring",
        "trends_caption": (
            "Each panel is one of the model's most-informative sensors over this "
            "engine's most recent cycles; the arrow shows its overall direction "
            "(↑ rising, ↓ falling, → flat). A **sustained drift that several "
            "physically-related sensors share** is the fingerprint of real "
            "degradation as the engine nears end-of-life; a single lone jump is "
            "more likely sensor noise."
        ),
        "table_exp_label": "🔎 What do these sensors measure?",
        "table_h_sensor": "Sensor",
        "table_h_symbol": "Symbol",
        "table_h_meaning": "What it measures",
        "table_provenance": (
            "Physical meanings are from Saxena et al., *Damage Propagation "
            "Modeling for Aircraft Engine Run-to-Failure Simulation* (PHM08, "
            "Table 1) — the paper cited in the dataset readme. The dataset's own "
            "`readme.txt` labels these columns only as 'sensor measurement 1–21'; "
            "the names here are the community-standard reading of that reference, "
            "not printed in the readme. Sensors 20/21 (HPT/LPT coolant bleed) "
            "have a known ambiguity across sources; the canonical Table 1 "
            "ordering is used."
        ),
        "rep_summary_h": "In plain English",
        "rep_evidence_h": "What the data shows",
        "rep_evidence_note": "",
        "rep_fm_h": "Possible failure modes (quoted from the reference library)",
        "rep_fm_note": "",
        "rep_steps_h": "Recommended next steps",
        "rep_sources_label": "Sources cited in this report",
        "rep_sources_caption": (
            "Every failure mode and next step above is a verbatim quote from one "
            "of these knowledge-base sections — the assistant composes, it does "
            "not invent."
        ),
        "rep_uncertainty_label": "Uncertainty.",
        "rep_uncertainty": "",  # EN uses the assistant's own note text
        "rep_humanreview": (
            "Human review required — a qualified engineer must confirm this "
            "before any maintenance action. This tool focuses attention; it does "
            "not decide."
        ),
        "safety_note": (
            "Independent R&D prototype built on public NASA C-MAPSS simulation "
            "data. It does not make safety-critical decisions, does not command "
            "any action, and is not affiliated with any equipment manufacturer. "
            "Every output is advisory and must be reviewed by a qualified human."
        ),
        "no_sensor": "No sensor data available for this engine.",
        "no_evidence": (
            "No evidence record for this engine. Ensure the DS artifacts exist "
            "and run `src/diagnostics/build_evidence.py`."
        ),
        "pred_missing": (
            "Predictions artifact not found at {path}. Run the data/model "
            "pipeline first."
        ),
    },
    "中文": {
        "app_title": "状态监测——自动巡检 + 证据查看器",
        "limitations": (
            "独立研究原型，使用 NASA 公开模拟数据，与任何设备制造商无关。"
        ),
        "mode_label": "模式",
        # ---------- Agent Chat ----------
        "chat_badge": "确定性 · 有据可查 · 无 LLM",
        "chat_intro": (
            "本智能体运行状态监测管道并回答机队问题。可以试试：扫机队、诊断一台、"
            "或看它怎么工作。"
        ),
        "greeting": (
            "你好——我可以扫机队、诊断某一台、或解释我怎么工作。你想做啥？"
        ),
        "showcase": (
            "示例——运行 list_units_by_risk：**{n} 台高风险**；优先检查 "
            "**{ids}** 号（预测剩余寿命 ≤{cap} 周期）。"
        ),
        "showcase_src": "test_predictions.csv · count:risk_band=high",
        "chip_fleet": "🛰 扫描机队",
        "chip_diag": "🩺 诊断一台",
        "chip_how": "⚙️ 怎么工作的?",
        "chat_input_ph": "输入一个请求……（比如 diagnose unit 81）",
        "chat_reset": "🗑 清空对话",
        "howitworks": (
            "运行一条固定的 **10 阶段** 管道（读入 → 清理 → 特征 → 打分 → 选型 → "
            "预测 → 证据 → 诊断 → 评测），确定、种子固定。每个数字都能追到具名产物，"
            "每条建议都引用知识库出处。回路里没有 LLM。"
        ),
        "plan_preview": (
            "计划：载入 100 台 → 清理 → 算寿命 → 按风险排序 → 给需你拍板的备证据。"
            "约 10 秒，只读，不发工单。跑吗？"
        ),
        "chat_btn_start": "开始 ▶",
        "chat_btn_dry": "改范围→dry-run",
        "chat_starting": "已开始完整分析（把关模式，只读）。",
        "chat_dry_starting": "已开始试运行——只报告流程会怎么走，不产生决策。",
        "run_progress_running": "正在跑管线……当前：{stage}",
        "run_cache_reuse": "{n} 步复用了缓存",
        "run_done_expander": "✓ 完成 · 查看步骤",
        "chat_fallback": (
            "这个不在本智能体的范围内。它能：**扫机队**、"
            "**诊断某台发动机**（比如 “diagnose unit 81”）、或**解释它怎么工作**。"
        ),
        "chat_tools_label": "🔧 工具",
        "chat_grounded": "有据可查 — 追踪",
        "chat_citations": "引用",
        "decision_pending": "{n} 个决策待办 ↓",
        "why_expander": "为什么?",
        "chat_resolved": "✓ 已确认 · {act}",
        "skip_results": "⏭ 跳到结果",
        "fail_step": "{stage} 步失败。没改动任何东西。",
        "fail_retry": "重试",
        "fail_details": "查看细节",
        "queued_run": "已排队——本次跑完再运行。",
        "chat_done_note": "每张卡片的依据都锚定在具名的产物上，绝不编造。",
        # ---------- Autopilot page ----------
        "ap_intro": (
            "智能体把重活干完——读数→建模→预测→总结——只把最关键的信号浓缩成一个"
            "决策收件箱交给你。最后你来拍板。"
        ),
        "ap_naming": (
            "确定性管道智能体，人工决策把关。它把分析自动化；决定权在你。"
            "它不是自主运行，也绝不闭合任何安全关键回路。"
        ),
        "ap_run_header": "1 · 运行智能体",
        "btn_run_gated": "▶ 运行自动巡检（把关模式）",
        "btn_run_dry": "试运行（不产生决策卡）",
        "ap_running": "智能体运行中……正在实时观察管线（约每秒刷新）。",
        "ap_watch_header": "2 · 看它干活",
        "ap_no_run": "还没有运行——点 **运行自动巡检** 开始跑管线。",
        "ap_run_line": "运行 {rid} — 走 {n} 个阶段",
        "ap_skipped": "已是最新——跳过",
        "ap_ran": "已运行",
        "ap_details": "细节",
        "ap_why": "为什么",
        "ap_rundone": "完成：{ran} 个阶段运行，{skipped} 个跳过，用时 {sec:.2f} 秒。",
        "ap_halt": "已叫停",
        "inbox_header": "3 · 决策收件箱",
        "inbox_empty": (
            "目前没有待处理的决策。用把关模式运行智能体来生成收件箱——"
            "旗舰级的分诊卡会出现在这里。"
        ),
        "done_banner": (
            "智能体给 {n} 台发动机打了分，标记了 {h} 台高风险，并为每台整理了证据。"
            "**你有 {d} 个待决策。**"
        ),
        "digest_header": "🟢 健康发动机（已自动放行）",
        "digest_line": "{low} 台发动机为低风险、已自动放行——无需处理。",
        "handled_header": "✅ 已处理",
        "handled_empty": "还没有已处理的。",
        "handled_line": "决策卡 `{cid}` → 动作 **{act}**（{actor}）",
        "card_signals_h": "依据",
        "card_actions_h": "你的选择",
        "card_grounded": "依据",
        "card_evidence_btn": "🔍 查看 {u} 号机组的证据（教学模式）",
        "card_evidence_link": "证据：",
        "action_done": "已记录你的决定——正在继续运行智能体，把后面的阶段走完。",
        # ---------- Fleet + EDA ----------
        "fleet_header": "4 · 机队总览——全部发动机",
        "fleet_caption": (
            "每一台测试发动机，连同它的预测剩余寿命和风险等级，最急的排在前面。"
            "点列头可以重新排序。"
        ),
        "fleet_c_unit": "发动机",
        "fleet_c_last": "已飞周期",
        "fleet_c_pred": "预测剩余寿命",
        "fleet_c_band": "风险",
        "eda_header": "📊 看看数据 / Explore the data",
        "eda_cap_mono": (
            "哪些传感器跟着磨损走：每个传感器与剩余寿命的相关性。相关性强的"
            "（Ps30、T50、phi…）就是模型主要依赖的退化信号。"
        ),
        "eda_cap_flat": (
            "被丢掉的哑传感器：在 FD001 上有几路几乎不动，带不出退化信号，"
            "建模前就被排除。"
        ),
        "eda_cap_life": (
            "发动机能跑多久才坏（中位数约 199 个周期）。因为约 39% 的训练数据处在 "
            "{cap} 周期以上的健康阶段，剩余寿命标签被封顶在 {cap}。"
        ),
        "eda_cap_predvtrue": (
            "测试集上预测寿命 vs 真实寿命——落在对角线上就是完美；模型在中期更准，"
            "接近报废时更松。"
        ),
        "eda_cap_errhist": (
            "预测误差的分布——大致以零为中心，尾部是模型接近寿命末端时偏乐观的部分。"
        ),
        "eda_cap_degrad": (
            "几条退化轨迹示例：几台发动机接近报废时关键传感器的漂移。"
        ),
        # ---------- Teaching mode (the wizard) ----------
        "title": "预测发动机什么时候需要维护",
        "what_is_this": (
            "一个 5 步向导：每按一个按钮，就跑管线的下一步。它读取一台飞机发动机的"
            "传感器，估计它还能安全再飞多久，并写一段维护说明——里面每个说法和每一步"
            "建议都标了知识库出处，绝不瞎编。"
        ),
        "lbl_function": "功能：",
        "lbl_purpose": "目的：",
        "sb_controls": "控制项",
        "sb_cycles_label": "画最近多少个周期",
        "sb_kb_empty": "知识库是空的——暂时给不了诊断建议。",
        "step1_title": "① 第 1 步——选一台发动机",
        "step1_desc": "看传感器——取这台发动机最近几个飞行周期的读数（温度、压力、转速）。",
        "step1_func": "从 NASA 测试机队里选一台发动机，载入它的原始传感器历史。",
        "step1_purpose": "后面所有步骤都是针对这一台发动机的——你来决定看哪一台。",
        "step1_engine_label": "发动机（测试单元）编号",
        "step2_title": "② 第 2 步——看结论",
        "step2_desc": (
            "做预测——一个简单的机器学习模型估计这台发动机的剩余可用寿命：还能再飞"
            "多少个周期才需要维护。"
        ),
        "step2_func": "模型根据这台发动机的传感器历史，算出大概还能飞多少个周期。",
        "step2_purpose": "把 21 路传感器数字变成一个能做决策的数字。",
        "step3_title": "③ 第 3 步——看证据",
        "step3_desc": "看传感器——取这台发动机最近几个飞行周期的读数（温度、压力、转速）。",
        "step3_func": "把最有信息量的几个传感器最近的走势画出来，并标出每个到底测的是什么。",
        "step3_purpose": "让人能拿预测去核对原始信号——发动机是真在退化，还是传感器噪声？",
        "step4_title": "④ 第 4 步——看建议",
        "step4_desc_retrieval": (
            "查资料——系统在一个小的维护/工程知识库里，找跟这台发动机状况相关的段落。"
        ),
        "step4_desc_report": (
            "出报告——写一段简短诊断，里面每一个可能故障和每一步建议，都从那些资料里"
            "带出处地引用；要是没找到相关内容，它会直说，绝不瞎猜。"
        ),
        "step4_func": "先在知识库上做 TF-IDF 检索，再按模板拼出一份带引用的报告（不靠大模型瞎编）。",
        "step4_purpose": "让每条建议都有出处、可追溯，而不是 AI 拍脑袋。",
        "step5_title": "⑤ 第 5 步——记住边界",
        "step5_desc": "最终一定由合格的人来拍板。",
        "step5_func": "把模型的不确定性摆出来，任何动作前都要求人工复核。",
        "step5_purpose": "这是决策支持，不是决策本身——它帮你集中注意力，绝不替你做安全关键的决定。",
        "btn_load": "① 载入传感器历史",
        "btn_predict": "② 预测剩余寿命",
        "btn_evidence": "③ 看证据",
        "btn_report": "④ 生成带引用的诊断报告",
        "btn_runall": "▶ 一键跑完",
        "concl_high": "{u} 号发动机预计还能飞约 {r} 个周期——高风险，建议尽快安排检查。",
        "concl_medium": "{u} 号发动机预计还能飞约 {r} 个周期——中风险，正在退化，盯紧点。",
        "concl_low": "{u} 号发动机预计还能飞约 {r} 个周期——低风险，看起来挺健康，继续监测就行。",
        "health_text": "预计剩余寿命：{r} / {cap} 个周期（越满越健康）",
        "health_caption": (
            "这个条表示预测的剩余寿命，满格是模型设的上限 {cap} 个周期。"
            "条快空了，就说明这台发动机快到寿命尽头了。"
        ),
        "cycles_unit": "个周期",
        "m_rul_label": "剩余可用寿命",
        "m_rul_gloss": (
            "**剩余可用寿命（RUL）**——大概还能再飞多少个周期，数字越大越健康。"
            "（真实寿命只用来打分，不会给模型看。）"
        ),
        "m_rul_miss": (
            "这个估计通常有大约 ±{miss} 个周期的误差，越接近寿命末端越大，"
            "而且这时模型往往偏乐观。"
        ),
        "m_risk_label": "风险等级",
        "m_lastcycle_label": "已经飞了多少周期",
        "risk_high": "尽快安排检查——快到寿命尽头了",
        "risk_medium": "在退化，盯紧点",
        "risk_low": "健康，继续监测",
        "trends_caption": (
            "每一格是模型最看重的一个传感器在最近这些周期的走势；箭头表示总体方向"
            "（↑ 上升、↓ 下降、→ 基本平稳）。**好几个相关传感器一起持续往一个方向漂**，"
            "才是真退化的信号；单独一个猛跳一下，多半是传感器噪声。"
        ),
        "table_exp_label": "🔎 这些传感器测的是什么？",
        "table_h_sensor": "传感器",
        "table_h_symbol": "符号",
        "table_h_meaning": "测量的是",
        "table_provenance": (
            "这些物理含义来自 Saxena 等人的论文《Damage Propagation Modeling for "
            "Aircraft Engine Run-to-Failure Simulation》（PHM08，表 1）——也就是"
            "数据集 readme 里引用的那篇。数据集自带的 `readme.txt` 只把这些列标成 "
            "'sensor measurement 1–21'，具体物理名字是学界对那篇论文的通用解读，"
            "readme 里并没有写。其中 20/21 号（高压/低压涡轮冷却引气）在不同资料里"
            "有已知的歧义，这里用的是该论文表 1 的标准顺序。"
        ),
        "rep_summary_h": "大白话总结",
        "rep_evidence_h": "数据显示了什么",
        "rep_evidence_note": "（下面是传感器和模型的原始数据，英文）",
        "rep_fm_h": "可能的故障模式",
        "rep_fm_note": "知识库原文（英文）：",
        "rep_steps_h": "建议的下一步（知识库原文，英文）",
        "rep_sources_label": "本报告引用的资料",
        "rep_sources_caption": (
            "上面每一个故障模式和下一步，都是从这些知识库段落里一字不差引用的——"
            "助手只做拼装，不编造。"
        ),
        "rep_uncertainty_label": "不确定性说明：",
        "rep_uncertainty": (
            "这个剩余寿命只是一个粗略的点估计，没有置信区间。模型训练时把寿命目标"
            "压到了上限（约 {cap} 个周期），所以健康发动机的预测会被压到接近上限，"
            "不能当成精确的周期数。越接近寿命尽头，预测误差越大。这个结果只能用来帮忙"
            "排查优先级，不能当成权威的失效时间，必须人工复核。"
        ),
        "rep_humanreview": (
            "必须人工复核——任何维护动作之前，都要由合格的工程师确认。这个工具只是帮你"
            "把注意力集中到该看的地方，它不做决定。"
        ),
        "safety_note": (
            "独立研究原型，基于 NASA 公开的 C-MAPSS 模拟数据。它不做安全关键决策，"
            "不下达任何指令，也与任何设备厂商无关。所有输出都只是建议，任何维护动作前"
            "都必须由合格的人工复核。"
        ),
        "no_sensor": "这台发动机没有可用的传感器数据。",
        "no_evidence": (
            "这台发动机还没有证据记录。确认 DS 产物存在后，运行 "
            "src/diagnostics/build_evidence.py。"
        ),
        "pred_missing": "在 {path} 找不到预测文件。请先跑数据/模型流水线。",
    },
}


# =============================================================================
# Shared helpers
# =============================================================================
def _short_label(sensor_col: str) -> str:
    meta = SENSOR_META.get(sensor_col)
    return meta[0] if meta else sensor_col


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (ValueError, OSError):
        return None


def _load_cfg() -> PipelineConfig:
    return PipelineConfig.load()


def _state_dir(cfg: PipelineConfig) -> Path:
    """Where the app reads the autopilot journal/inbox and writes answers.

    Override with ``CM_APP_REPORTS_DIR`` (used by tests to point at a fixture);
    defaults to the real ``reports/`` the agent subprocess writes to.
    """
    override = os.environ.get("CM_APP_REPORTS_DIR")
    return Path(override) if override else cfg.path("reports")


def _pipeline_values(cfg: PipelineConfig) -> dict:
    """Single source of truth: RUL cap, typical-miss, risk cutoffs.

    Read from ``reports/metrics_model.json`` + ``config/pipeline.yaml`` — never
    hard-coded. ``miss`` is the capped-truth MAE (the honest 'typical miss').
    """
    rul_cap = int(cfg.rul_cap)
    miss = 12
    m = _read_json(cfg.path("reports") / "metrics_model.json")
    if m:
        rul_cap = int(m.get("rul_cap", rul_cap))
        capped = m.get("metrics_vs_capped_truth") or {}
        if isinstance(capped.get("mae"), (int, float)):
            miss = int(round(float(capped["mae"])))
    return {
        "rul_cap": rul_cap,
        "miss": miss,
        "high_max": cfg.risk_thresholds.high_max,
        "medium_max": cfg.risk_thresholds.medium_max,
    }


def _launch(cfg: PipelineConfig, autonomy: str):
    """Launch the agent as a subprocess (never train in-process). Returns the
    Popen handle, or None when subprocess launching is disabled for tests."""
    if os.environ.get("CM_APP_NO_SUBPROCESS"):
        return None
    return subprocess.Popen(
        [sys.executable, "-m", "src.agent", "run", "--all", "--autonomy", autonomy],
        cwd=str(cfg.root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _answer_card(cfg: PipelineConfig, card_id: str, action_id: str) -> Path:
    """Write the answered-card file the supervisor polls to resume.

    The supervisor reads ``action``; the plan's schema names ``action_id`` — we
    write both, plus ``card_id``, ``ts``, ``actor`` so resume works and the
    documented schema is satisfied.
    """
    ans_dir = _state_dir(cfg) / "autopilot_inbox" / "answered"
    ans_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "card_id": card_id,
        "action": action_id,
        "action_id": action_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "actor": "ui",
    }
    path = ans_dir / f"{card_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def _circled(i: int) -> str:
    return _CIRCLED[i - 1] if 1 <= i <= len(_CIRCLED) else f"{i}."


def compose_zh_summary(evidence: dict) -> str:
    """Plain-Chinese summary composed deterministically from evidence fields.

    Kept in the app (not in the assistant) so src/rag/assistant.py and its
    contract stay untouched; uses unit, cycle, predicted RUL, risk band, and the
    top sensors + their trends.
    """
    u = evidence.get("asset_id")
    cycle = evidence.get("last_cycle")
    r = round(float(evidence.get("predicted_rul", 0)))
    band = str(evidence.get("risk_band", "")).lower()
    band_zh = _BAND_ZH.get(band, band)
    ss = evidence.get("sensor_summary", {})
    parts = []
    for name in list(ss.keys())[:3]:
        cn = SENSOR_META.get(name, (name, "", name))[2] or name
        tr = _TREND_ZH.get(ss[name].get("trend"), "有变化")
        parts.append(f"{cn}（{tr}）")
    sensors_clause = "、".join(parts) if parts else "几个关键信号"
    return (
        f"{u} 号发动机已经飞了 {cycle} 个周期。模型预测它大概还能再飞约 {r} 个周期，"
        f"属于{band_zh}。最近这段时间，{sensors_clause}这几个关键信号在变化。"
        f"注意：这只是一个粗略的点估计，不是精确倒计时；下面列出的可能故障和处理建议，"
        f"都直接引用自知识库原文（英文），最终必须由工程师人工确认。"
    )


def _load_evidence(unit_id: int) -> dict | None:
    """Return the evidence record for a unit, building all records on demand if
    the DS artifacts exist but evidence hasn't been generated yet."""
    path = be.EVIDENCE_DIR / f"unit_{unit_id}.json"
    if not path.exists():
        if be.PRED_PATH.exists() and be.FI_PATH.exists() and be.RAW_TEST_PATH.exists():
            be.run()
        if not path.exists():
            return None
    return json.loads(path.read_text())


# =============================================================================
# Autopilot page
# =============================================================================
def _autopilot_page(st, plt, pd, cfg, tt, lang, pv) -> None:
    st.markdown(f"#### {tt['ap_intro']}")
    st.info(tt["ap_naming"], icon="🤖")

    flash = st.session_state.pop("ap_flash", None)
    if flash:
        st.success(flash)

    proc = st.session_state.get("ap_proc")
    running = proc is not None and proc.poll() is None

    # --- 1 · Run controls --------------------------------------------------
    st.subheader(tt["ap_run_header"])
    c1, c2 = st.columns(2)
    if c1.button(tt["btn_run_gated"], key="ap_run_gated", disabled=running,
                 width="stretch"):
        st.session_state.ap_proc = _launch(cfg, "gated")
        st.rerun()
    if c2.button(tt["btn_run_dry"], key="ap_run_dry", disabled=running,
                 width="stretch"):
        st.session_state.ap_proc = _launch(cfg, "dry-run")
        st.rerun()

    # --- 2 · Watch it work (live timeline) ---------------------------------
    st.subheader(tt["ap_watch_header"])
    _render_timeline(st, cfg, tt, lang)

    if running:
        st.info(tt["ap_running"])
        time.sleep(1.0)
        st.rerun()
        return
    if proc is not None:
        st.session_state.ap_proc = None  # just finished

    # --- 3 · Decision Inbox ------------------------------------------------
    _render_inbox(st, pd, cfg, tt, lang, pv)

    # --- 4 · Fleet view + EDA ----------------------------------------------
    _render_fleet(st, plt, pd, cfg, tt, lang, pv)


def _render_timeline(st, cfg, tt, lang) -> None:
    """Render the latest run's journal as a narrative step-by-step timeline."""
    events = read_events(_state_dir(cfg) / "autopilot_journal.jsonl")
    if not events:
        st.info(tt["ap_no_run"])
        return
    last_run = events[-1].get("run_id")
    events = [e for e in events if e.get("run_id") == last_run]

    started = next((e for e in events if e["type"] == "run_started"), None)
    if started:
        st.caption(tt["ap_run_line"].format(rid=last_run, n=len(started.get("stages", []))))

    halted = next((e for e in events if e["type"] == "halt"), None)

    for idx, stage in enumerate(STAGE_ORDER, 1):
        sev = [e for e in events if e.get("stage") == stage]
        if not any(e["type"] == "stage_started" for e in sev):
            continue
        spec = STAGE_SPECS[stage]
        headline = spec.what if lang == "English" else spec.zh_what
        done = next((e for e in sev if e["type"] == "stage_done"), None)
        skipped = bool(done and done.get("skipped"))
        icon = "⏭️" if skipped else "✅"
        gate_raised = [e for e in sev if e["type"] == "gate_raised"]
        if gate_raised:
            icon = "⚠️"
        st.markdown(f"**{icon} {_circled(idx)}  {headline}**")
        if skipped:
            st.caption(f"— {tt['ap_skipped']}")
        for gr in gate_raised:
            st.caption(f"⚠️ {tt['card_actions_h']}: {gr.get('kind')} ({gr.get('card_id')})")
        with st.expander(tt["ap_details"]):
            why = spec.why if lang == "English" else spec.zh_why
            st.caption(f"**{tt['ap_why']}** {why}")
            if done:
                st.caption(
                    f"{tt['ap_ran'] if not skipped else tt['ap_skipped']} · "
                    f"{done.get('seconds', 0)}s · rows={done.get('rows')}"
                )
            for pe in [e for e in sev if e["type"] == "stage_progress"]:
                st.caption(f"· {pe.get('message')}")
            for ar in [e for e in sev if e["type"] == "artifact"]:
                st.caption(f"📦 {ar.get('path')}")

    if halted:
        st.error(f"⛔ {tt['ap_halt']} @ {halted.get('stage')} — {halted.get('detail')}")
    ran = next((e for e in events if e["type"] == "run_done"), None)
    if ran:
        st.caption(
            tt["ap_rundone"].format(
                ran=ran.get("stages_run", 0),
                skipped=ran.get("stages_skipped", 0),
                sec=ran.get("seconds", 0.0),
            )
        )


def _render_inbox(st, pd, cfg, tt, lang, pv) -> None:
    st.subheader(tt["inbox_header"])
    state = _state_dir(cfg)
    pending = sorted((state / "autopilot_inbox" / "pending").glob("*.json"))
    answered = sorted((state / "autopilot_inbox" / "answered").glob("*.json"))

    preds = pd.read_csv(be.PRED_PATH)
    n_scored = int(len(preds))
    n_high = int((preds["risk_band"] == "high").sum())
    n_low = int((preds["risk_band"] == "low").sum())

    banner = tt["done_banner"].format(n=n_scored, h=n_high, d=len(pending))
    (st.warning if pending else st.success)(banner)

    if not pending:
        st.caption(tt["inbox_empty"])
    for pf in pending:
        card = _read_json(pf)
        if card:
            _render_card(st, card, cfg, tt, lang)

    with st.expander(tt["handled_header"]):
        if answered:
            for af in answered:
                a = _read_json(af) or {}
                st.markdown(
                    tt["handled_line"].format(
                        cid=a.get("card_id", af.stem),
                        act=a.get("action_id") or a.get("action", "?"),
                        actor=a.get("actor", "?"),
                    )
                )
        else:
            st.caption(tt["handled_empty"])

    with st.expander(tt["digest_header"]):
        st.caption(tt["digest_line"].format(low=n_low))


def _render_card(st, card, cfg, tt, lang) -> None:
    en = lang == "English"
    prio = card.get("priority", "P3")
    verdict = card["verdict_en"] if en else card["verdict_zh"]
    alt = card["verdict_zh"] if en else card["verdict_en"]
    box = {"P1": st.error, "P2": st.warning}.get(prio, st.info)
    box(f"**{prio} · {card.get('kind')}** — {verdict}")
    st.caption(alt)

    st.markdown(f"**{tt['card_signals_h']}**")
    for s in card.get("signals", []):
        st.markdown(f"- {s['text_en'] if en else s['text_zh']}")
        st.caption(f"{tt['card_grounded']}: `{s.get('artifact')}` [{s.get('field')}]")

    link = card.get("evidence_link", "")
    m = re.search(r"unit_(\d+)", str(link))
    if m:
        uid = int(m.group(1))
        if st.button(tt["card_evidence_btn"].format(u=uid), key=f"ev_{card['id']}"):
            st.session_state["app_mode"] = MODE_TEACH
            st.session_state["engine_select"] = uid
            st.session_state["stage"] = 4
            st.session_state["stage_unit"] = uid
            st.rerun()
    elif link:
        st.caption(f"{tt['card_evidence_link']} `{link}`")

    st.markdown(f"**{tt['card_actions_h']}**")
    actions = card.get("actions", [])
    cols = st.columns(len(actions)) if actions else []
    for col, a in zip(cols, actions):
        label = (a["label_en"] if en else a["label_zh"]) + (
            " ✅" if a.get("safe_default") else ""
        )
        if col.button(label, key=f"act_{card['id']}_{a['id']}"):
            _answer_card(cfg, card["id"], a["id"])
            st.session_state["ap_flash"] = tt["action_done"]
            st.session_state.ap_proc = _launch(cfg, "gated")
            st.rerun()
        col.caption(a["consequence_en"] if en else a["consequence_zh"])


def _render_fleet(st, plt, pd, cfg, tt, lang, pv) -> None:
    st.subheader(tt["fleet_header"])
    st.caption(tt["fleet_caption"])
    preds = pd.read_csv(be.PRED_PATH).sort_values("pred_rul").reset_index(drop=True)
    view = preds[["unit_id", "last_cycle", "pred_rul", "risk_band"]].rename(
        columns={
            "unit_id": tt["fleet_c_unit"],
            "last_cycle": tt["fleet_c_last"],
            "pred_rul": tt["fleet_c_pred"],
            "risk_band": tt["fleet_c_band"],
        }
    )
    band_col = tt["fleet_c_band"]

    def _row_style(row):
        color = BAND_HEX.get(str(row[band_col]).lower(), "")
        return [
            f"background-color:{color};color:white;font-weight:600"
            if (color and c == band_col)
            else ""
            for c in row.index
        ]

    st.dataframe(view.style.apply(_row_style, axis=1),
                 width="stretch", height=360)

    with st.expander(tt["eda_header"]):
        charts = [
            (cfg.path("eda") / "monotonicity.png", tt["eda_cap_mono"]),
            (cfg.path("eda") / "flat_sensors.png", tt["eda_cap_flat"]),
            (cfg.path("eda") / "lifetime_distribution.png",
             tt["eda_cap_life"].format(cap=pv["rul_cap"])),
            (cfg.path("figures") / "pred_vs_true.png", tt["eda_cap_predvtrue"]),
            (cfg.path("figures") / "error_hist.png", tt["eda_cap_errhist"]),
            (cfg.path("figures") / "degradation_units.png", tt["eda_cap_degrad"]),
        ]
        shown = 0
        for png, cap in charts:
            if Path(png).exists():
                st.image(str(png), width="stretch")
                st.caption(cap)
                shown += 1
        if shown == 0:
            st.caption("EDA figures not found — run the pipeline's EDA stage.")


# =============================================================================
# Teaching mode (the button-driven wizard, reused as evidence viewer)
# =============================================================================
def _step_header(st, tt, title_key, desc, func_key, purpose_key) -> None:
    st.subheader(tt[title_key])
    for d in desc if isinstance(desc, list) else [desc]:
        st.caption(d)
    st.markdown(
        f"**{tt['lbl_function']}** {tt[func_key]}  \n"
        f"**{tt['lbl_purpose']}** {tt[purpose_key]}"
    )


def _teaching_mode(st, plt, pd, cfg, tt, lang, pv) -> None:
    from src.rag.retriever import Retriever  # local import keeps module import light

    @st.cache_resource
    def get_retriever() -> Retriever:
        return Retriever(be.KB_DIR)

    @st.cache_data
    def get_predictions():
        return pd.read_csv(be.PRED_PATH)

    @st.cache_data
    def get_raw():
        return be.load_raw_test()

    st.markdown(f"### {tt['title']}")
    st.markdown(f"#### {tt['what_is_this']}")

    window = st.sidebar.slider(tt["sb_cycles_label"], 10, 60, be.LAST_WINDOW)

    if not be.PRED_PATH.exists():
        st.error(tt["pred_missing"].format(path=be.PRED_PATH))
        return

    preds = get_predictions()
    raw = get_raw()
    retriever = get_retriever()
    if len(retriever) == 0:
        st.sidebar.info(tt["sb_kb_empty"])

    if "stage" not in st.session_state:
        st.session_state.stage = 0
    if "stage_unit" not in st.session_state:
        st.session_state.stage_unit = None

    rul_cap = pv["rul_cap"]

    # STEP ① — Pick an engine
    st.divider()
    _step_header(st, tt, "step1_title", tt["step1_desc"], "step1_func", "step1_purpose")
    unit_ids = sorted(preds["unit_id"].astype(int).tolist())
    unit_id = st.selectbox(tt["step1_engine_label"], unit_ids, key="engine_select")
    if st.session_state.stage_unit != unit_id:
        st.session_state.stage = 0
        st.session_state.stage_unit = unit_id

    b1, b_all = st.columns(2)
    if b1.button(tt["btn_load"], key="btn_load", width="stretch"):
        st.session_state.stage = max(st.session_state.stage, 1)
    if b_all.button(tt["btn_runall"], key="btn_runall", width="stretch"):
        st.session_state.stage = 4

    prow = preds[preds["unit_id"] == int(unit_id)].iloc[0]
    band = str(prow["risk_band"]).lower()
    rul = round(float(prow["pred_rul"]))
    last_cycle = int(prow["last_cycle"])

    # STEP ② — Read the conclusion (conclusion-first)
    if st.session_state.stage >= 1:
        st.divider()
        _step_header(st, tt, "step2_title", tt["step2_desc"], "step2_func", "step2_purpose")
        if st.button(tt["btn_predict"], key="btn_predict"):
            st.session_state.stage = max(st.session_state.stage, 2)
        if st.session_state.stage >= 2:
            conclusion = tt[f"concl_{band}"].format(u=unit_id, r=rul)
            if band == "high":
                st.error(f"### {conclusion}")
            elif band == "medium":
                st.warning(f"### {conclusion}")
            else:
                st.success(f"### {conclusion}")
            frac = min(max(rul / rul_cap, 0.0), 1.0)
            st.progress(frac, text=tt["health_text"].format(r=rul, cap=rul_cap))
            st.caption(tt["health_caption"].format(cap=rul_cap))
            c1, c2, c3 = st.columns(3)
            c1.metric(tt["m_rul_label"], f"{rul} {tt['cycles_unit']}")
            c2.metric(tt["m_risk_label"], RISK_WORD[lang][band], help=tt[f"risk_{band}"])
            c3.metric(tt["m_lastcycle_label"], last_cycle)
            st.caption(tt["m_rul_gloss"])
            st.caption(tt["m_rul_miss"].format(miss=pv["miss"]))

    evidence = _load_evidence(int(unit_id)) if st.session_state.stage >= 2 else None

    # STEP ③ — See the evidence
    if st.session_state.stage >= 2:
        st.divider()
        _step_header(st, tt, "step3_title", tt["step3_desc"], "step3_func", "step3_purpose")
        if st.button(tt["btn_evidence"], key="btn_evidence"):
            st.session_state.stage = max(st.session_state.stage, 3)
        if st.session_state.stage >= 3:
            unit_raw = raw[raw["unit"] == int(unit_id)].sort_values("cycle").tail(window)
            if evidence and evidence.get("sensor_summary"):
                sensors = list(evidence["sensor_summary"].keys())
                trends = {k: v.get("trend") for k, v in evidence["sensor_summary"].items()}
            else:
                sensors = be.rank_sensors_from_importances(
                    be.load_feature_importances(), be.TOP_K_SENSORS
                )
                trends = {}
            sensors = [s for s in sensors if s in unit_raw.columns][:6]
            if sensors and not unit_raw.empty:
                ncol = 2
                nrow = (len(sensors) + ncol - 1) // ncol
                fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.3 * nrow), squeeze=False)
                for i, s in enumerate(sensors):
                    ax = axes[i // ncol][i % ncol]
                    ax.plot(unit_raw["cycle"], unit_raw[s], marker=".", linewidth=1)
                    arrow = _TREND_ARROW.get(trends.get(s), "")
                    ax.set_title(f"{_short_label(s)}  {arrow}", fontsize=9)
                    ax.set_xlabel("cycle", fontsize=7)
                    ax.tick_params(labelsize=7)
                for j in range(len(sensors), nrow * ncol):
                    axes[j // ncol][j % ncol].axis("off")
                fig.tight_layout()
                st.pyplot(fig)
                st.caption(tt["trends_caption"])
                with st.expander(tt["table_exp_label"]):
                    midx = 1 if lang == "English" else 2
                    table = pd.DataFrame(
                        [
                            {
                                tt["table_h_sensor"]: s,
                                tt["table_h_symbol"]: SENSOR_META.get(s, ("?", "", ""))[0],
                                tt["table_h_meaning"]: SENSOR_META.get(s, ("", "—", "—"))[midx],
                            }
                            for s in sensors
                        ]
                    )
                    st.table(table)
                    st.caption(tt["table_provenance"])
            else:
                st.info(tt["no_sensor"])

    # STEP ④ — Read the guidance (retrieval + cited report)
    if st.session_state.stage >= 3:
        st.divider()
        _step_header(
            st, tt, "step4_title",
            [tt["step4_desc_retrieval"], tt["step4_desc_report"]],
            "step4_func", "step4_purpose",
        )
        if st.button(tt["btn_report"], key="btn_report"):
            st.session_state.stage = max(st.session_state.stage, 4)
        if st.session_state.stage >= 4:
            if not evidence:
                st.info(tt["no_evidence"])
            else:
                report = diagnose(evidence, retriever)
                st.markdown(f"**{tt['rep_summary_h']}**")
                st.write(report["summary"] if lang == "English" else compose_zh_summary(evidence))
                st.markdown(f"**{tt['rep_evidence_h']}**")
                if tt["rep_evidence_note"]:
                    st.caption(tt["rep_evidence_note"])
                for item in report["supporting_evidence"]:
                    st.markdown(f"- {item}")
                st.markdown(f"**{tt['rep_fm_h']}**")
                if tt["rep_fm_note"]:
                    st.caption(tt["rep_fm_note"])
                for fm in report["possible_failure_modes"]:
                    if fm.get("source_file"):
                        st.markdown(f"> **{fm['failure_mode']}.** {fm['evidence']}")
                        st.caption(f"📄 {fm['source_file']} · {fm['section']}")
                    else:
                        st.info(fm["evidence"])
                st.markdown(f"**{tt['rep_steps_h']}**")
                for stp in report["recommended_next_steps"]:
                    if stp.get("source_file"):
                        st.markdown(f"☐ **{stp['step']}** — {stp['detail']}")
                        st.caption(f"📄 {stp['source_file']} · {stp['section']}")
                    else:
                        st.info(stp["detail"])
                st.markdown(f"**{tt['rep_sources_label']}**")
                if report["citations"]:
                    for cite in report["citations"]:
                        st.markdown(f"- `{cite['source_file']}` · **{cite['section']}**")
                st.caption(tt["rep_sources_caption"])

    # STEP ⑤ — Remember the limits (persistent footer, always visible)
    st.divider()
    _step_header(st, tt, "step5_title", tt["step5_desc"], "step5_func", "step5_purpose")
    st.warning(f"**{tt['step5_desc']}** {tt['rep_humanreview']}", icon="🧑‍🔧")
    uncertainty = (
        be.UNCERTAINTY_NOTE if lang == "English"
        else tt["rep_uncertainty"].format(cap=pv["rul_cap"])
    )
    st.info(f"**{tt['rep_uncertainty_label']}** {uncertainty}", icon="ℹ️")
    st.caption(tt["safety_note"])


# =============================================================================
# Agent Chat — a conversational rendering layer over the SAME machinery
# (rule planner + query orchestrator + autopilot subprocess + journal + inbox).
# Zero new agent logic lives here.
# =============================================================================
def _canonical(text: str) -> str:
    """Rendering-layer input normalisation only: map common Chinese phrasings to
    the shared rule planner's canonical English. No agent logic."""
    t = (text or "").strip()
    m = re.search(r"(\d{1,4})\s*号", t)
    if m and ("诊断" in t or "分析" in t) and "巡检" not in t:
        return f"diagnose unit {m.group(1)}"
    if "哪些" in t and ("检查" in t or "维护" in t):
        return "which engines need inspection?"
    return t


def _detect_intent(text: str) -> tuple[str, str]:
    """Classify chat input: 'howitworks', 'run' (full pipeline), 'query'
    (read-only ask), or 'unknown' — reusing the shared rule planner for the
    query/unknown split. Out-of-scope input falls to 'unknown' (bounded menu)."""
    from src.agent.planner import make_planner

    t = text or ""
    if _HOW_RE.search(t):
        return "howitworks", t
    if _RUN_RE.search(t):
        return "run", t
    canon = _canonical(t)
    if make_planner("rule").plan(canon):
        return "query", canon
    return "unknown", t


def _tools_line(canon: str) -> str:
    from src.agent.planner import make_planner

    names: list[str] = []
    for c in make_planner("rule").plan(canon):
        names.append(c.tool)
        names.extend(f"(fan-out) {ft}" for ft in c.fan_out)
    return " → ".join(names)


def _run_query(cfg, canon: str) -> dict:
    """In-process grounded query (read-only + fast) → a chat 'answer' message."""
    from src.agent.query import answer_query

    res = answer_query(cfg, canon)
    a = res["answer"]
    return {
        "role": "agent", "kind": "answer", "tools": _tools_line(canon),
        "en": a.get("answer_en", ""), "zh": a.get("answer_zh", ""),
        "citations": a.get("citations", []), "grounded": res.get("grounded", False),
        "trace": Path(res.get("trace_path", "")).name,
    }


def _greeting_msg() -> dict:
    return {"role": "agent", "kind": "text",
            "en": T["English"]["greeting"], "zh": T["中文"]["greeting"]}


def _showcase_msg() -> dict:
    """A grounded example result on the landing screen (never an empty box).
    Computed from the predictions artifact — reporting voice, no LLM."""
    import pandas as pd

    try:
        preds = pd.read_csv(be.PRED_PATH)
        high = preds[preds["risk_band"] == "high"].sort_values("pred_rul")
        first = high[high["pred_rul"] <= 10]
        if first.empty:
            first = high.head(3)
        ids = ", ".join(str(int(u)) for u in first["unit_id"].head(3))
        return {"role": "agent", "kind": "showcase",
                "n": int(len(high)), "ids": ids, "cap": 10}
    except (OSError, ValueError, KeyError):
        return {"role": "agent", "kind": "howitworks"}


def _handle_user_intent(st, cfg, tt, display_text, intent_text=None,
                        running=False) -> None:
    intent_text = intent_text if intent_text is not None else display_text
    st.session_state["chat"].append({"role": "user", "text": display_text})
    kind, payload = _detect_intent(intent_text)
    if kind == "howitworks":
        st.session_state["chat"].append({"role": "agent", "kind": "howitworks"})
    elif kind == "run":
        if running:  # a run is in flight — queue, never launch concurrently
            st.session_state["chat"].append({"role": "agent", "kind": "text",
                "en": T["English"]["queued_run"], "zh": T["中文"]["queued_run"]})
        else:
            st.session_state["chat"].append({"role": "agent", "kind": "plan"})
            st.session_state["awaiting_start"] = True
    elif kind == "query":
        # read-only: answerable immediately from existing artifacts, even mid-run.
        st.session_state["chat"].append(_run_query(cfg, payload))
    else:
        st.session_state["chat"].append({"role": "agent", "kind": "fallback"})


def _render_chat_msg(st, msg, tt, lang) -> None:
    en = lang == "English"
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        kind = msg.get("kind")
        if msg["role"] == "user":
            st.markdown(msg["text"])
        elif kind in ("text", "note"):
            st.markdown(msg["en"] if en else msg["zh"])
        elif kind == "done":
            st.markdown(msg["en"] if en else msg["zh"])
            st.caption(tt["chat_done_note"])
        elif kind == "resolved_card":
            st.success(msg["en"] if en else msg["zh"])
        elif kind == "howitworks":
            st.markdown(tt["howitworks"])
        elif kind == "showcase":
            st.markdown(tt["showcase"].format(n=msg["n"], ids=msg["ids"], cap=msg["cap"]))
            st.caption(f"📄 {tt['showcase_src']}")
        elif kind == "fallback":
            st.markdown(tt["chat_fallback"])
        elif kind == "plan":
            st.markdown(tt["plan_preview"])
        elif kind == "answer":
            st.caption(f"{tt['chat_tools_label']}: {msg['tools']}")
            st.markdown(msg["en"] if en else msg["zh"])
            lines = []
            for c in msg.get("citations") or []:
                if c.get("source_file"):
                    lines.append(f"`{c['source_file']}` › {c.get('section')}")
                elif c.get("artifact"):
                    lines.append(f"`{c['artifact']}` — {c.get('note', '')}")
            if lines:
                st.caption(f"{tt['chat_citations']}: " + "; ".join(lines))
            if msg.get("trace"):
                mark = "✓ " if msg.get("grounded") else ""
                st.caption(f"{mark}{tt['chat_grounded']}: `{msg['trace']}`")


def _latest_run_rows(cfg):
    """Per-stage rows for the latest journal run: (idx, stage, skipped, done, gate,
    artifacts). Returns (run_id, rows) or (None, [])."""
    events = read_events(_state_dir(cfg) / "autopilot_journal.jsonl")
    if not events:
        return None, []
    last = events[-1].get("run_id")
    ev = [e for e in events if e.get("run_id") == last]
    rows = []
    for idx, stage in enumerate(STAGE_ORDER, 1):
        sev = [e for e in ev if e.get("stage") == stage]
        if not any(e["type"] == "stage_started" for e in sev):
            continue
        done = next((e for e in sev if e["type"] == "stage_done"), None)
        gate = [e for e in sev if e["type"] == "gate_raised"]
        arts = [e for e in sev if e["type"] == "artifact"]
        rows.append((idx, stage, bool(done and done.get("skipped")), done, gate, arts))
    return last, rows


def _render_progress_bubble(st, cfg, tt, lang, running) -> None:
    """BUBBLE ECONOMY: stage progress lives in ONE bubble. While running it shows
    the current stage; when finished it collapses to a '✓ Done · view steps'
    expander whose CONTENT is the per-node reports (numbers + citations). Cached
    stages compress to a single 'N steps reused cache' line. No 'typing…' fakery."""
    last, rows = _latest_run_rows(cfg)
    if not rows:
        return
    n_cached = sum(1 for r in rows if r[2] and not r[4])

    if running:
        with st.chat_message("assistant"):
            cur = rows[-1][1].split("_", 1)[1]
            st.markdown(f"⏳ {tt['run_progress_running'].format(stage=cur)}")
            if n_cached:
                st.caption(tt["run_cache_reuse"].format(n=n_cached))
        return

    with st.chat_message("assistant"):
        with st.expander(tt["run_done_expander"], expanded=False):
            st.caption(f"`{last}`")
            if n_cached:
                st.markdown(f"⏭️ {tt['run_cache_reuse'].format(n=n_cached)}")
            for idx, stage, skipped, done, gate, arts in rows:
                if skipped and not gate:
                    continue  # compressed into the cache line above
                spec = STAGE_SPECS[stage]
                what = spec.what if lang == "English" else spec.zh_what
                secs = done.get("seconds", 0) if done else 0
                extra = f", {done.get('rows')} rows" if done and done.get("rows") else ""
                st.markdown(f"**{_circled(idx)} {what}** — `{secs}s`{extra}")
                for a in arts:
                    km = a.get("key_metrics") or {}
                    st.caption(f"📦 {a.get('path')}" + (f"  {km}" if km else ""))
                for g in gate:
                    st.caption(f"⚠️ decision raised: {g.get('kind')}")


def _render_failure(st, cfg, tt, lang) -> None:
    """FAILURE TURN: name the step, state nothing changed, offer Retry + details.
    No red ERROR wall (uses a warning, stack detail behind an expander)."""
    events = read_events(_state_dir(cfg) / "autopilot_journal.jsonl")
    last = events[-1].get("run_id") if events else None
    halted = next((e for e in events
                   if e.get("run_id") == last and e["type"] == "halt"), None)
    if not halted:
        return
    with st.chat_message("assistant"):
        st.warning(tt["fail_step"].format(stage=halted.get("stage")))
        if st.button(tt["fail_retry"], key="chat_retry"):
            st.session_state.ap_proc = _launch(cfg, "gated")
            st.rerun()
        with st.expander(tt["fail_details"]):
            st.code(f"{halted.get('reason')}\n{halted.get('detail')}")


def _uncertainty_index(card) -> int:
    """Index of the uncertainty/optimism caveat signal (always kept visible)."""
    sigs = card.get("signals", [])
    for i, s in enumerate(sigs):
        if str(s.get("field", "")).endswith("rmse") or "±" in s.get("text_en", ""):
            return i
    return len(sigs) - 1 if sigs else -1


def _render_active_card(st, card, cfg, tt, lang) -> None:
    """A pinned, unanswered decision card: verdict + always-visible uncertainty;
    other signals behind a 'why?' expander; actions morph the card on press."""
    en = lang == "English"
    with st.chat_message("assistant"):
        prio = card.get("priority", "P3")
        verdict = card["verdict_en"] if en else card["verdict_zh"]
        {"P1": st.error, "P2": st.warning}.get(prio, st.info)(
            f"**{prio} · {card.get('kind')}** — {verdict}")
        st.caption(card["verdict_zh"] if en else card["verdict_en"])

        sigs = card.get("signals", [])
        ui = _uncertainty_index(card)
        if 0 <= ui < len(sigs):
            s = sigs[ui]
            st.markdown(f"⚠️ {s['text_en'] if en else s['text_zh']}")
        others = [s for i, s in enumerate(sigs) if i != ui]
        if others:
            with st.expander(tt["why_expander"]):
                for s in others:
                    st.markdown(f"- {s['text_en'] if en else s['text_zh']}")
                    st.caption(f"{tt['card_grounded']}: `{s.get('artifact')}` [{s.get('field')}]")

        m = re.search(r"unit_(\d+)", str(card.get("evidence_link", "")))
        if m:
            uid = int(m.group(1))
            if st.button(tt["card_evidence_btn"].format(u=uid), key=f"chatev_{card['id']}"):
                st.session_state["app_mode"] = MODE_TEACH
                st.session_state["engine_select"] = uid
                st.session_state["stage"] = 4
                st.session_state["stage_unit"] = uid
                st.rerun()

        actions = card.get("actions", [])
        cols = st.columns(len(actions)) if actions else []
        for col, a in zip(cols, actions):
            label = (a["label_en"] if en else a["label_zh"]) + (
                " ✅" if a.get("safe_default") else "")
            if col.button(label, key=f"chatact_{card['id']}_{a['id']}"):
                _answer_card(cfg, card["id"], a["id"])
                # The card MORPHS to a resolved milestone — no praise echo bubble.
                st.session_state.setdefault("answered_cards", {})[card["id"]] = a["id"]
                st.session_state["chat"].append({
                    "role": "agent", "kind": "resolved_card",
                    "en": T["English"]["chat_resolved"].format(act=a["label_en"]),
                    "zh": T["中文"]["chat_resolved"].format(act=a["label_zh"]),
                })
                st.session_state.ap_proc = _launch(cfg, "gated")
                st.rerun()
            col.caption(a["consequence_en"] if en else a["consequence_zh"])


def _maybe_append_done(st, cfg) -> None:
    """Append the done-state summary bubble once per completed run."""
    events = read_events(_state_dir(cfg) / "autopilot_journal.jsonl")
    if not events:
        return
    last = events[-1].get("run_id")
    if st.session_state.get("summary_done_for") == last:
        return
    import pandas as pd

    preds = pd.read_csv(be.PRED_PATH)
    n, h = int(len(preds)), int((preds["risk_band"] == "high").sum())
    d = len(list((_state_dir(cfg) / "autopilot_inbox" / "pending").glob("*.json")))
    st.session_state["chat"].append({
        "role": "agent", "kind": "done",
        "en": T["English"]["done_banner"].format(n=n, h=h, d=d),
        "zh": T["中文"]["done_banner"].format(n=n, h=h, d=d),
    })
    st.session_state["summary_done_for"] = last


def _active_cards(st, cfg):
    """Unanswered pending cards (answered ones have morphed into history)."""
    answered = st.session_state.get("answered_cards", {})
    out = []
    if st.session_state.get("chat_run_active"):
        for pf in sorted((_state_dir(cfg) / "autopilot_inbox" / "pending").glob("*.json")):
            card = _read_json(pf)
            if card and card["id"] not in answered:
                out.append(card)
    return out


def _chat_mode(st, plt, pd, cfg, tt, lang, pv) -> None:
    # Persistent trust badge + capability opener (reporting voice, no persona).
    st.caption(f"🔒 **{tt['chat_badge']}**")
    st.markdown(f"#### {tt['chat_intro']}")
    st.info(tt["ap_naming"], icon="🤖")

    if st.sidebar.button(tt["chat_reset"], key="chat_reset_btn"):
        for k in ("chat", "awaiting_start", "chat_run_active", "ap_proc",
                  "summary_done_for", "answered_cards", "ap_skip"):
            st.session_state.pop(k, None)
    # Landing never shows an empty box: greeting + a grounded example.
    if not st.session_state.get("chat"):
        st.session_state["chat"] = [_greeting_msg(), _showcase_msg()]

    for msg in st.session_state["chat"]:
        _render_chat_msg(st, msg, tt, lang)

    proc = st.session_state.get("ap_proc")
    skip = st.session_state.pop("ap_skip", False)
    running = proc is not None and proc.poll() is None and not skip

    # ONE rewriting progress bubble + honest failure turn.
    if st.session_state.get("chat_run_active"):
        _render_progress_bubble(st, cfg, tt, lang, running)
        _render_failure(st, cfg, tt, lang)

    if proc is not None and not running:
        st.session_state.ap_proc = None
        _maybe_append_done(st, cfg)

    active = _active_cards(st, cfg)

    # Action row: a pending decision takes precedence over chips (decision focus).
    if active:
        st.caption(f"📌 {tt['decision_pending'].format(n=len(active))}")
        for card in active:
            _render_active_card(st, card, cfg, tt, lang)
    elif not running and st.session_state.get("awaiting_start"):
        c1, c2 = st.columns(2)
        if c1.button(tt["chat_btn_start"], key="chat_start"):
            st.session_state["awaiting_start"] = False
            st.session_state["chat_run_active"] = True
            st.session_state.ap_proc = _launch(cfg, "gated")
            st.rerun()
        if c2.button(tt["chat_btn_dry"], key="chat_dry"):
            st.session_state["awaiting_start"] = False
            st.session_state["chat_run_active"] = True
            st.session_state.ap_proc = _launch(cfg, "dry-run")
            st.rerun()
    elif not running:
        cols = st.columns(len(CHIPS))
        for col, (key, canon) in zip(cols, CHIPS):
            if col.button(tt[key], key=f"chip_{key}", width="stretch"):
                _handle_user_intent(st, cfg, tt, tt[key], canon)
                st.rerun()

    if running:  # anti-conversation-tax: jump straight to results
        if st.button(tt["skip_results"], key="chat_skip"):
            st.session_state["ap_skip"] = True
            st.rerun()

    prompt = st.chat_input(tt["chat_input_ph"])
    if prompt:
        _handle_user_intent(st, cfg, tt, prompt, running=running)
        st.rerun()

    if running:
        time.sleep(1.0)
        st.rerun()


# =============================================================================
# Dispatcher
# =============================================================================
def _run() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Condition Monitoring · 状态监测", layout="wide")

    cfg = _load_cfg()
    pv = _pipeline_values(cfg)

    lang = st.sidebar.radio("Language / 语言", ["English", "中文"], index=0)
    tt = T[lang]
    st.sidebar.header(tt["sb_controls"])

    st.title(tt["app_title"])
    st.warning(tt["limitations"], icon="⚠️")

    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = MODE_CHAT
    mode = st.radio(
        tt["mode_label"], [MODE_CHAT, MODE_DASH, MODE_TEACH],
        key="app_mode", horizontal=True,
    )

    if mode == MODE_TEACH:
        _teaching_mode(st, plt, pd, cfg, tt, lang, pv)
    elif mode == MODE_DASH:
        _autopilot_page(st, plt, pd, cfg, tt, lang, pv)
    else:
        _chat_mode(st, plt, pd, cfg, tt, lang, pv)


if __name__ == "__main__":
    _run()
