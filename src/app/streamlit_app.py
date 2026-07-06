"""Streamlit demo for the GenAI-assisted condition-monitoring prototype.

A button-driven, bilingual (English default / 中文) guided tour: the page reads
top-to-bottom as five numbered steps, and each step's output is gated behind a
button so every press maps to one real stage of the pipeline

    ① load sensor history → ② predict remaining life → ③ show the evidence
    → ④ generate a cited diagnostic report → ⑤ remember the limits

Each step carries its canonical pipeline description (always visible once the
step exists) plus a two-line 功能 Function / 目的 Purpose gloss. Progress is
tracked in st.session_state, keyed by the selected engine, and resets cleanly
when the engine changes. Conclusion-first: step ② leads with a colour-coded
one-sentence verdict, a remaining-life bar, and three key numbers.

The Chinese diagnostic summary is composed deterministically here in the app
from the evidence fields — src/rag/assistant.py and its contract are untouched;
knowledge-base quotes stay in their original English under a Chinese caption.

Run:  .venv/bin/streamlit run src/app/streamlit_app.py

The UI body is guarded under ``if __name__ == "__main__"`` so the module imports
cleanly (no Streamlit side effects) for testing; Streamlit executes the file
with ``__name__ == "__main__"``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- make `src` importable when run via `streamlit run` -------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import build_evidence as be  # noqa: E402
from src.rag.assistant import diagnose  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402

RUL_CAP = 125  # piecewise-linear RUL ceiling used to train the model

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
        "title": "Predicting When an Engine Needs Maintenance",
        "what_is_this": (
            "A 5-step guided tour: press each button to run the next stage of the "
            "pipeline. It reads an aircraft engine's sensors, estimates how much "
            "longer it can safely run, and writes a maintenance note in which "
            "every cause and next step is quoted from a cited reference — it never "
            "invents a diagnosis."
        ),
        "limitations": (
            "Independent R&D prototype on public NASA C-MAPSS turbofan data. Not "
            "production-validated, not affiliated with any equipment "
            "manufacturer, and not a source of safety-critical decisions. Every "
            "output is advisory and requires review by a qualified human."
        ),
        "lbl_function": "Function:",
        "lbl_purpose": "Purpose:",
        "sb_controls": "Controls",
        "sb_cycles_label": "How many recent cycles to plot",
        "sb_kb_empty": "Knowledge base is empty — diagnostic guidance unavailable.",
        # ---- Step 1 ----
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
        # ---- Step 2 ----
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
        # ---- Step 3 ----
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
        # ---- Step 4 ----
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
        # ---- Step 5 ----
        "step5_title": "⑤ Step 5 — Remember the limits",
        "step5_desc": "A qualified human always makes the final call.",
        "step5_func": "Surfaces the model's uncertainty and forces human review before any action.",
        "step5_purpose": (
            "This is decision support, not a decision — it focuses attention, it "
            "never closes a safety-critical loop."
        ),
        # ---- Buttons ----
        "btn_load": "① Load sensor history",
        "btn_predict": "② Predict remaining life",
        "btn_evidence": "③ Show the evidence",
        "btn_report": "④ Generate cited diagnostic report",
        "btn_runall": "▶ Run all steps",
        # ---- Conclusion / metrics ----
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
        "m_risk_label": "Risk",
        "m_lastcycle_label": "Flight cycles flown so far",
        "risk_high": "schedule inspection soon — near end-of-life",
        "risk_medium": "degrading — monitor closely",
        "risk_low": "healthy — keep monitoring",
        # ---- Evidence (charts + glossary) ----
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
        # ---- Report ----
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
        # ---- Step 5 footer content ----
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
            "pipeline (Phases 2-3) first."
        ),
    },
    "中文": {
        "title": "预测发动机什么时候需要维护",
        "what_is_this": (
            "一个 5 步向导：每按一个按钮，就跑管线的下一步。它读取一台飞机发动机的"
            "传感器，估计它还能安全再飞多久，并写一段维护说明——里面每个说法和每一步"
            "建议都标了知识库出处，绝不瞎编。"
        ),
        "limitations": (
            "独立研究原型，使用 NASA 公开模拟数据，与任何设备制造商无关。"
        ),
        "lbl_function": "功能：",
        "lbl_purpose": "目的：",
        "sb_controls": "控制项",
        "sb_cycles_label": "画最近多少个周期",
        "sb_kb_empty": "知识库是空的——暂时给不了诊断建议。",
        # ---- Step 1 ----
        "step1_title": "① 第 1 步——选一台发动机",
        "step1_desc": "看传感器——取这台发动机最近几个飞行周期的读数（温度、压力、转速）。",
        "step1_func": "从 NASA 测试机队里选一台发动机，载入它的原始传感器历史。",
        "step1_purpose": "后面所有步骤都是针对这一台发动机的——你来决定看哪一台。",
        "step1_engine_label": "发动机（测试单元）编号",
        # ---- Step 2 ----
        "step2_title": "② 第 2 步——看结论",
        "step2_desc": (
            "做预测——一个简单的机器学习模型估计这台发动机的剩余可用寿命：还能再飞"
            "多少个周期才需要维护。"
        ),
        "step2_func": "模型根据这台发动机的传感器历史，算出大概还能飞多少个周期。",
        "step2_purpose": "把 21 路传感器数字变成一个能做决策的数字。",
        # ---- Step 3 ----
        "step3_title": "③ 第 3 步——看证据",
        "step3_desc": "看传感器——取这台发动机最近几个飞行周期的读数（温度、压力、转速）。",
        "step3_func": "把最有信息量的几个传感器最近的走势画出来，并标出每个到底测的是什么。",
        "step3_purpose": "让人能拿预测去核对原始信号——发动机是真在退化，还是传感器噪声？",
        # ---- Step 4 ----
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
        # ---- Step 5 ----
        "step5_title": "⑤ 第 5 步——记住边界",
        "step5_desc": "最终一定由合格的人来拍板。",
        "step5_func": "把模型的不确定性摆出来，任何动作前都要求人工复核。",
        "step5_purpose": "这是决策支持，不是决策本身——它帮你集中注意力，绝不替你做安全关键的决定。",
        # ---- Buttons ----
        "btn_load": "① 载入传感器历史",
        "btn_predict": "② 预测剩余寿命",
        "btn_evidence": "③ 看证据",
        "btn_report": "④ 生成带引用的诊断报告",
        "btn_runall": "▶ 一键跑完",
        # ---- Conclusion / metrics ----
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
        "m_risk_label": "风险等级",
        "m_lastcycle_label": "已经飞了多少周期",
        "risk_high": "尽快安排检查——快到寿命尽头了",
        "risk_medium": "在退化，盯紧点",
        "risk_low": "健康，继续监测",
        # ---- Evidence ----
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
        # ---- Report ----
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
        # ---- Step 5 footer content ----
        "rep_uncertainty_label": "不确定性说明：",
        "rep_uncertainty": (
            "这个剩余寿命只是一个粗略的点估计，没有置信区间。模型训练时把寿命目标压到了"
            "上限（一般是 125 个周期），所以健康发动机的预测会被压到接近上限，不能当成"
            "精确的周期数。越接近寿命尽头，预测误差越大。这个结果只能用来帮忙排查优先级，"
            "不能当成权威的失效时间，必须人工复核。"
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
        "pred_missing": "在 {path} 找不到预测文件。请先跑数据/模型流水线（第 2-3 阶段）。",
    },
}


def _short_label(sensor_col: str) -> str:
    meta = SENSOR_META.get(sensor_col)
    return meta[0] if meta else sensor_col


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


def _step_header(st, tt: dict, title_key: str, desc, func_key: str, purpose_key: str):
    """Render a numbered step header + its always-visible canonical description
    + the 功能/目的 (Function/Purpose) two-liner."""
    st.subheader(tt[title_key])
    for d in desc if isinstance(desc, list) else [desc]:
        st.caption(d)
    st.markdown(
        f"**{tt['lbl_function']}** {tt[func_key]}  \n"
        f"**{tt['lbl_purpose']}** {tt[purpose_key]}"
    )


def _run() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    # set_page_config MUST be the first Streamlit call.
    st.set_page_config(page_title="Condition Monitoring · 状态监测", layout="wide")

    @st.cache_resource
    def get_retriever() -> Retriever:
        return Retriever(be.KB_DIR)

    @st.cache_data
    def get_predictions() -> "pd.DataFrame":
        return pd.read_csv(be.PRED_PATH)

    @st.cache_data
    def get_raw() -> "pd.DataFrame":
        return be.load_raw_test()

    # --- Language + secondary controls (sidebar) ---------------------------
    lang = st.sidebar.radio("Language / 语言", ["English", "中文"], index=0)
    tt = T[lang]
    st.sidebar.header(tt["sb_controls"])
    window = st.sidebar.slider(tt["sb_cycles_label"], 10, 60, be.LAST_WINDOW)

    # --- Header (always visible) -------------------------------------------
    st.title(tt["title"])
    st.markdown(f"#### {tt['what_is_this']}")
    st.warning(tt["limitations"], icon="⚠️")

    if not be.PRED_PATH.exists():
        st.error(tt["pred_missing"].format(path=be.PRED_PATH))
        st.stop()

    preds = get_predictions()
    raw = get_raw()
    retriever = get_retriever()
    if len(retriever) == 0:
        st.sidebar.info(tt["sb_kb_empty"])

    # --- Session state: wizard progress, keyed by engine -------------------
    if "stage" not in st.session_state:
        st.session_state.stage = 0
    if "stage_unit" not in st.session_state:
        st.session_state.stage_unit = None

    # ======================================================================
    # STEP ① — Pick an engine
    # ======================================================================
    st.divider()
    _step_header(st, tt, "step1_title", tt["step1_desc"], "step1_func", "step1_purpose")
    unit_ids = sorted(preds["unit_id"].astype(int).tolist())
    unit_id = st.selectbox(tt["step1_engine_label"], unit_ids, key="engine_select")

    # Reset progress when the engine changes (state keyed by unit).
    if st.session_state.stage_unit != unit_id:
        st.session_state.stage = 0
        st.session_state.stage_unit = unit_id

    b1, b_all = st.columns(2)
    if b1.button(tt["btn_load"], key="btn_load", use_container_width=True):
        st.session_state.stage = max(st.session_state.stage, 1)
    if b_all.button(tt["btn_runall"], key="btn_runall", use_container_width=True):
        st.session_state.stage = 4

    prow = preds[preds["unit_id"] == int(unit_id)].iloc[0]
    band = str(prow["risk_band"]).lower()
    rul = round(float(prow["pred_rul"]))
    last_cycle = int(prow["last_cycle"])

    # ======================================================================
    # STEP ② — Read the conclusion (conclusion-first)
    # ======================================================================
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

            frac = min(max(rul / RUL_CAP, 0.0), 1.0)
            st.progress(frac, text=tt["health_text"].format(r=rul, cap=RUL_CAP))
            st.caption(tt["health_caption"].format(cap=RUL_CAP))

            c1, c2, c3 = st.columns(3)
            c1.metric(tt["m_rul_label"], f"{rul} {tt['cycles_unit']}")
            c2.metric(tt["m_risk_label"], RISK_WORD[lang][band], help=tt[f"risk_{band}"])
            c3.metric(tt["m_lastcycle_label"], last_cycle)
            st.caption(tt["m_rul_gloss"])

    # Evidence is loaded lazily once the user has advanced past prediction.
    evidence = _load_evidence(int(unit_id)) if st.session_state.stage >= 2 else None

    # ======================================================================
    # STEP ③ — See the evidence
    # ======================================================================
    if st.session_state.stage >= 2:
        st.divider()
        _step_header(st, tt, "step3_title", tt["step3_desc"], "step3_func", "step3_purpose")
        if st.button(tt["btn_evidence"], key="btn_evidence"):
            st.session_state.stage = max(st.session_state.stage, 3)
        if st.session_state.stage >= 3:
            unit_raw = (
                raw[raw["unit"] == int(unit_id)].sort_values("cycle").tail(window)
            )
            if evidence and evidence.get("sensor_summary"):
                sensors = list(evidence["sensor_summary"].keys())
                trends = {
                    k: v.get("trend") for k, v in evidence["sensor_summary"].items()
                }
            else:
                sensors = be.rank_sensors_from_importances(
                    be.load_feature_importances(), be.TOP_K_SENSORS
                )
                trends = {}
            sensors = [s for s in sensors if s in unit_raw.columns][:6]

            if sensors and not unit_raw.empty:
                ncol = 2
                nrow = (len(sensors) + ncol - 1) // ncol
                fig, axes = plt.subplots(
                    nrow, ncol, figsize=(10, 2.3 * nrow), squeeze=False
                )
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
                    meaning_idx = 1 if lang == "English" else 2
                    table = pd.DataFrame(
                        [
                            {
                                tt["table_h_sensor"]: s,
                                tt["table_h_symbol"]: SENSOR_META.get(s, ("?", "", ""))[0],
                                tt["table_h_meaning"]: SENSOR_META.get(
                                    s, ("", "—", "—")
                                )[meaning_idx],
                            }
                            for s in sensors
                        ]
                    )
                    st.table(table)
                    st.caption(tt["table_provenance"])
            else:
                st.info(tt["no_sensor"])

    # ======================================================================
    # STEP ④ — Read the guidance (retrieval + cited report)
    # ======================================================================
    if st.session_state.stage >= 3:
        st.divider()
        _step_header(
            st,
            tt,
            "step4_title",
            [tt["step4_desc_retrieval"], tt["step4_desc_report"]],
            "step4_func",
            "step4_purpose",
        )
        if st.button(tt["btn_report"], key="btn_report"):
            st.session_state.stage = max(st.session_state.stage, 4)
        if st.session_state.stage >= 4:
            if not evidence:
                st.info(tt["no_evidence"])
            else:
                report = diagnose(evidence, retriever)

                st.markdown(f"**{tt['rep_summary_h']}**")
                summary = (
                    report["summary"]
                    if lang == "English"
                    else compose_zh_summary(evidence)
                )
                st.write(summary)

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
                        st.markdown(
                            f"- `{cite['source_file']}` · **{cite['section']}**"
                        )
                st.caption(tt["rep_sources_caption"])

    # ======================================================================
    # STEP ⑤ — Remember the limits (persistent footer, always visible)
    # ======================================================================
    st.divider()
    _step_header(st, tt, "step5_title", tt["step5_desc"], "step5_func", "step5_purpose")
    st.warning(f"**{tt['step5_desc']}** {tt['rep_humanreview']}", icon="🧑‍🔧")
    uncertainty = be.UNCERTAINTY_NOTE if lang == "English" else tt["rep_uncertainty"]
    st.info(f"**{tt['rep_uncertainty_label']}** {uncertainty}", icon="ℹ️")
    st.caption(tt["safety_note"])


if __name__ == "__main__":
    _run()
