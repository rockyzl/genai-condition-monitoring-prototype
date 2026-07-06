"""App-level AppTest checks for the v2 Streamlit app (autopilot + teaching).

Hermetic: a fixture journal + a fixture pending decision card are written into a
tmp dir the app is pointed at via ``CM_APP_REPORTS_DIR``; the agent subprocess is
disabled via ``CM_APP_NO_SUBPROCESS``. No live pipeline run is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

import src.app.streamlit_app as app  # noqa: E402
from src.pipeline.specs import STAGE_ORDER  # noqa: E402

APP_FILE = str(ROOT / "src" / "app" / "streamlit_app.py")

FIXTURE_CARD = {
    "id": "high_risk_triage-testcard",
    "kind": "high_risk_triage",
    "priority": "P1",
    "verdict_en": "3 engines need inspection first — units 39, 57, 81.",
    "verdict_zh": "有 3 台发动机需要优先检查——39、57、81 号。",
    "signals": [
        {
            "text_en": "Units 39, 57, 81 sit at predicted RUL 6.6 / 8.0 / 9.2 cycles.",
            "text_zh": "39、57、81 号机组预测剩余寿命 6.6 / 8.0 / 9.2 个周期。",
            "artifact": "data/processed/test_predictions.csv",
            "field": "count:pred_rul<=10",
        },
        {
            "text_en": "18 engines are high-risk overall.",
            "text_zh": "整机队共有 18 台高风险。",
            "artifact": "data/processed/test_predictions.csv",
            "field": "count:risk_band=high",
        },
        {
            "text_en": "Point estimates, typically off by about ±18 cycles.",
            "text_zh": "点估计，误差大约 ±18 个周期。",
            "artifact": "reports/metrics_model.json",
            "field": "metrics_vs_uncapped_truth.rmse",
        },
    ],
    "evidence_link": "data/processed/evidence/unit_39.json",
    "actions": [
        {
            "id": "schedule_inspection",
            "label_en": "Schedule inspection for units 39, 57, 81",
            "label_zh": "为 39、57、81 号安排检查",
            "consequence_en": "Draft work orders only, never commands.",
            "consequence_zh": "只是工单草稿，绝不是执行指令。",
            "safe_default": True,
        },
        {
            "id": "defer",
            "label_en": "Defer to next cycle",
            "label_zh": "推迟到下个周期",
            "consequence_en": "Units stay flagged; no work order drafted.",
            "consequence_zh": "机组仍保持标记；不生成工单。",
        },
    ],
}


def _write_fixtures(reports_dir: Path, with_card: bool = True) -> None:
    inbox = reports_dir / "autopilot_inbox"
    (inbox / "pending").mkdir(parents=True, exist_ok=True)
    (inbox / "answered").mkdir(parents=True, exist_ok=True)
    # fixture journal: one run that reaches s07 and raises the triage card
    rid = "auto_testrun"
    events = [
        {"ts": "t", "run_id": rid, "seq": 0, "type": "run_started", "stages": STAGE_ORDER},
        {"ts": "t", "run_id": rid, "seq": 1, "type": "stage_started",
         "stage": "s07_predict", "what": "score every test unit", "why": "reusable tool"},
        {"ts": "t", "run_id": rid, "seq": 2, "type": "stage_progress",
         "stage": "s07_predict", "message": "gate high_risk_triage: card"},
        {"ts": "t", "run_id": rid, "seq": 3, "type": "gate_raised",
         "card_id": FIXTURE_CARD["id"], "kind": "high_risk_triage",
         "stage": "s07_predict", "payload_summary": {}},
        {"ts": "t", "run_id": rid, "seq": 4, "type": "stage_done",
         "stage": "s07_predict", "seconds": 0.5, "rows": 100, "skipped": False},
        {"ts": "t", "run_id": rid, "seq": 5, "type": "run_done",
         "stages_run": 7, "stages_skipped": 3, "seconds": 1.2},
    ]
    (reports_dir / "autopilot_journal.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )
    if with_card:
        (inbox / "pending" / f"{FIXTURE_CARD['id']}.json").write_text(
            json.dumps(FIXTURE_CARD, ensure_ascii=False)
        )


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("CM_APP_REPORTS_DIR", str(reports))
    monkeypatch.setenv("CM_APP_NO_SUBPROCESS", "1")
    return reports


def _alltext(at) -> str:
    out = []
    for lst in (at.error, at.warning, at.success, at.info, at.markdown,
                at.caption, at.subheader, at.title):
        for e in lst:
            v = getattr(e, "value", None)
            if v:
                out.append(v)
    return "\n".join(out)


def _click(at, key):
    for b in at.button:
        if getattr(b, "key", None) == key:
            b.click()
            return at.run()
    raise AssertionError(f"button key {key!r} not in {[getattr(b,'key',None) for b in at.button]}")


def _radio_with(at, option):
    """Return the radio whose options include ``option`` (robust to widget order)."""
    for r in at.radio:
        if option in list(r.options):
            return r
    raise AssertionError(f"no radio has option {option!r}")


# =============================================================================
# 1. Mode switch renders both modes
# =============================================================================
def test_mode_switch_renders_both_modes(hermetic):
    _write_fixtures(hermetic, with_card=False)
    at = AppTest.from_file(APP_FILE, default_timeout=90).run()
    assert not at.exception, at.exception
    # default = autopilot: inbox header present
    assert "Decision Inbox" in _alltext(at)
    # switch to teaching mode via the mode radio
    _radio_with(at, app.MODE_TEACH).set_value(app.MODE_TEACH).run()
    assert not at.exception, at.exception
    assert "① Step 1 — Pick an engine" in _alltext(at)


# =============================================================================
# 2. Inbox renders a fixture card with actions (EN + 中文, no crash)
# =============================================================================
@pytest.mark.parametrize("lang,verdict_key,action_label", [
    ("English", "verdict_en", "Schedule inspection for units 39, 57, 81"),
    ("中文", "verdict_zh", "为 39、57、81 号安排检查"),
])
def test_inbox_renders_fixture_card(hermetic, lang, verdict_key, action_label):
    _write_fixtures(hermetic, with_card=True)
    at = AppTest.from_file(APP_FILE, default_timeout=90).run()
    _radio_with(at, "English").set_value(lang).run()  # language
    assert not at.exception, at.exception
    text = _alltext(at)
    assert FIXTURE_CARD[verdict_key] in text          # verdict headline
    assert "18" in text                                # done-banner: flagged 18 high-risk
    labels = [getattr(b, "label", "") for b in at.button]
    assert any(action_label in lbl for lbl in labels)  # action button present
    # a signal (incl. the uncertainty caveat) rendered
    assert "high-risk" in text or "高风险" in text


# =============================================================================
# 3. Pressing an action writes the answered file (schema: card_id, action_id...)
# =============================================================================
def test_action_press_writes_answered_file(hermetic):
    _write_fixtures(hermetic, with_card=True)
    at = AppTest.from_file(APP_FILE, default_timeout=90).run()
    assert not at.exception, at.exception
    key = f"act_{FIXTURE_CARD['id']}_schedule_inspection"
    _click(at, key)
    assert not at.exception, at.exception
    answered = hermetic / "autopilot_inbox" / "answered" / f"{FIXTURE_CARD['id']}.json"
    assert answered.exists(), "answered card file was not written"
    data = json.loads(answered.read_text())
    assert data["card_id"] == FIXTURE_CARD["id"]
    assert data["action_id"] == "schedule_inspection"
    assert data["action"] == "schedule_inspection"  # what the supervisor reads
    assert data["actor"] == "ui"
