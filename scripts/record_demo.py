"""Record the 60-90s flagship demo of the Agent Chat mode (reproducible asset).

Drives the approved storyboard through a headless Chromium with video capture:

  1. Landing — chat mode, trust badge + greeting + chips.
  2. "Scan fleet" → plan preview → Start.
  3. Watch-it-work progress bubble → "✓ Done · view steps" (expanded to show
     per-node numbers + citations).
  4. Pinned decision card (units 81/34/35) → "why?" → action → "✓ Confirmed".
  5. Done-summary + a free-text "diagnose unit 81" → grounded cited answer.
  6. Close on the summary + badge (+ a beat of the Dashboard fleet table).

The script starts the app on a free port, resets the autopilot inbox so a fresh
gated run raises the triage card cleanly, records the walk, then RESTORES the
pre-run reports state. Output: docs/media/demo.webm (+ demo.mp4 if ffmpeg).

Run:  .venv/bin/python scripts/record_demo.py
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app" / "streamlit_app.py"
PORT = 8600
BASE = f"http://127.0.0.1:{PORT}"
MEDIA = ROOT / "docs" / "media"
REPORTS = ROOT / "reports"
VIEWPORT = {"width": 1280, "height": 800}

# reports/ state that a clean recording needs reset + restored afterwards.
STATE_PATHS = ["autopilot_inbox", "autopilot_journal.jsonl", "autopilot_state.json"]


# --------------------------------------------------------------------------- #
# reports/ state backup + reset + restore
# --------------------------------------------------------------------------- #
def backup_state(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    for name in STATE_PATHS:
        src = REPORTS / name
        if src.exists():
            dst = tmp / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def reset_inbox() -> None:
    """Empty the inbox so a fresh gated run raises the (unanswered) triage card."""
    inbox = REPORTS / "autopilot_inbox"
    if inbox.exists():
        shutil.rmtree(inbox)
    (inbox / "pending").mkdir(parents=True, exist_ok=True)
    (inbox / "answered").mkdir(parents=True, exist_ok=True)
    # start the journal fresh so the transcript shows only the demo run
    jp = REPORTS / "autopilot_journal.jsonl"
    if jp.exists():
        jp.unlink()


def restore_state(tmp: Path) -> None:
    for name in STATE_PATHS:
        live = REPORTS / name
        with contextlib.suppress(FileNotFoundError):
            if live.is_dir():
                shutil.rmtree(live)
            else:
                live.unlink()
    for name in STATE_PATHS:
        src = tmp / name
        if src.exists():
            dst = REPORTS / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# app lifecycle
# --------------------------------------------------------------------------- #
def start_app() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.headless", "true", "--server.port", str(PORT),
         "--server.runOnSave", "false", "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,  # own process group so we can reap children
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{BASE}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("app did not become healthy on time")


def stop_app(proc: subprocess.Popen) -> None:
    # Streamlit spawns children; kill the whole process group, then wait.
    with contextlib.suppress(Exception):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    with contextlib.suppress(Exception):
        proc.wait(timeout=10)
    with contextlib.suppress(Exception):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


# --------------------------------------------------------------------------- #
# storyboard driver
# --------------------------------------------------------------------------- #
def _btn(page, name: str):
    return page.get_by_role("button", name=re.compile(name, re.I)).first


def _text(page, pattern: str):
    return page.get_by_text(re.compile(pattern, re.I)).first


def _show(page, locator, pause: int) -> None:
    with contextlib.suppress(Exception):
        locator.scroll_into_view_if_needed(timeout=5000)
    page.wait_for_timeout(pause)


def _expand(page, label: str) -> None:
    """Toggle a Streamlit expander (a <details><summary> element) by its label."""
    summary = page.locator(f'details:has-text("{label}") summary').first
    with contextlib.suppress(Exception):
        summary.scroll_into_view_if_needed(timeout=5000)
        summary.click()


def drive(page) -> None:
    page.goto(BASE, wait_until="load")
    # Streamlit boots asynchronously; wait for the greeting/badge.
    _text(page, "scan the fleet").wait_for(timeout=45000)
    page.wait_for_timeout(1000)
    _show(page, _text(page, "No LLM"), 7000)                       # 1) landing

    _btn(page, "Scan fleet").click()                               # 2) intent
    _text(page, "Run it").wait_for(timeout=15000)
    _show(page, _text(page, "load 100 engines"), 4000)
    _btn(page, "Start").click()                                    #    start

    _text(page, "view steps").wait_for(timeout=45000)              # 3) watch-it-work
    page.wait_for_timeout(1500)
    _expand(page, "view steps")                                    #    expand steps
    _show(page, _text(page, "view steps"), 10000)

    _text(page, "need inspection first").wait_for(timeout=45000)   # 4) decision card
    _show(page, _text(page, "need inspection first"), 3500)
    _expand(page, "why?")                                          #    reveal signals
    page.wait_for_timeout(5000)
    _btn(page, "Schedule inspection").click()                      #    take the action
    _text(page, "Confirmed").wait_for(timeout=20000)
    _show(page, _text(page, "Confirmed"), 8000)

    _show(page, _text(page, "scored 100 engines"), 3500)           # 5) done + free-text
    box = page.get_by_placeholder(re.compile("Type a request", re.I))
    box.click()
    box.fill("diagnose unit 81")
    box.press("Enter")
    _text(page, "Asset 81").wait_for(timeout=30000)
    _show(page, _text(page, "Asset 81"), 12000)                     #    grounded answer

    # 6) closing capstone: switch to the Autopilot dashboard fleet table
    with contextlib.suppress(Exception):
        page.get_by_text(re.compile("Autopilot dashboard")).first.click()
        _text(page, "Fleet view").wait_for(timeout=15000)
        page.wait_for_timeout(1200)
        _show(page, _text(page, "Fleet view"), 8000)


def record() -> Path:
    from playwright.sync_api import sync_playwright

    MEDIA.mkdir(parents=True, exist_ok=True)
    raw = MEDIA / "_raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, record_video_dir=str(raw))
        page = ctx.new_page()
        try:
            drive(page)
        finally:
            ctx.close()  # flush the video
            browser.close()
    webm = next(raw.glob("*.webm"))
    out = MEDIA / "demo.webm"
    if out.exists():
        out.unlink()
    shutil.move(str(webm), str(out))
    shutil.rmtree(raw, ignore_errors=True)
    return out


def to_mp4(webm: Path) -> Path | None:
    if not shutil.which("ffmpeg"):
        return None
    mp4 = webm.with_suffix(".mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-c:v", "libx264", "-pix_fmt",
         "yuv420p", "-movflags", "+faststart", "-loglevel", "error", str(mp4)],
        check=True,
    )
    return mp4


def probe(path: Path) -> str:
    if not shutil.which("ffprobe"):
        return "?"
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    return r.stdout.strip() or "?"


def main() -> int:
    tmp = ROOT / ".demo_state_backup"
    backup_state(tmp)
    app = None
    try:
        reset_inbox()
        app = start_app()
        webm = record()
    finally:
        if app is not None:
            stop_app(app)
        restore_state(tmp)
        shutil.rmtree(tmp, ignore_errors=True)

    mp4 = to_mp4(webm)
    dur = probe(webm)
    size_mb = webm.stat().st_size / 1e6
    print(f"[demo] {webm}  ({size_mb:.1f} MB, {dur}s)")
    if mp4:
        print(f"[demo] {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
