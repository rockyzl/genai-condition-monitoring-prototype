# Demo media

`demo.mp4` / `demo.webm` — the ~68-second flagship demo of the **Agent Chat**
mode, referenced from the top-level `README.md`. Recorded at 1280×800.

## Regenerate

```bash
.venv/bin/pip install playwright        # dev-only tool; NOT in requirements.txt
.venv/bin/playwright install chromium
.venv/bin/python scripts/record_demo.py
```

`scripts/record_demo.py` is a reproducible asset. It:

1. **Backs up** `reports/autopilot_inbox`, `autopilot_journal.jsonl`,
   `autopilot_state.json`, then **resets the inbox** so a fresh gated run raises
   the triage decision card cleanly.
2. Starts the app on port 8600, waits for health.
3. Drives the 6-beat storyboard through headless Chromium with video capture.
4. Saves `demo.webm`, converts to `demo.mp4` (h264) if `ffmpeg` is on `PATH`.
5. **Restores** the pre-run `reports/` state.

## Storyboard (6 beats)

1. **Landing** — chat mode: trust badge (`Deterministic · Grounded · No LLM`),
   greeting, grounded showcase, suggestion chips.
2. **Scan fleet** → the plan-preview bubble → **▶ Start**.
3. **Watch it work** — the single progress bubble → `✓ Done · view steps`
   (expanded to show per-node numbers + citations).
4. **Decision card** (pinned) — "3 engines need inspection first — units 81, 34,
   35" → `why?` (grounded signals) → action → `✓ Confirmed`.
5. **Done-summary** + a free-text `diagnose unit 81` → the grounded, cited answer.
6. **Close** on the Autopilot dashboard fleet table (risk-coloured, most-urgent
   first).

## Notes

- `ffmpeg` is required only for the `.mp4`; the `.webm` is produced regardless.
- Both outputs are well under 25 MB (webm ≈ 2 MB, mp4 ≈ 1 MB).
- Playwright + Chromium are dev-only and intentionally kept out of
  `requirements.txt`.
