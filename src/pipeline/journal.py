"""Append-only NDJSON event journal — the live visualization contract.

Every pipeline run streams one JSON object per line to
``reports/pipeline_journal.jsonl``; the app (Phase D) tails the file (~1s poll,
no websockets) to render progress as it happens, and the events replay
deterministically. Each event carries ``ts``, ``run_id``, ``seq``, and ``type``.

Core event types (from the plan's "Live visualization contract"):

* ``run_started``  {stages}
* ``stage_started``{stage, what, why}
* ``stage_progress``{stage, message}
* ``artifact``     {stage, path, key_metrics}
* ``stage_done``   {stage, seconds, rows, skipped}
* ``run_done``     {stages_run, stages_skipped, seconds}

The schema is intentionally open (extra ``type`` values + fields are allowed) so
Phase C can add gate events — ``gate_raised`` / ``gate_resolved`` / ``halt`` —
without changing this writer or breaking existing readers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

#: Core event types a Phase-A journal is guaranteed to contain.
CORE_EVENT_TYPES = frozenset(
    {
        "run_started",
        "stage_started",
        "stage_progress",
        "artifact",
        "stage_done",
        "run_done",
    }
)

#: Reserved for Phase C (autopilot HITL gates); declared here so readers can
#: whitelist them today.
GATE_EVENT_TYPES = frozenset({"gate_raised", "gate_resolved", "halt"})


class Journal:
    """Append-only NDJSON writer for one pipeline run."""

    def __init__(self, path: Path | str, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, event_type: str, **fields) -> dict:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "seq": self._seq,
            "type": event_type,
            **fields,
        }
        self._seq += 1
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    # --- core events ----------------------------------------------------------
    def run_started(self, stages: list[str]) -> dict:
        return self._emit("run_started", stages=list(stages))

    def stage_started(self, stage: str, what: str, why: str) -> dict:
        return self._emit("stage_started", stage=stage, what=what, why=why)

    def stage_progress(self, stage: str, message: str) -> dict:
        return self._emit("stage_progress", stage=stage, message=message)

    def artifact(self, stage: str, path: str, key_metrics: dict | None = None) -> dict:
        return self._emit(
            "artifact", stage=stage, path=str(path), key_metrics=key_metrics or {}
        )

    def stage_done(
        self, stage: str, seconds: float, rows: int | None, skipped: bool = False
    ) -> dict:
        return self._emit(
            "stage_done",
            stage=stage,
            seconds=round(float(seconds), 4),
            rows=rows,
            skipped=skipped,
        )

    def run_done(
        self, stages_run: int, stages_skipped: int, seconds: float
    ) -> dict:
        return self._emit(
            "run_done",
            stages_run=stages_run,
            stages_skipped=stages_skipped,
            seconds=round(float(seconds), 4),
        )


def read_events(path: Path | str) -> list[dict]:
    """Read all events from a journal file (skips blank/corrupt lines)."""
    path = Path(path)
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def events_for_run(path: Path | str, run_id: str) -> list[dict]:
    """All events belonging to a single run, in emission order."""
    return [e for e in read_events(path) if e.get("run_id") == run_id]
