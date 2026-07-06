"""Execution trace — the audit record for one agent run (query or autopilot).

Every tool call and every gate outcome is appended to an in-memory trace and
then written to ``reports/agent_trace_<run_id>.json``. The trace is the backbone
of two guarantees the plan calls non-negotiable:

* **Grounding.** An answer or card claim carries the ``source_seq`` of the tool
  call it came from plus the ``field`` and ``value`` it read. :meth:`Trace.output_of`
  lets a checker re-read that call's output and confirm the value is really there
  — so no claim can float free of a tool output.
* **Auditability + anti-silent-weakening.** The run's ``thresholds_hash`` (a
  digest of the gate configuration) is recorded, so a later run that quietly
  loosened a gate produces a different hash — a meta-test can catch it.

The file stores digests + compact previews (not full outputs) to stay small;
the live in-memory :class:`Trace` retains full outputs for grounding checks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.registry import ToolResult, digest


@dataclass
class ToolCallRecord:
    seq: int
    tool: str
    args: dict
    status: str  # "ok" | "error"
    output_digest: str
    output_preview: Any = None
    error: str | None = None
    seconds: float = 0.0


@dataclass
class GateRecord:
    stage: str
    gate: str
    disposition: str  # PASS | CARD | HALT | SKIP
    card_id: str | None = None
    detail: str = ""


@dataclass
class Claim:
    """One factual claim in an answer/card, bound to the tool output it came from."""

    text_en: str
    source_seq: int
    tool: str
    field: str
    value: Any


@dataclass
class Trace:
    """Append-only, in-memory audit record for one agent run."""

    run_id: str
    kind: str  # "query" | "autopilot"
    planner: str | None = None
    autonomy: str | None = None
    thresholds_hash: str | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
    )
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    gates: list[GateRecord] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    answer: dict | None = None
    _outputs: dict[int, Any] = field(default_factory=dict, repr=False)

    # --- recording -----------------------------------------------------------
    def record_tool(self, result: ToolResult) -> int:
        """Append a tool result; return its trace seq (referenced by claims)."""
        seq = len(self.tool_calls)
        self.tool_calls.append(
            ToolCallRecord(
                seq=seq,
                tool=result.tool,
                args=result.args,
                status="ok" if result.ok else "error",
                output_digest=result.digest,
                output_preview=result.preview() if result.ok else None,
                error=result.error,
                seconds=result.seconds,
            )
        )
        self._outputs[seq] = result.output
        return seq

    def record_gate(
        self,
        stage: str,
        gate: str,
        disposition: str,
        card_id: str | None = None,
        detail: str = "",
    ) -> None:
        self.gates.append(
            GateRecord(
                stage=stage,
                gate=gate,
                disposition=disposition,
                card_id=card_id,
                detail=detail,
            )
        )

    def add_card(self, card: dict) -> None:
        self.cards.append(card)

    def add_claim(
        self, text_en: str, source_seq: int, tool: str, field_: str, value: Any
    ) -> None:
        self.claims.append(
            Claim(
                text_en=text_en,
                source_seq=source_seq,
                tool=tool,
                field=field_,
                value=value,
            )
        )

    def set_answer(self, answer: dict) -> None:
        self.answer = answer

    # --- grounding helpers ---------------------------------------------------
    def output_of(self, seq: int) -> Any:
        """Full recorded output for a tool-call seq (used to verify a claim)."""
        return self._outputs.get(seq)

    def claim_is_grounded(self, claim: Claim) -> bool:
        """True iff ``claim.value`` is present at ``claim.field`` of its source call."""
        out = self.output_of(claim.source_seq)
        if out is None:
            return False
        found = _resolve_path(out, claim.field)
        return _values_match(found, claim.value)

    def all_claims_grounded(self) -> bool:
        return all(self.claim_is_grounded(c) for c in self.claims)

    # --- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "planner": self.planner,
            "autonomy": self.autonomy,
            "thresholds_hash": self.thresholds_hash,
            "started_at": self.started_at,
            "written_at": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "n_tool_calls": len(self.tool_calls),
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "gates": [asdict(g) for g in self.gates],
            "cards": self.cards,
            "claims": [asdict(c) for c in self.claims],
            "grounding_ok": self.all_claims_grounded(),
            "answer": self.answer,
        }

    def write(self, out_dir: Path | str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"agent_trace_{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path


def config_hash(cfg_dict: dict) -> str:
    """Stable digest of a gate configuration (the anti-silent-weakening anchor)."""
    return digest(cfg_dict)


# --- field-path resolution ---------------------------------------------------
def _resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dotted/indexed path (e.g. ``units.0.pred_rul``) inside a dict/list.

    Returns a sentinel ``_MISSING`` when any segment is absent, so a claim that
    points at a non-existent field is treated as ungrounded rather than matching.
    """
    cur = obj
    for seg in path.split("."):
        if isinstance(cur, dict):
            if seg not in cur:
                return _MISSING
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError:
                return _MISSING
            if not (0 <= idx < len(cur)):
                return _MISSING
            cur = cur[idx]
        else:
            return _MISSING
    return cur


_MISSING = object()


def _values_match(found: Any, value: Any) -> bool:
    if found is _MISSING:
        return False
    if isinstance(found, float) or isinstance(value, float):
        try:
            return abs(float(found) - float(value)) < 1e-6
        except (TypeError, ValueError):
            return False
    return found == value
