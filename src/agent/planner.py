"""Intent planner — maps a natural-language query to a fixed tool-call plan.

The default :class:`RuleBasedPlanner` is deterministic and key-free: a handful of
regex intent patterns turn a query into an ordered list of :class:`ToolCall`s.
This is not an LLM and is deliberately not marketed as one — reproducibility and
auditability are the point. The two canonical intents from the plan:

* ``"diagnose unit 81"`` → ``get_evidence(81)`` → ``retrieve(...)`` → ``diagnose(81)``
* ``"which engines need inspection?"`` → ``list_units_by_risk("high")`` with a
  ``get_prediction`` fan-out, so the answer can cite each flagged unit's own
  predicted RUL (grounding per unit).

An :class:`LLMPlanner` exists only as a flagged interface stub in the shape of a
subscription-CLI adapter; it holds no API key and raises
:class:`PlannerNotConfigured` the moment it is asked to plan. ``--planner llm``
therefore fails loudly instead of silently degrading to something non-deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ToolCall:
    """One planned tool invocation, optionally fanning out per returned unit.

    ``fan_out`` names tools to call once per unit in this call's ``units`` output
    (used so ``list_units_by_risk`` results can be grounded unit-by-unit with
    ``get_prediction``). It is empty for ordinary calls.
    """

    tool: str
    args: dict = field(default_factory=dict)
    fan_out: tuple[str, ...] = ()


@runtime_checkable
class Planner(Protocol):
    """A planner turns a query string into an ordered list of tool calls."""

    kind: str

    def plan(self, query: str) -> list[ToolCall]:
        ...


class PlannerNotConfigured(RuntimeError):
    """Raised when the (stub) LLM planner is invoked without a configured backend."""


# Terms fed to the retrieve step for a per-unit diagnosis. Deterministic and
# broad enough to surface both failure-mode and checklist guidance; the diagnose
# tool does its own targeted retrieval on top of this.
_DIAG_RETRIEVE_Q = (
    "failure mode fault wear degradation maintenance review checklist "
    "inspection next steps human review"
)

_UNIT_RE = re.compile(r"\b(?:unit|engine|asset)\s*#?\s*(\d{1,4})\b", re.IGNORECASE)
_DIAGNOSE_RE = re.compile(r"\b(diagnose|diagnosis|explain|why)\b", re.IGNORECASE)
_INSPECT_RE = re.compile(
    r"\b(inspect|inspection|need.*(?:attention|service|maintenance)|"
    r"which.*(?:engines|units|need)|high[\s-]*risk|flag|triage|worst|urgent)\b",
    re.IGNORECASE,
)
_REPORT_RE = re.compile(r"\breport\b", re.IGNORECASE)


@dataclass
class RuleBasedPlanner:
    """Deterministic regex intent → tool-call plan. The default planner."""

    kind: str = "rule"

    def plan(self, query: str) -> list[ToolCall]:
        q = (query or "").strip()
        unit_m = _UNIT_RE.search(q)

        # "report for unit N" — a full composed bundle for one unit.
        if unit_m and _REPORT_RE.search(q) and not _DIAGNOSE_RE.search(q):
            unit = int(unit_m.group(1))
            return [ToolCall("report", {"unit": unit})]

        # "diagnose unit N" — evidence → retrieval → grounded diagnosis.
        if unit_m and (_DIAGNOSE_RE.search(q) or not _INSPECT_RE.search(q)):
            unit = int(unit_m.group(1))
            return [
                ToolCall("get_evidence", {"unit": unit}),
                ToolCall("retrieve", {"q": _DIAG_RETRIEVE_Q, "k": 4}),
                ToolCall("diagnose", {"unit": unit}),
            ]

        # "which engines need inspection?" — high-risk list, grounded per unit.
        if _INSPECT_RE.search(q):
            return [
                ToolCall(
                    "list_units_by_risk",
                    {"band": "high"},
                    fan_out=("get_prediction",),
                )
            ]

        # Unmapped intent: empty plan. The orchestrator answers honestly that it
        # could not map the query, rather than inventing tool calls.
        return []


@dataclass
class LLMPlanner:
    """Flagged interface stub for a subscription-CLI LLM planner. Never runs.

    It matches the adapter shape a real LLM planner would take (a ``model`` name,
    an optional ``backend`` handle) but is intentionally inert: with no backend
    configured, :meth:`plan` raises immediately so ``--planner llm`` fails loudly.
    """

    kind: str = "llm"
    model: str = "unconfigured"
    backend: object | None = None

    def plan(self, query: str) -> list[ToolCall]:
        raise PlannerNotConfigured(
            "The LLM planner is a flagged interface stub with no configured "
            "backend. This prototype runs deterministically on the rule-based "
            "planner (--planner rule). Wiring a subscription-CLI or API backend "
            "is intentionally out of scope; nothing here calls a model."
        )


def make_planner(kind: str = "rule") -> Planner:
    """Factory: ``"rule"`` → :class:`RuleBasedPlanner`, ``"llm"`` → stub."""
    if kind == "rule":
        return RuleBasedPlanner()
    if kind == "llm":
        return LLMPlanner()
    raise ValueError(f"unknown planner {kind!r}; valid: 'rule', 'llm'")
