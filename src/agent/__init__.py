"""Deterministic pipeline agent with human-in-the-loop decision gates (Phase C).

This package is a thin, auditable orchestration layer over the Phase-A pipeline
(:mod:`src.pipeline`). It never reimplements modelling or diagnosis: it *calls*
the existing stage functions and ``src/`` code through a small typed tool
registry, and adds two entry modes on top of one registry —

* **Query** (:mod:`src.agent.query`): a rule-based planner maps an intent like
  ``"diagnose unit 81"`` to a tool-call plan; the answer is composed only from
  tool outputs, every claim carrying a trace id / KB citation.
* **Autopilot** (:mod:`src.agent.autopilot`): a supervisor walks the fixed
  10-stage DAG with a per-stage state machine (EXECUTE → VALIDATE → pass /
  RAISE a decision card / HALT), streaming a journal the app renders live and
  handing the human a small Decision Inbox of cards.

Naming discipline (enforced by the plan): this is a *deterministic pipeline
supervisor with HITL decision gates*, never a "fully autonomous" or "LLM" agent.
The only LLM planner is a flagged interface stub that errors if invoked.
"""

from __future__ import annotations

__all__ = ["registry", "planner", "trace", "cards", "autopilot", "query"]
