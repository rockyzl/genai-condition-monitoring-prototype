"""Agent CLI — two entry modes over one registry.

    # Autopilot: walk s01→s10 with HITL decision gates
    python -m src.agent run --all                       # gated (default)
    python -m src.agent run --all --autonomy dry-run    # raise nothing, report would-raise
    python -m src.agent run --all --yes-safe-defaults   # accept safe defaults at the CLI
    python -m src.agent run --all --autonomy auto        # auto-pass all but triage/sign-off

    # Query: a rule-planned, grounded, cited answer
    python -m src.agent ask "which engines need inspection?"
    python -m src.agent ask "diagnose unit 81"

Gated runs with no UI: the supervisor prints the pending card and its safe
default, then either (a) applies the safe default if ``--yes-safe-defaults`` is
given, or (b) exits leaving the card in ``autopilot_inbox/pending/``. To resume,
drop a ``{"card_id": ..., "action": ...}`` file into ``autopilot_inbox/answered/``
and re-run — earlier stages skip via provenance.
"""

from __future__ import annotations

import argparse
import sys

from src.agent.autopilot import Autopilot
from src.agent.planner import PlannerNotConfigured
from src.agent.query import answer_query
from src.pipeline.config import PipelineConfig


def _print_card(card: dict, indent: str = "  ") -> None:
    print(f"{indent}[{card['priority']}] {card['kind']}  ({card['id']})")
    print(f"{indent}  {card['verdict_en']}")
    print(f"{indent}  {card['verdict_zh']}")
    for s in card["signals"]:
        print(f"{indent}  · {s['text_en']}")
        print(f"{indent}    ↳ {s['artifact']} [{s['field']}]")
    print(f"{indent}  actions:")
    for a in card["actions"]:
        star = " (safe default)" if a["safe_default"] else ""
        print(f"{indent}    - {a['id']}{star}: {a['label_en']}")
        print(f"{indent}      → {a['consequence_en']}")


def _cmd_run(args) -> int:
    cfg = PipelineConfig.load(args.config)
    pilot = Autopilot(
        cfg,
        autonomy=args.autonomy,
        yes_safe_defaults=args.yes_safe_defaults,
        force=args.force,
    )
    report = pilot.run()

    print(f"[autopilot] run {report.run_id}  autonomy={report.autonomy}  "
          f"status={report.status.upper()}")
    print(f"[autopilot] gate thresholds hash: {report.thresholds_hash}")
    for row in report.stages:
        gates = ", ".join(
            f"{g['gate']}:{g['disposition']}" for g in row["gates"]
        )
        tag = "skip" if row["skipped"] else "ran "
        print(f"  [{tag}] {row['stage']:<14} {row['seconds']:6.3f}s  {gates}")

    if report.halt:
        print(f"\n[autopilot] HALTED at {report.halt['stage']} "
              f"({report.halt['gate']}): {report.halt['reason']}")

    if report.autonomy == "dry-run" and report.would_raise:
        print("\n[dry-run] cards that WOULD be raised:")
        for w in report.would_raise:
            block = "  (blocking — never auto-passes)" if w["would_block"] else ""
            print(f"  · [{w['priority']}] {w['kind']} — {w['verdict_en']}{block}")

    if report.cards_resolved:
        print("\n[autopilot] resolved cards:")
        for c in report.cards_resolved:
            print(f"  · {c['kind']} → {c['action']}  (via {c['source']})")

    if report.cards_pending:
        print("\n[autopilot] AWAITING YOUR DECISION — pending card:")
        for c in report.cards_pending:
            _print_card(c)
        safe = report.cards_pending[0]["actions"]
        safe_id = next((a["id"] for a in safe if a["safe_default"]), None)
        print(f"\n  To accept the safe default now, re-run with --yes-safe-defaults")
        print(f"  (unless this card forbids it), or write an answer file:")
        print(f"    {report.inbox_pending_dir.replace('/pending', '/answered')}/"
              f"{report.cards_pending[0]['id']}.json")
        print(f'    {{"card_id": "{report.cards_pending[0]["id"]}", '
              f'"action": "{safe_id}"}}')
        print(f"  then re-run `python -m src.agent run --all` to resume.")

    if report.digest:
        print(f"\n[autopilot] {report.digest['sentence_en']}")
        n_decisions = len(report.cards_resolved) + len(report.cards_pending)
        print(f"[autopilot] You had {n_decisions} decision(s); "
              f"{report.digest['n_healthy']} healthy auto-cleared.")

    print(f"\n[autopilot] journal -> {report.journal_path}")
    print(f"[autopilot] trace   -> {report.trace_path}")
    print(f"[autopilot] state   -> {report.state_path}")
    return 0


def _cmd_ask(args) -> int:
    cfg = PipelineConfig.load(args.config)
    try:
        result = answer_query(cfg, args.query, planner_kind=args.planner)
    except PlannerNotConfigured as exc:
        print(f"[ask] {exc}", file=sys.stderr)
        return 3
    ans = result["answer"]
    print(f"Q: {ans['question']}")
    print(f"\n{ans['answer_en']}")
    print(f"\n{ans['answer_zh']}")
    if ans.get("citations"):
        print("\nCitations:")
        for c in ans["citations"]:
            if "source_file" in c:
                print(f"  · {c['source_file']} › {c.get('section')}")
            else:
                print(f"  · {c.get('artifact')} — {c.get('note', '')}")
    print(f"\n[ask] claims grounded in tool outputs: {result['grounded']} "
          f"({len(ans.get('claims', []))} claims)")
    print(f"[ask] trace -> {result['trace_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.agent",
        description="Deterministic pipeline agent with human-in-the-loop decision gates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="autopilot: walk s01→s10 with decision gates")
    run_p.add_argument("--all", action="store_true", help="walk all 10 stages (default)")
    run_p.add_argument("--autonomy", choices=["gated", "auto", "dry-run"],
                       default="gated", help="autonomy mode (default: gated)")
    run_p.add_argument("--yes-safe-defaults", action="store_true",
                       help="accept a raised card's safe default at the CLI "
                       "(never applies to sign-off)")
    run_p.add_argument("--force", action="store_true",
                       help="ignore provenance; re-execute every stage")
    run_p.add_argument("--config", metavar="PATH", help="path to pipeline.yaml")

    ask_p = sub.add_parser("ask", help="query: a grounded, cited answer")
    ask_p.add_argument("query", help="e.g. 'which engines need inspection?'")
    ask_p.add_argument("--planner", choices=["rule", "llm"], default="rule",
                       help="planner backend (llm is a not-configured stub)")
    ask_p.add_argument("--config", metavar="PATH", help="path to pipeline.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "ask":
        return _cmd_ask(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
