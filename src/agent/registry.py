"""Typed tool registry — the agent's only way to touch the system.

Seven hand-written tools wrap the functions that already exist in ``src/`` and
``src/pipeline/``; there is no plugin system and no schema DSL. Each tool
declares a small dataclass argument schema, and :meth:`Registry.call` validates
before dispatching — unknown tool names, unknown argument keys, missing required
arguments, wrong types, and out-of-vocabulary choices are all rejected with a
clear :class:`ToolError` *before* any handler runs. That validation boundary is
what lets the planner and supervisor stay honest: nothing reaches the pipeline
except a well-formed, in-vocabulary call.

Every call returns a :class:`ToolResult` carrying the JSON-serialisable output,
a short content digest, and a duration — the unit the trace records and the
grounding rule checks against (answers/cards compose only from these outputs).

Tools
-----
* ``run_stage(stage)``            — run one pipeline stage via the Phase-A runner.
* ``get_prediction(unit)``        — the model's row for one test unit.
* ``get_evidence(unit)``          — the structured evidence record for one unit.
* ``retrieve(q, k)``              — top-k knowledge-base chunks (TF-IDF).
* ``diagnose(unit)``              — the grounded, cited diagnostic report.
* ``list_units_by_risk(band)``    — units in a risk band, ordered by predicted RUL.
* ``report(unit)``                — a composed prediction + evidence + diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.pipeline.config import PipelineConfig

VALID_BANDS = ("high", "medium", "low")


class ToolError(ValueError):
    """Raised for an unknown tool, unknown/missing arg, bad type, or bad value."""


# --- schema -----------------------------------------------------------------
@dataclass(frozen=True)
class ArgSpec:
    """One tool argument: its coercion type, whether it is required, and choices."""

    name: str
    type: type
    required: bool = True
    default: Any = None
    choices: tuple | None = None

    def coerce(self, value: Any) -> Any:
        """Validate + coerce ``value`` to this arg's type, or raise ToolError."""
        if self.type is int:
            # Reject bools and non-integral values explicitly (bool is an int).
            if isinstance(value, bool):
                raise ToolError(f"arg {self.name!r} must be an int, got bool")
            try:
                coerced = int(value)
            except (TypeError, ValueError):
                raise ToolError(f"arg {self.name!r} must be an int, got {value!r}")
        elif self.type is str:
            if not isinstance(value, str):
                raise ToolError(f"arg {self.name!r} must be a str, got {value!r}")
            coerced = value
        elif self.type is float:
            try:
                coerced = float(value)
            except (TypeError, ValueError):
                raise ToolError(f"arg {self.name!r} must be a number, got {value!r}")
        else:  # pragma: no cover - no other arg types are declared
            coerced = value
        if self.choices is not None and coerced not in self.choices:
            raise ToolError(
                f"arg {self.name!r} must be one of {self.choices}, got {coerced!r}"
            )
        return coerced


@dataclass(frozen=True)
class ToolSpec:
    """A named tool: its human description, argument schema, and handler."""

    name: str
    description: str
    args: tuple[ArgSpec, ...]
    handler: Callable[["Registry", dict], dict]

    def validate(self, raw_args: dict | None) -> dict:
        """Return a fully-validated, defaults-filled kwargs dict, or raise."""
        raw_args = dict(raw_args or {})
        known = {a.name for a in self.args}
        unknown = set(raw_args) - known
        if unknown:
            raise ToolError(
                f"tool {self.name!r} got unknown args {sorted(unknown)}; "
                f"valid: {sorted(known)}"
            )
        out: dict = {}
        for spec in self.args:
            if spec.name in raw_args:
                out[spec.name] = spec.coerce(raw_args[spec.name])
            elif spec.required:
                raise ToolError(f"tool {self.name!r} missing required arg {spec.name!r}")
            else:
                out[spec.name] = spec.default
        return out


@dataclass
class ToolResult:
    """The outcome of one tool call — the grounded unit the trace records."""

    tool: str
    args: dict
    ok: bool
    output: Any = None
    error: str | None = None
    digest: str = ""
    seconds: float = 0.0

    def preview(self) -> Any:
        """A compact, log-safe view of the output for the trace file."""
        return _preview(self.output)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def digest(obj: Any) -> str:
    """Short stable content digest of a tool output (for the trace)."""
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()[:16]


def _preview(obj: Any, _depth: int = 0) -> Any:
    """Shrink a nested output to a small, readable summary for the trace file."""
    if isinstance(obj, dict):
        if _depth >= 2:
            return {"keys": sorted(obj)[:12]}
        return {k: _preview(v, _depth + 1) for k, v in list(obj.items())[:12]}
    if isinstance(obj, list):
        head = [_preview(v, _depth + 1) for v in obj[:3]]
        if len(obj) > 3:
            head.append(f"… (+{len(obj) - 3} more)")
        return head
    if isinstance(obj, str) and len(obj) > 200:
        return obj[:200] + "…"
    return obj


# --- handlers ----------------------------------------------------------------
def _preds_frame(cfg: PipelineConfig):
    import pandas as pd

    path = cfg.path("data_processed") / "test_predictions.csv"
    if not path.exists():
        raise ToolError(
            f"predictions not built yet ({path}); run stage s07_predict first"
        )
    df = pd.read_csv(path)
    return df, path


def _h_run_stage(reg: "Registry", args: dict) -> dict:
    from src.pipeline.runner import run_pipeline
    from src.pipeline.specs import STAGE_ORDER

    stage = args["stage"]
    if stage not in STAGE_ORDER:
        raise ToolError(f"unknown stage {stage!r}; valid: {STAGE_ORDER}")
    ctx = run_pipeline(reg.cfg, [stage])
    r = ctx.results[0]
    return {
        "stage": r.name,
        "skipped": r.skipped,
        "seconds": round(r.seconds, 4),
        "rows": r.rows,
        "outputs": r.outputs,
        "key_metrics": r.key_metrics,
    }


def _h_get_prediction(reg: "Registry", args: dict) -> dict:
    df, path = _preds_frame(reg.cfg)
    unit = args["unit"]
    row = df[df["unit_id"] == unit]
    if row.empty:
        raise ToolError(f"unit {unit} not found in {path.name}")
    r = row.iloc[0]
    return {
        "unit_id": int(r["unit_id"]),
        "last_cycle": int(r["last_cycle"]),
        "pred_rul": float(r["pred_rul"]),
        "risk_band": str(r["risk_band"]),
        "artifact": reg.rel(path),
    }


def _h_get_evidence(reg: "Registry", args: dict) -> dict:
    unit = args["unit"]
    path = reg.cfg.path("evidence") / f"unit_{unit}.json"
    if not path.exists():
        raise ToolError(
            f"evidence for unit {unit} not built yet ({path}); run s08_evidence"
        )
    ev = json.loads(path.read_text())
    ev["artifact"] = reg.rel(path)
    return ev


def _h_retrieve(reg: "Registry", args: dict) -> dict:
    from src.rag.assistant import _first_sentence

    hits = reg.retriever.retrieve(args["q"], k=args["k"])
    return {
        "query": args["q"],
        "k": args["k"],
        "n_hits": len(hits),
        "hits": [
            {
                "source_file": h["source_file"],
                "section": h["section"],
                "score": h["score"],
                "excerpt": _first_sentence(h["text"]),
            }
            for h in hits
        ],
    }


def _h_diagnose(reg: "Registry", args: dict) -> dict:
    from src.rag.assistant import diagnose as diagnose_fn

    ev = _h_get_evidence(reg, {"unit": args["unit"]})
    report = diagnose_fn(ev, reg.retriever)
    return report


def _h_list_units_by_risk(reg: "Registry", args: dict) -> dict:
    df, path = _preds_frame(reg.cfg)
    band = args["band"]
    sub = df[df["risk_band"] == band].sort_values("pred_rul").reset_index(drop=True)
    units = [
        {
            "unit_id": int(r["unit_id"]),
            "last_cycle": int(r["last_cycle"]),
            "pred_rul": float(r["pred_rul"]),
            "risk_band": str(r["risk_band"]),
        }
        for _, r in sub.iterrows()
    ]
    return {
        "band": band,
        "n": len(units),
        "units": units,
        "artifact": reg.rel(path),
    }


def _h_report(reg: "Registry", args: dict) -> dict:
    unit = args["unit"]
    pred = _h_get_prediction(reg, {"unit": unit})
    ev = _h_get_evidence(reg, {"unit": unit})
    diag = _h_diagnose(reg, {"unit": unit})
    return {
        "unit_id": unit,
        "prediction": pred,
        "evidence_summary": {
            "predicted_rul": ev.get("predicted_rul"),
            "risk_band": ev.get("risk_band"),
            "top_contributing_signals": ev.get("top_contributing_signals", [])[:3],
        },
        "diagnosis": {
            "summary": diag.get("summary"),
            "possible_failure_modes": diag.get("possible_failure_modes", [])[:2],
            "recommended_next_steps": diag.get("recommended_next_steps", [])[:2],
            "citations": diag.get("citations", []),
            "uncertainty": diag.get("uncertainty"),
            "human_review_required": diag.get("human_review_required"),
        },
    }


# --- registry ----------------------------------------------------------------
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "run_stage",
        "Run one pipeline stage through the deterministic Phase-A runner.",
        (ArgSpec("stage", str),),
        _h_run_stage,
    ),
    ToolSpec(
        "get_prediction",
        "Return the model's predicted RUL + risk band for one test unit.",
        (ArgSpec("unit", int),),
        _h_get_prediction,
    ),
    ToolSpec(
        "get_evidence",
        "Return the structured evidence record (sensors, signals, uncertainty) "
        "for one test unit.",
        (ArgSpec("unit", int),),
        _h_get_evidence,
    ),
    ToolSpec(
        "retrieve",
        "Return the top-k knowledge-base chunks for a query (TF-IDF).",
        (ArgSpec("q", str), ArgSpec("k", int, required=False, default=4)),
        _h_retrieve,
    ),
    ToolSpec(
        "diagnose",
        "Return the grounded, cited diagnostic report for one test unit.",
        (ArgSpec("unit", int),),
        _h_diagnose,
    ),
    ToolSpec(
        "list_units_by_risk",
        "List test units in a risk band, ordered by ascending predicted RUL.",
        (ArgSpec("band", str, choices=VALID_BANDS),),
        _h_list_units_by_risk,
    ),
    ToolSpec(
        "report",
        "Compose a full prediction + evidence + diagnosis bundle for one unit.",
        (ArgSpec("unit", int),),
        _h_report,
    ),
)


@dataclass
class Registry:
    """The typed tool boundary. Construct once per run; call tools by name."""

    cfg: PipelineConfig
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tools:
            self.tools = {t.name: t for t in TOOL_SPECS}
        self._retriever = None

    @property
    def retriever(self):
        """Lazily-built KB retriever (shared across calls in a run)."""
        if self._retriever is None:
            from src.rag.retriever import Retriever

            self._retriever = Retriever(self.cfg.path("knowledge_base"))
        return self._retriever

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).relative_to(self.cfg.root))
        except ValueError:
            return str(path)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        """Validate ``args`` against the tool schema, then dispatch and time it."""
        if name not in self.tools:
            raise ToolError(
                f"unknown tool {name!r}; valid tools: {self.names()}"
            )
        spec = self.tools[name]
        validated = spec.validate(args)  # raises ToolError on any bad input
        t0 = time.perf_counter()
        try:
            output = spec.handler(self, validated)
        except ToolError:
            raise
        except Exception as exc:  # surface handler failures as a failed result
            seconds = time.perf_counter() - t0
            return ToolResult(
                tool=name,
                args=validated,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                seconds=round(seconds, 4),
            )
        seconds = time.perf_counter() - t0
        return ToolResult(
            tool=name,
            args=validated,
            ok=True,
            output=output,
            digest=digest(output),
            seconds=round(seconds, 4),
        )
