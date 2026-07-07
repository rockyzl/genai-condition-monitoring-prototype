"""Tests for the demo bootstrap + root Streamlit entry.

1. Bootstrap is idempotent: when the artifacts already exist it is a fast no-op
   that never touches the network or the pipeline.
2. The root ``streamlit_app.py`` imports with no side effects — importing it must
   not run the app, call Streamlit, or even eagerly import the bootstrap module
   (all of that is deferred to ``main()`` under ``if __name__ == '__main__'``).

Run:
    .venv/bin/python -m pytest tests/test_bootstrap.py -q
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bootstrap_demo as bd  # noqa: E402


# --- 1. idempotency ----------------------------------------------------------
def test_bootstrap_is_noop_when_artifacts_present(monkeypatch):
    """Artifacts present → bootstrap returns immediately, no download, no pipeline."""

    # network + pipeline tripwires: bootstrap must touch neither when ready
    def _no_download(*a, **k):
        raise AssertionError("bootstrap must not download when artifacts exist")

    def _no_pipeline(*a, **k):
        raise AssertionError("bootstrap must not run the pipeline when artifacts exist")

    monkeypatch.setattr(bd, "artifacts_present", lambda: True)
    monkeypatch.setattr(bd, "ensure_data", _no_download)
    monkeypatch.setattr(bd, "run_pipeline_stages", _no_pipeline)
    monkeypatch.setattr(bd, "_download", _no_download)  # hard offline guard

    logs: list[str] = []
    summary = bd.bootstrap(force=False, log=logs.append)

    assert summary["skipped_all"] is True
    assert summary["downloaded"] is False
    assert summary["stages_run"] == 0
    assert summary["stages_skipped"] == 0


# --- 2. import-safety of the root entry --------------------------------------
def test_root_entry_imports_without_side_effects():
    # force a clean import so the assertions reflect *this* import
    sys.modules.pop("streamlit_app", None)
    sys.modules.pop("scripts.bootstrap_demo", None)
    app_preloaded = "src.app.streamlit_app" in sys.modules  # may be loaded by app tests

    mod = importlib.import_module("streamlit_app")

    # public surface present
    assert hasattr(mod, "main") and callable(mod.main)
    assert hasattr(mod, "artifacts_ready") and callable(mod.artifacts_ready)
    assert mod.APP_FILE.name == "streamlit_app.py"
    assert mod.APP_FILE.parent.name == "app"

    # the bootstrap module is imported lazily (inside functions), so a bare import
    # of the entry point must not have pulled it in — proof of no import-time work
    assert "scripts.bootstrap_demo" not in sys.modules
    # and importing the entry must not execute/delegate to the real app
    if not app_preloaded:
        assert "src.app.streamlit_app" not in sys.modules
