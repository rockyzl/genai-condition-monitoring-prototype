"""Public web-demo entry point.

Hugging Face Spaces (streamlit SDK) and Streamlit Community Cloud both auto-detect
a ``streamlit_app.py`` at the repo root and run it with ``streamlit run``. This
thin wrapper makes the demo self-bootstrapping:

* **Fresh Space / cold start** — the pipeline artifacts (data, model, evidence,
  metrics) are all gitignored, so on first boot they are absent. We render a
  bilingual bootstrap screen, download NASA C-MAPSS + run the 10-stage pipeline
  (~2-3 min), then rerun.
* **Warm boot** — artifacts present, so we delegate straight to the real app in
  ``src/app/streamlit_app.py``.

Delegation is deliberate: the real app owns ``st.set_page_config`` and is being
actively developed, so we execute it as ``__main__`` (via ``runpy``) and never
import-time-execute it or call ``set_page_config`` ourselves on the delegate
path — that would double-configure the page.

Import-safe: all active logic is guarded under ``if __name__ == "__main__"`` (the
name Streamlit uses when it runs this file), so ``import streamlit_app`` in a
test has no side effects.

Test/ops seams (both off in normal use):
* ``DEMO_FORCE_BOOTSTRAP`` — treat artifacts as missing (exercise the boot path).
* ``DEMO_BOOTSTRAP_DRYRUN`` — render the boot screen but skip the real
  download/pipeline (lets a headless serve check hit the boot path without a
  3-minute download).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_FILE = ROOT / "src" / "app" / "streamlit_app.py"

PAGE_TITLE = "Condition Monitoring · 状态监测"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def artifacts_ready() -> bool:
    """Whether the demo can run without bootstrapping (env seam can force False)."""
    if _truthy(os.environ.get("DEMO_FORCE_BOOTSTRAP")):
        return False
    from scripts.bootstrap_demo import artifacts_present

    return artifacts_present()


def _run_bootstrap_screen() -> None:
    """Render the first-boot screen and build the artifacts, then rerun."""
    import streamlit as st

    # We own the page here (the delegate path does NOT reach this branch), so it
    # is safe and correct to configure the page once.
    st.set_page_config(page_title=PAGE_TITLE, layout="centered")
    st.title("🛠️ Condition Monitoring Demo")
    st.subheader("First boot / 首次启动")
    st.write(
        "**EN** — Preparing the demo: downloading the public NASA C-MAPSS "
        "turbofan dataset (~12 MB) and running the 10-stage pipeline "
        "(train → predict → evidence → diagnose → evaluate). This takes about "
        "**2-3 minutes** and only happens on a cold start."
    )
    st.write(
        "**中文** — 正在准备演示：下载公开的 NASA C-MAPSS 涡扇数据集（约 12 MB），"
        "并运行 10 阶段流水线（训练 → 预测 → 证据 → 诊断 → 评估）。大约需要 "
        "**2-3 分钟**，且只在冷启动时发生。"
    )

    if _truthy(os.environ.get("DEMO_BOOTSTRAP_DRYRUN")):
        st.warning("DRY RUN: skipping the real download/pipeline (test seam).")
        st.stop()

    from scripts.bootstrap_demo import bootstrap

    progress_bar = st.progress(0.0, text="Downloading dataset / 下载数据…")

    with st.status("Building the demo… / 正在构建演示…", expanded=True) as status:

        def log(message: str) -> None:
            status.write(message)

        def progress(fraction: float) -> None:
            frac = min(max(float(fraction), 0.0), 1.0)
            progress_bar.progress(frac, text=f"Downloading dataset / 下载数据… {frac:.0%}")

        try:
            summary = bootstrap(force=False, log=log, progress=progress)
        except Exception as exc:  # keep the Space alive with an actionable message
            status.update(label="Bootstrap failed / 构建失败", state="error")
            st.error(
                f"Bootstrap failed: {exc}\n\n"
                "The NASA S3 mirror may be temporarily unreachable. Reload the page "
                "to retry, or check the Space logs."
            )
            st.stop()
            return

        progress_bar.progress(1.0, text="Done / 完成")
        status.update(label="Demo ready ✅ / 演示就绪", state="complete")

    st.success(
        f"Ready — trained the model and prepared {summary.get('stages_run', 0)} "
        "pipeline stage(s). Reloading… / 就绪，正在重新加载…"
    )
    st.rerun()


def _delegate_to_app() -> None:
    """Execute the real app as ``__main__``; it owns ``st.set_page_config``."""
    if not APP_FILE.exists():
        import streamlit as st

        st.set_page_config(page_title=PAGE_TITLE, layout="centered")
        st.error(f"App entry not found at {APP_FILE}. This is a packaging error.")
        st.stop()
        return
    # runpy executes the file top-to-bottom with __name__ == "__main__", exactly
    # as `streamlit run src/app/streamlit_app.py` would — no import-time coupling.
    runpy.run_path(str(APP_FILE), run_name="__main__")


def main() -> None:
    if artifacts_ready():
        _delegate_to_app()
    else:
        _run_bootstrap_screen()


if __name__ == "__main__":
    main()
