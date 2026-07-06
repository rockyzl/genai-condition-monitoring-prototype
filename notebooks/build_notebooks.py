"""Build and execute the prototype notebooks without a Jupyter install.

The project venv intentionally has no jupyter / nbconvert / ipykernel (kept
lean, no heavy deps). This script authors the two notebooks as nbformat-4.5
JSON and then executes each code cell in-process — capturing stdout and any
matplotlib figures as embedded PNG outputs — so the committed ``.ipynb`` files
open with their story already rendered, exactly as ``jupyter nbconvert
--execute`` would produce.

Run:
    .venv/bin/python notebooks/build_notebooks.py

Outputs:
    notebooks/01_eda.ipynb
    notebooks/02_baseline_model.ipynb
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")  # headless before any pyplot import

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NB_DIR = pathlib.Path(__file__).resolve().parent
ROOT = NB_DIR.parent


# --- Minimal nbformat-4.5 construction ---------------------------------------


def md(*lines: str) -> dict:
    text = "\n".join(lines)
    return {"cell_type": "markdown", "metadata": {}, "source": _split(text)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _split(src.strip("\n")),
    }


def _split(text: str) -> list[str]:
    """nbformat stores source as a list of lines each ending in newline
    (except the last)."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --- Execution ---------------------------------------------------------------


def execute(nb: dict) -> dict:
    """Execute code cells in a shared namespace, embedding outputs."""
    ns: dict = {"__name__": "__notebook__"}
    counter = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        counter += 1
        cell["execution_count"] = counter
        outputs: list[dict] = []
        src = "".join(cell["source"])
        stdout = io.StringIO()
        import contextlib

        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(src, f"<cell {counter}>", "exec"), ns)
        except Exception:
            tb = traceback.format_exc()
            outputs.append(
                {
                    "output_type": "error",
                    "ename": "Error",
                    "evalue": "cell failed",
                    "traceback": tb.splitlines(),
                }
            )
            cell["outputs"] = outputs
            _dump(nb)  # persist partial for debugging
            raise RuntimeError(f"Cell {counter} failed:\n{tb}")

        text = stdout.getvalue()
        if text:
            outputs.append(
                {"output_type": "stream", "name": "stdout", "text": _split(text)}
            )
        # Embed any figures the cell created.
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"image/png": b64, "text/plain": ["<Figure>"]},
                    "metadata": {},
                }
            )
            plt.close(fig)
        cell["outputs"] = outputs
    return nb


def _dump(nb: dict, path: pathlib.Path | None = None) -> None:
    path = path or _dump.current  # type: ignore[attr-defined]
    with open(path, "w") as fh:
        json.dump(nb, fh, indent=1)


# --- Notebook content --------------------------------------------------------

SETUP = """
import sys, pathlib
ROOT = pathlib.Path.cwd()
ROOT = ROOT if (ROOT / "src").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.width", 120)
print("repo root:", ROOT)
"""


def build_eda() -> dict:
    cells = [
        md(
            "# 01 — Exploratory Data Analysis (C-MAPSS FD001)",
            "",
            "**Independent R&D prototype on public NASA data — not Caterpillar data, "
            "not production.** See `docs/data-sources.md` for the citation and the "
            "proxy caveat.",
            "",
            "This notebook is intentionally thin: all logic lives in `src/`, the "
            "notebook just tells the story.",
        ),
        code(SETUP),
        code(
            "from src.data.load_cmapss import load_raw, add_training_rul, RUL_CAP\n"
            "from src.features.build_features import (\n"
            "    INFORMATIVE_SENSORS, DROPPED_SENSORS)\n"
            "train = load_raw('FD001', 'train')\n"
            "print('shape:', train.shape, '| units:', train.unit.nunique())\n"
            "train.head()"
        ),
        md(
            "## Target: Remaining Useful Life (RUL)",
            "",
            "Training RUL = (last cycle of the unit) − (current cycle), capped at "
            f"**{125}** cycles. The cap encodes 'anything beyond ~125 cycles of "
            "remaining life is simply healthy', which stabilises the regression "
            "target — see the loader docstring.",
        ),
        code(
            "train_rul = add_training_rul(train)\n"
            "print(train_rul['rul'].describe().round(1).to_string())\n"
            "fig, ax = plt.subplots(figsize=(7,3.5))\n"
            "ax.hist(train_rul['rul'], bins=40, edgecolor='k', alpha=0.8)\n"
            "ax.set_xlabel('capped RUL'); ax.set_ylabel('rows')\n"
            "ax.set_title('FD001 train: capped RUL distribution')\n"
            "None"
        ),
        md(
            "## Which sensors carry signal?",
            "",
            "FD001 is a single operating condition + single fault mode, so several "
            "sensors are constant or near-constant and are dropped. The rest trend "
            "monotonically as the engine degrades.",
        ),
        code(
            "sensor_cols = [c for c in train.columns if c.startswith('sensor_')]\n"
            "stds = train[sensor_cols].std().sort_values()\n"
            "print('Dropped (constant/near-constant):', DROPPED_SENSORS)\n"
            "print('Informative (%d):' % len(INFORMATIVE_SENSORS), INFORMATIVE_SENSORS)\n"
            "stds.round(4).to_frame('std')"
        ),
        code(
            "# Degradation: informative sensors vs cycles-to-failure for 3 units;\n"
            "# plus one flat (dropped) sensor for contrast.\n"
            "units = [1, 2, 3]\n"
            "panels = ['sensor_4', 'sensor_11', 'sensor_15', 'sensor_1']\n"
            "fig, axes = plt.subplots(len(panels), 1, figsize=(8, 9))\n"
            "for ax, s in zip(axes, panels):\n"
            "    for u in units:\n"
            "        g = train[train.unit == u]\n"
            "        ax.plot(g.cycle.max() - g.cycle, g[s], linewidth=1, label=f'unit {u}')\n"
            "    ax.set_ylabel(s); ax.invert_xaxis(); ax.grid(alpha=0.3)\n"
            "flat = ' (flat / dropped)'\n"
            "axes[-1].set_ylabel(panels[-1] + flat)\n"
            "axes[0].set_title('Sensor trends vs cycles remaining to failure')\n"
            "axes[-1].set_xlabel('cycles remaining until failure')\n"
            "axes[0].legend(fontsize=8)\n"
            "None"
        ),
        md(
            "**Takeaways.** Informative sensors (4, 11, 15, …) drift steadily as "
            "failure approaches — that drift is the learnable signal. Sensor 1 is "
            "flat and is dropped. The rolling mean/std features in "
            "`src/features/build_features.py` denoise and quantify this drift.",
        ),
    ]
    return notebook(cells)


def build_model() -> dict:
    cells = [
        md(
            "# 02 — RUL Baseline Model (RandomForest, FD001)",
            "",
            "**Independent R&D prototype on public NASA data.** Metrics and figures "
            "below are regenerated from the artifacts written by "
            "`src/models/train_baseline.py`.",
        ),
        code(SETUP),
        code(
            "import json\n"
            "from src.models.train_baseline import train_and_evaluate\n"
            "proc = ROOT / 'data' / 'processed'\n"
            "metrics_path = ROOT / 'reports' / 'metrics_model.json'\n"
            "if not (proc / 'test_predictions.csv').exists():\n"
            "    train_and_evaluate()\n"
            "metrics = json.loads(metrics_path.read_text())\n"
            "print('model      :', metrics['model'])\n"
            "print('rul_cap    :', metrics['rul_cap'])\n"
            "print('test units :', metrics['n_test_units'])\n"
            "print('vs capped   truth:', metrics['metrics_vs_capped_truth'])\n"
            "print('vs uncapped truth:', metrics['metrics_vs_uncapped_truth'])"
        ),
        md(
            "## Predictions and risk bands",
            "",
            "Each test unit is scored at its **last** recorded cycle. `risk_band` is "
            "derived from predicted RUL: high ≤ 30 < medium ≤ 80 < low.",
        ),
        code(
            "preds = pd.read_csv(proc / 'test_predictions.csv')\n"
            "print('rows:', len(preds), '| bands:', preds.risk_band.value_counts().to_dict())\n"
            "preds.head()"
        ),
        code(
            "# Predicted vs true (capped), coloured by risk band.\n"
            "cap = metrics['rul_cap']\n"
            "tc = preds.true_rul.clip(upper=cap)\n"
            "colors = {'high':'#c0392b','medium':'#e67e22','low':'#27ae60'}\n"
            "fig, ax = plt.subplots(figsize=(6,6))\n"
            "for b,c in colors.items():\n"
            "    m = preds.risk_band == b\n"
            "    ax.scatter(tc[m], preds.pred_rul[m], c=c, label=b, edgecolor='k', linewidth=.3, alpha=.8)\n"
            "lim = max(tc.max(), preds.pred_rul.max())+5\n"
            "ax.plot([0,lim],[0,lim],'k--',alpha=.6)\n"
            "ax.set_xlabel('true RUL (capped)'); ax.set_ylabel('predicted RUL')\n"
            "ax.set_title('Predicted vs true RUL'); ax.legend(title='risk band')\n"
            "None"
        ),
        code(
            "# Feature importances (top 12).\n"
            "fi = pd.read_csv(proc / 'feature_importances.csv').head(12).iloc[::-1]\n"
            "fig, ax = plt.subplots(figsize=(7,5))\n"
            "ax.barh(fi.feature, fi.importance, color='#2c7fb8')\n"
            "ax.set_xlabel('importance'); ax.set_title('Top-12 RandomForest feature importances')\n"
            "None"
        ),
        md(
            "## Highest-error units",
            "",
            "Full write-up in `reports/error_analysis.md`. The largest residuals are "
            "optimistic over-predictions on mid-life units — the operationally risky "
            "direction, which is why the deployment story keeps a human in the loop.",
        ),
        code(
            "preds['abs_err'] = (preds.pred_rul - preds.true_rul).abs()\n"
            "preds.sort_values('abs_err', ascending=False).head(5)"
        ),
        md(
            "**Bottom line.** A plain RandomForest on 14 informative sensors + short "
            "rolling stats reaches RMSE ≈ 17 (capped truth) on FD001 test — a "
            "credible, honest baseline, not a tuned SOTA model. Limitations and next "
            "steps: `reports/error_analysis.md` and `docs/data-sources.md`.",
        ),
    ]
    return notebook(cells)


def main() -> None:
    targets = [
        (NB_DIR / "01_eda.ipynb", build_eda),
        (NB_DIR / "02_baseline_model.ipynb", build_model),
    ]
    for path, builder in targets:
        nb = builder()
        _dump.current = path  # type: ignore[attr-defined]
        nb = execute(nb)
        _dump(nb, path)
        n_out = sum(len(c.get("outputs", [])) for c in nb["cells"] if c["cell_type"] == "code")
        print(f"[write] {path.name}  ({n_out} outputs embedded)")


if __name__ == "__main__":
    main()
