"""Loader for the NASA C-MAPSS turbofan degradation dataset (FD001 first).

The raw files are space-separated with 26 columns and no header:

    1) unit number
    2) time, in cycles
    3-5) operational settings 1..3
    6-26) sensor measurements 1..21

See ``docs/data-sources.md`` for the full column meanings and citation.

Design notes
------------
* Column names are assigned here so the rest of the pipeline never refers to
  bare integer positions.
* Training RUL is computed as ``max_cycle_per_unit - cycle`` and then *capped*
  at ``RUL_CAP`` (125). This piecewise-linear target is standard practice on
  C-MAPSS: early in an engine's life the true remaining life is large and only
  weakly reflected in the sensors, so regressing against an uncapped, very large
  RUL teaches the model to fit noise. Capping tells the model "anything beyond
  ~125 cycles of remaining life is simply "healthy"", which both matches how a
  maintenance team would act and markedly improves error metrics. The cap is a
  modelling choice, documented and configurable, not a property of the data.
"""

from __future__ import annotations

import pathlib

import pandas as pd

# --- Constants ---------------------------------------------------------------

#: Piecewise-linear RUL cap (cycles). See module docstring for rationale.
RUL_CAP = 125

#: Repository root, derived from this file's location (src/data/load_cmapss.py).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Default location of the extracted C-MAPSS text files.
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "CMAPSSData"

#: The three operational-setting columns.
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]

#: The 21 sensor columns.
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

#: Full ordered column list matching the raw file layout.
COLUMNS = ["unit", "cycle"] + OP_SETTING_COLS + SENSOR_COLS


# --- Loaders -----------------------------------------------------------------


def _raw_path(dataset: str, split: str, data_dir: pathlib.Path) -> pathlib.Path:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return data_dir / f"{split}_{dataset}.txt"


def load_raw(
    dataset: str = "FD001",
    split: str = "train",
    data_dir: pathlib.Path | str | None = None,
) -> pd.DataFrame:
    """Load a raw C-MAPSS split into a DataFrame with named columns.

    Parameters
    ----------
    dataset:
        One of ``FD001``..``FD004``. This prototype only exercises ``FD001``.
    split:
        ``"train"`` or ``"test"``.
    data_dir:
        Directory holding the extracted text files. Defaults to
        ``data/raw/CMAPSSData`` under the repo root.

    Returns
    -------
    DataFrame with columns :data:`COLUMNS` (unit, cycle, 3 op settings,
    21 sensors), one row per operational cycle.
    """
    data_dir = pathlib.Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = _raw_path(dataset, split, data_dir)
    # The files use one-or-more spaces as the separator and carry two trailing
    # spaces per line, which would otherwise create a spurious all-NaN column.
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)].copy()
    df.columns = COLUMNS
    # unit and cycle are integers; keep them that way for clean groupby output.
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_test_rul(
    dataset: str = "FD001",
    data_dir: pathlib.Path | str | None = None,
) -> pd.Series:
    """Load the official true-RUL vector for a test split.

    These are the *uncapped* ground-truth remaining cycles at each test unit's
    last recorded cycle, one value per unit, in unit order (1..N).

    Returns
    -------
    Series named ``true_rul`` indexed by ``unit`` (1-based).
    """
    data_dir = pathlib.Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = data_dir / f"RUL_{dataset}.txt"
    vals = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(int)
    vals.index = pd.RangeIndex(start=1, stop=len(vals) + 1, name="unit")
    vals.name = "true_rul"
    return vals


# --- Target construction -----------------------------------------------------


def add_training_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Add a capped RUL target column to a *training* frame.

    For each unit, RUL at a given cycle is ``max_cycle_for_unit - cycle`` (the
    unit runs to failure in the training set). The result is clipped at ``cap``
    when ``cap`` is not ``None``.

    Returns a new DataFrame with an added ``rul`` column; the input is not
    mutated.
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - df["cycle"]
    if cap is not None:
        rul = rul.clip(upper=cap)
    df["rul"] = rul.astype(int)
    return df


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unit: the row at that unit's maximum cycle.

    Used to score a test frame, where the prediction target is the RUL at each
    unit's final recorded cycle. Rows are returned sorted by unit.
    """
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    train = load_raw("FD001", "train")
    train = add_training_rul(train)
    test = load_raw("FD001", "test")
    rul = load_test_rul("FD001")
    print(f"train rows={len(train)} units={train['unit'].nunique()}")
    print(f"test rows={len(test)} units={test['unit'].nunique()}")
    print(f"true RUL values={len(rul)}  head={rul.head().tolist()}")
    print(f"train RUL: min={train['rul'].min()} max={train['rul'].max()} (cap={RUL_CAP})")
