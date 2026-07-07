"""First-boot bootstrap for the public web demo.

On a fresh clone / fresh Hugging Face Space the repo ships **no data, no model,
no processed artifacts** (all gitignored). This script makes the demo runnable
from nothing:

1. Download the NASA C-MAPSS turbofan zip from the public S3 mirror (~12 MB),
   with transfer-size + integrity + checksum sanity checks.
2. Extract the FD001 train/test/RUL files (descending into the nested
   ``CMAPSSData.zip``) into ``data/raw/CMAPSSData/``.
3. Run the full 10-stage pipeline in-process (equivalent to
   ``python -m src.pipeline run --all``), which trains the model and writes every
   downstream artifact. Provenance makes this idempotent — a second run skips.

Standalone:  ``.venv/bin/python scripts/bootstrap_demo.py [--force]``
Importable:  ``from scripts.bootstrap_demo import bootstrap, artifacts_present``

Top-level imports are stdlib-only so importing this module stays cheap (the heavy
pandas/sklearn pipeline is imported lazily inside :func:`run_pipeline_stages`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Public NASA S3 mirror of the C-MAPSS "Turbofan Engine Degradation" dataset.
CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

RAW_DIR = ROOT / "data" / "raw" / "CMAPSSData"
FD001_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")

#: Transfer-size sanity window (bytes). The mirror zip is ~12.4 MB.
MIN_ZIP_BYTES = 5_000_000
MAX_ZIP_BYTES = 60_000_000

#: SHA-256 of the mirror zip observed on 2026-07-06. Soft-checked: a mismatch
#: warns (mirrors can be re-packed) but does not abort — the extracted-file shape
#: validation below is the real integrity guard. Pass ``--strict-checksum`` to
#: turn the mismatch into a hard error.
EXPECTED_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"

#: Artifacts that must all exist for the demo app to run without bootstrapping.
REQUIRED_ARTIFACTS = (
    ROOT / "data" / "processed" / "test_predictions.csv",
    ROOT / "models" / "rul_baseline.joblib",
    ROOT / "reports" / "metrics_model.json",
    ROOT / "data" / "processed" / "evidence",  # directory, must be non-empty
)


# --- state checks ------------------------------------------------------------
def raw_present() -> bool:
    """True when all three FD001 raw files are already extracted."""
    return all((RAW_DIR / name).exists() for name in FD001_FILES)


def artifacts_present() -> bool:
    """True when every required pipeline artifact exists (evidence dir non-empty)."""
    for p in REQUIRED_ARTIFACTS:
        if p.name == "evidence":
            if not p.is_dir() or not any(p.glob("unit_*.json")):
                return False
        elif not p.exists():
            return False
    return True


# --- download + extract ------------------------------------------------------
def _download(url: str, dest: Path, log, progress=None) -> Path:
    """Stream ``url`` to ``dest`` with a progress callback and size sanity."""
    log(f"downloading C-MAPSS dataset (~12 MB) from {url.split('/')[2]} …")
    req = urllib.request.Request(url, headers={"User-Agent": "cm-demo-bootstrap/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        got = 0
        chunks = bytearray()
        while True:
            chunk = resp.read(262144)
            if not chunk:
                break
            chunks.extend(chunk)
            got += len(chunk)
            if progress and total:
                progress(got / total)
    if total and got != total:
        raise IOError(f"incomplete download: got {got} of {total} bytes")
    if not (MIN_ZIP_BYTES <= got <= MAX_ZIP_BYTES):
        raise IOError(
            f"downloaded size {got} bytes outside sane range "
            f"[{MIN_ZIP_BYTES}, {MAX_ZIP_BYTES}] — refusing to trust it"
        )
    dest.write_bytes(bytes(chunks))
    log(f"downloaded {got:,} bytes")
    return dest


def _check_checksum(zip_path: Path, log, strict: bool) -> None:
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest == EXPECTED_SHA256:
        log(f"checksum OK (sha256 {digest[:12]}…)")
        return
    msg = (
        f"checksum mismatch: expected {EXPECTED_SHA256[:12]}… got {digest[:12]}… — "
        "the mirror zip may have been re-packed"
    )
    if strict:
        raise IOError(msg)
    log(f"WARNING: {msg}; continuing (extracted-file shape is validated next)")


def _extract_targets(zf: zipfile.ZipFile, targets: set[str], out_dir: Path, log) -> set[str]:
    """Extract ``targets`` (by basename) from a zip, descending into nested zips."""
    found: set[str] = set()
    for info in zf.infolist():
        if info.is_dir():
            continue
        base = os.path.basename(info.filename)
        if base in targets:
            (out_dir / base).write_bytes(zf.read(info))
            found.add(base)
            log(f"extracted {base}")
        elif base.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                found |= _extract_targets(inner, targets - found, out_dir, log)
    return found


def _validate_extracted(log) -> None:
    """Shape-check the extracted FD001 files (the real integrity guard)."""
    import pandas as pd

    train = pd.read_csv(RAW_DIR / "train_FD001.txt", sep=r"\s+", header=None)
    rul = pd.read_csv(RAW_DIR / "RUL_FD001.txt", sep=r"\s+", header=None)
    n_units = train.iloc[:, 0].nunique()
    if train.shape[1] < 26 or n_units != 100:
        raise IOError(
            f"train_FD001.txt failed shape check: cols={train.shape[1]} "
            f"units={n_units} (expected >=26 cols, 100 units)"
        )
    if len(rul) != 100:
        raise IOError(f"RUL_FD001.txt failed shape check: {len(rul)} rows (expected 100)")
    log(f"validated FD001: {len(train):,} train rows, {n_units} units")


def ensure_data(force: bool, log, progress=None, strict_checksum: bool = False) -> dict:
    """Ensure the FD001 raw files exist; download+extract if missing (or forced)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not force and raw_present():
        log("C-MAPSS FD001 files already present — skipping download")
        return {"downloaded": False, "extracted": False}

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = _download(CMAPSS_URL, Path(tmp) / "cmapss.zip", log, progress)
        if not zipfile.is_zipfile(zip_path):
            raise IOError("downloaded file is not a valid zip archive")
        _check_checksum(zip_path, log, strict=strict_checksum)
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise IOError(f"corrupt entry in zip: {bad}")
            found = _extract_targets(zf, set(FD001_FILES), RAW_DIR, log)
    missing = set(FD001_FILES) - found
    if missing:
        raise IOError(f"FD001 files not found in archive: {sorted(missing)}")
    _validate_extracted(log)
    return {"downloaded": True, "extracted": True}


# --- pipeline ----------------------------------------------------------------
def run_pipeline_stages(force: bool, log) -> dict:
    """Run the full 10-stage pipeline in-process (idempotent via provenance)."""
    from src.pipeline.config import PipelineConfig
    from src.pipeline.runner import run_pipeline
    from src.pipeline.specs import STAGE_ORDER

    log("running 10-stage pipeline (train + predict + evidence + diagnose + eval) …")
    cfg = PipelineConfig.load()
    ctx = run_pipeline(cfg, list(STAGE_ORDER), force=force)
    ran = sum(1 for r in ctx.results if not r.skipped)
    skipped = sum(1 for r in ctx.results if r.skipped)
    log(f"pipeline done: {ran} stage(s) ran, {skipped} skipped")
    return {"stages_run": ran, "stages_skipped": skipped}


# --- orchestration -----------------------------------------------------------
def bootstrap(
    force: bool = False,
    log=print,
    progress=None,
    run_pipeline_after: bool = True,
    strict_checksum: bool = False,
) -> dict:
    """Full first-boot bootstrap. Idempotent: a no-op when everything exists.

    Returns a summary dict: ``downloaded`` / ``extracted`` / ``stages_run`` /
    ``stages_skipped`` / ``skipped_all``.
    """
    summary = {
        "downloaded": False,
        "extracted": False,
        "stages_run": 0,
        "stages_skipped": 0,
        "skipped_all": False,
    }
    if not force and artifacts_present():
        log("all demo artifacts already present — nothing to bootstrap")
        summary["skipped_all"] = True
        return summary

    summary.update(ensure_data(force, log, progress, strict_checksum))
    if run_pipeline_after:
        summary.update(run_pipeline_stages(force, log))
    log("bootstrap complete ✅")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download + rerun all stages")
    parser.add_argument(
        "--strict-checksum",
        action="store_true",
        help="fail (not warn) if the downloaded zip checksum differs",
    )
    args = parser.parse_args(argv)
    try:
        summary = bootstrap(force=args.force, strict_checksum=args.strict_checksum)
    except Exception as exc:  # surface a clean message for the CLI/Space logs
        print(f"[bootstrap] FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"[bootstrap] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
