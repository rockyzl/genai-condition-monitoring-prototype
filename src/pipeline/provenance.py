"""Artifact provenance and skip/idempotency logic.

Each stage output gets a ``<artifact>.prov.json`` sidecar recording the stage
name, the parameters that produced it, the SHA-256 of every input file, the
seed, the git SHA, and a timestamp. Provenance is the pipeline's state:

* **Skip.** A stage is *current* (skippable) when all of its declared outputs
  exist and their sidecars record the same input hashes + params + seed as the
  current run. ``--force`` overrides.
* **Audit.** The recorded git SHA + timestamp answer "which code produced this
  artifact, and when" — they are *not* part of the skip decision, so an unrelated
  commit never forces a rerun.

Determinism: same inputs + seed → same signature → the pipeline skips and the
artifacts stay byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROV_SUFFIX = ".prov.json"


def hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes, or ``"absent"`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return "absent"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(params: dict) -> str:
    """Stable JSON serialisation of params for hashing/comparison."""
    return json.dumps(params, sort_keys=True, default=str)


def git_sha(root: Path | None = None) -> str | None:
    """Short git SHA of the repo, or ``None`` when unavailable (not a repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def build_signature(
    stage: str,
    input_paths: list[Path],
    params: dict,
    seed: int,
    root: Path | None = None,
) -> dict:
    """Compute the provenance signature that identifies a stage's inputs+params.

    The ``input_hashes`` / ``params_hash`` / ``seed`` triple is the skip key;
    ``git_sha`` and ``timestamp`` are audit-only and excluded from comparison.
    """
    input_hashes = {str(p): hash_file(p) for p in input_paths}
    return {
        "stage": stage,
        "seed": seed,
        "params": params,
        "params_hash": hashlib.sha256(_canonical(params).encode()).hexdigest()[:16],
        "input_hashes": input_hashes,
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _skip_key(sig: dict) -> tuple:
    """The subset of a signature that decides skip vs rerun (audit fields dropped)."""
    return (
        sig.get("stage"),
        sig.get("seed"),
        sig.get("params_hash"),
        tuple(sorted(sig.get("input_hashes", {}).items())),
    )


def prov_path(artifact: Path) -> Path:
    """Sidecar path for an artifact (``foo.csv`` -> ``foo.csv.prov.json``)."""
    return Path(str(artifact) + PROV_SUFFIX)


def write_provenance(artifact: Path, sig: dict) -> Path:
    """Write the ``.prov.json`` sidecar next to an artifact."""
    p = prov_path(artifact)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sig, indent=2))
    return p


def read_provenance(artifact: Path) -> dict | None:
    """Read an artifact's sidecar, or ``None`` if missing/unreadable."""
    p = prov_path(artifact)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def is_stage_current(output_paths: list[Path], sig: dict) -> bool:
    """True when every output exists and its sidecar matches the skip key.

    A stage with no declared outputs is never current (always runs).
    """
    if not output_paths:
        return False
    want = _skip_key(sig)
    for out in output_paths:
        out = Path(out)
        if not out.exists():
            return False
        recorded = read_provenance(out)
        if recorded is None or _skip_key(recorded) != want:
            return False
    return True
