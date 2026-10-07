from __future__ import annotations

"""Reproducible run-artifact writer.

Every experiment (baseline comparison, sweep, RL training, held-out eval)
writes into ``<out_root>/<run_id>/`` with the following minimum contents:

- ``config.yaml``       — the fully resolved configuration actually used.
- ``run_manifest.json`` — commit SHA, config hash, seed list, wall-clock,
                          library versions, hostname, platform.
- ``metrics.csv``       — long-form per-episode rows.

Experiment drivers may add extra files (e.g. ``summary.json``, ``report.md``,
``sweep_grid.csv``), but the three files above are required.
"""

import csv
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .config import RunConfig, _canonical_yaml_dump


def generate_run_id(prefix: str = "run") -> str:
    """Return a default ``<prefix>_YYYYmmddTHHMMSSZ`` id."""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{now}"


def ensure_run_dir(out_root: Path, run_id: str) -> Path:
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(run_dir: Path, cfg: RunConfig) -> Path:
    """Write the resolved YAML used by this run."""
    path = run_dir / "config.yaml"
    path.write_text(_canonical_yaml_dump(cfg.raw), encoding="utf-8")
    return path


def _git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:  # noqa: BLE001 - we must not fail a run on env issues
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "yaml", "pytest", "matplotlib"):
        try:
            imported = __import__(mod)
            versions[mod] = getattr(imported, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[mod] = "missing"
    return versions


def write_manifest(
    run_dir: Path,
    cfg: RunConfig,
    seeds: Sequence[int],
    extra: Mapping[str, Any] | None = None,
    wall_clock_start: float | None = None,
) -> Path:
    """Write ``run_manifest.json`` for the current run."""
    now_iso = datetime.now(timezone.utc).isoformat()
    manifest: Dict[str, Any] = {
        "run_id": run_dir.name,
        "commit_sha": _git_commit_sha(),
        "git_dirty": _git_is_dirty(),
        "config_hash": cfg.config_hash(),
        "seed_list": list(int(seed) for seed in seeds),
        "wall_clock": {
            "start": datetime.fromtimestamp(
                wall_clock_start, tz=timezone.utc
            ).isoformat() if wall_clock_start else now_iso,
            "end": now_iso,
            "elapsed_seconds": (
                None if wall_clock_start is None else round(time.time() - wall_clock_start, 3)
            ),
        },
        "library_versions": _library_versions(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    if extra:
        manifest["extra"] = dict(extra)
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_metrics_csv(run_dir: Path, rows: Iterable[Dict[str, Any]], filename: str = "metrics.csv") -> Path:
    """Write per-seed metrics rows; every row must share the same schema."""
    rows_list: List[Dict[str, Any]] = list(rows)
    path = run_dir / filename
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)
    return path


def write_json(run_dir: Path, filename: str, payload: Any) -> Path:
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_text(run_dir: Path, filename: str, content: str) -> Path:
    path = run_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


__all__ = [
    "ensure_run_dir",
    "generate_run_id",
    "write_json",
    "write_manifest",
    "write_metrics_csv",
    "write_resolved_config",
    "write_text",
]
