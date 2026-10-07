from __future__ import annotations

"""Guard: the midterm numerics must remain reproducible.

The committed file ``results/canonical_midterm/metrics.csv`` is the
reference 10-seed baseline produced under the midterm weights (α, β, γ) =
(1.0, 0.4, 0.8). A clean refactor must leave its per-seed
``total_utility`` and ``total_compute_cost`` bit-for-bit unchanged when
re-run under ``config/midterm_weights.yaml``.

This test is fast (~10 seconds) because it only uses a subset of seeds;
the full 10-seed comparison is exercised by ``make baseline`` at CI time.
"""

import csv
from pathlib import Path

import pytest

from src.config import load_config
from src.reflex_agent import build_reflex_agent
from src.runner import run_episode
from src.utility_agent import build_utility_agent


CANONICAL_CSV = Path("results/canonical_midterm/metrics.csv")


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="reference run missing")
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_canonical_reflex_metrics_reproduce_under_midterm_weights(seed):
    reference = _load_reference(CANONICAL_CSV)
    key = (seed, "Reflex Agent")
    if key not in reference:
        pytest.skip(f"no reference row for {key}")
    cfg = load_config(Path("config/midterm_weights.yaml"))
    metrics = run_episode(cfg=cfg, agent_factory=build_reflex_agent, seed=seed)
    _assert_row_matches(metrics, reference[key])


@pytest.mark.skipif(not CANONICAL_CSV.exists(), reason="reference run missing")
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_canonical_utility_metrics_reproduce_under_midterm_weights(seed):
    reference = _load_reference(CANONICAL_CSV)
    key = (seed, "Utility-Based Agent (Non-Learning Baseline)")
    if key not in reference:
        pytest.skip(f"no reference row for {key}")
    cfg = load_config(Path("config/midterm_weights.yaml"))
    metrics = run_episode(cfg=cfg, agent_factory=build_utility_agent, seed=seed)
    _assert_row_matches(metrics, reference[key])


def _load_reference(path: Path) -> dict:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["seed"]), row["agent_name"])] = row
    return rows


def _assert_row_matches(metrics, reference):
    d = metrics.as_dict()
    # Use CSV-rounded fields — they are the canonical reference precision.
    assert float(reference["total_utility"]) == d["total_utility"], (
        f"total_utility drift for {reference['seed']}/{reference['agent_name']}: "
        f"{reference['total_utility']} → {d['total_utility']}"
    )
    assert float(reference["total_compute_cost"]) == d["total_compute_cost"], (
        f"total_compute_cost drift for {reference['seed']}/{reference['agent_name']}: "
        f"{reference['total_compute_cost']} → {d['total_compute_cost']}"
    )
    assert int(reference["completed_jobs"]) == d["completed_jobs"]
    assert int(reference["failed_jobs"]) == d["failed_jobs"]
