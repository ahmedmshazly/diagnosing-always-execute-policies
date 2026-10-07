from __future__ import annotations

"""Aggregate pipeline tests.

Covers the building blocks of ``scripts.aggregate``:
- percentile bootstrap CI on a known distribution converges to the sampling
  bounds of the sample mean within a tolerance.
- paired Wilcoxon on a synthetic delta returns the right p-value sign.
- the payload writer produces the expected file set and every metric has a
  complete per-agent + delta + wilcoxon entry.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate import (
    METRICS,
    REFLEX_NAME,
    UTILITY_NAME,
    bootstrap_ci,
    build_aggregate_payload,
    paired_wilcoxon,
    summarise,
    _write_aggregate_csv,
    _write_markdown,
    _write_wilcoxon_csv,
)


def test_bootstrap_ci_contains_true_mean_for_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=5.0, scale=2.0, size=200)
    lo, hi = bootstrap_ci(sample, rng=np.random.default_rng(1), num_resamples=2000)
    assert lo < 5.0 < hi
    assert lo < sample.mean() < hi


def test_bootstrap_ci_returns_nan_for_singleton():
    lo, hi = bootstrap_ci(np.array([3.0]), rng=np.random.default_rng(0), num_resamples=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_summarise_reports_standard_sample_std():
    rng = np.random.default_rng(2)
    values = rng.normal(size=50)
    stats = summarise(values, rng=np.random.default_rng(3), num_resamples=500)
    # sample std with ddof=1; summarise rounds to 6 decimals so compare at
    # that precision.
    assert abs(stats["std"] - float(values.std(ddof=1))) < 1e-6
    assert stats["n"] == 50


def test_paired_wilcoxon_detects_systematic_positive_delta():
    rng = np.random.default_rng(7)
    n = 40
    reflex = rng.normal(size=n)
    utility = reflex + 0.8  # systematic +0.8 shift
    result = paired_wilcoxon(reflex, utility)
    assert result["n_nonzero"] == n
    assert result["p_value"] < 1e-5
    assert result["mean_delta"] > 0.5


def test_paired_wilcoxon_handles_all_zero_deltas():
    reflex = np.zeros(10)
    utility = np.zeros(10)
    result = paired_wilcoxon(reflex, utility)
    assert result["n_nonzero"] == 0
    assert np.isnan(result["p_value"])


def _write_fake_metrics(run_dir: Path) -> None:
    fieldnames = [
        "seed",
        "agent_name",
        "alpha",
        "beta",
        "gamma",
        "num_jobs",
        "steps_executed",
        "completed_jobs",
        "failed_jobs",
        "completion_rate",
        "value_weighted_completion_rate",
        "uncapped_completion_rate",
        "failure_rate",
        "total_completed_value",
        "total_job_value",
        "total_compute_cost",
        "avg_compute_cost_per_step",
        "total_utility",
        "completed_all_jobs",
        "hit_step_budget",
    ]
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(12):
        # Reflex is the baseline; Utility shifts total_utility UP, shifts
        # completion_rate DOWN, shifts cost DOWN relative to Reflex.
        reflex_util = float(rng.normal(-50, 20))
        utility_util = reflex_util + float(rng.normal(15, 5))
        reflex_comp = float(rng.uniform(0.55, 0.7))
        utility_comp = reflex_comp - float(rng.uniform(0.05, 0.12))
        reflex_cost = float(rng.uniform(600, 800))
        utility_cost = reflex_cost - float(rng.uniform(100, 200))
        for agent, util, comp, cost in (
            (REFLEX_NAME, reflex_util, reflex_comp, reflex_cost),
            (UTILITY_NAME, utility_util, utility_comp, utility_cost),
        ):
            rows.append(
                {
                    "seed": seed,
                    "agent_name": agent,
                    "alpha": 1.0,
                    "beta": 0.1,
                    "gamma": 1.0,
                    "num_jobs": 100,
                    "steps_executed": 300,
                    "completed_jobs": int(round(comp * 100)),
                    "failed_jobs": 10,
                    "completion_rate": round(comp, 4),
                    "value_weighted_completion_rate": round(comp + 0.05, 4),
                    "uncapped_completion_rate": round(min(comp + 0.2, 0.95), 4),
                    "failure_rate": 0.1,
                    "total_completed_value": round(comp * 300, 4),
                    "total_job_value": 300.0,
                    "total_compute_cost": round(cost, 4),
                    "avg_compute_cost_per_step": round(cost / 300, 4),
                    "total_utility": round(util, 4),
                    "completed_all_jobs": False,
                    "hit_step_budget": True,
                }
            )
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_aggregate_payload_covers_every_metric(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_fake_metrics(run_dir)
    payload = build_aggregate_payload(run_dir=run_dir, num_resamples=400, bootstrap_seed=11)
    assert payload["num_seeds"] == 12
    for metric in METRICS:
        assert metric in payload["per_agent"][REFLEX_NAME]
        assert metric in payload["per_agent"][UTILITY_NAME]
        assert metric in payload["deltas_utility_minus_reflex"]
        assert metric in payload["paired_wilcoxon"]
    # Synthetic fixture puts Utility above Reflex on utility; signed-rank
    # should flag that as significant.
    util_test = payload["paired_wilcoxon"]["total_utility"]
    assert util_test["mean_delta"] > 0
    assert util_test["p_value"] < 0.05


def test_write_helpers_produce_expected_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_fake_metrics(run_dir)
    payload = build_aggregate_payload(run_dir=run_dir, num_resamples=200, bootstrap_seed=22)
    _write_aggregate_csv(run_dir / "aggregate.csv", payload)
    _write_wilcoxon_csv(run_dir / "wilcoxon.csv", payload)
    _write_markdown(run_dir / "aggregate.md", payload)
    assert (run_dir / "aggregate.csv").exists()
    assert (run_dir / "wilcoxon.csv").exists()
    md = (run_dir / "aggregate.md").read_text(encoding="utf-8")
    # Every metric name is present in the rendered markdown table.
    for metric in METRICS:
        assert metric in md
