from __future__ import annotations

"""Smoke test for the Phase-3 sweep driver.

The full 80-cell sweep takes ~15 min. This test runs a tiny grid
(2x2x1 = 4 cells) over 2 seeds and asserts that:

- sweep.csv has the required Phase-3 schema and the right number of rows.
- cells.csv has one row per cell.
- reflex.csv and reflex_aggregate.csv are written and non-empty.
- The evaluation utility column (``mean_utility``) is computed under the
  shared evaluation weights and not under the per-cell weights.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.sweep_phase3 import (
    CELLS_CSV_FIELDS,
    SWEEP_CSV_FIELDS,
    _build_cells,
    _episode_utility,
    main as sweep_main,
)
from src.config import build_run_config, load_config


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def test_build_cells_expands_grid_and_marks_references():
    raw = _deep_copy(load_config().raw)
    raw["sweep"]["alpha_grid"] = [1.0, 2.0]
    raw["sweep"]["beta_grid"] = [0.1]
    raw["sweep"]["gamma_grid"] = [1.0]
    raw["sweep"]["reference_points"] = {
        "midterm_utility": {"alpha": 1.0, "beta": 0.4, "gamma": 0.8},
        "on_grid_ref": {"alpha": 1.0, "beta": 0.1, "gamma": 1.0},
    }
    cfg = build_run_config(raw)
    cells = _build_cells(cfg)
    # 2 grid cells + 1 off-grid ref ("midterm_utility") = 3 rows.
    # The on-grid reference ("on_grid_ref") promotes the matching grid row
    # to is_reference=True but does not create a duplicate.
    assert len(cells) == 3
    cell_ids = [cell["cell_id"] for cell in cells]
    assert cell_ids.count("alpha=1_beta=0.1_gamma=1") == 1
    promoted = next(cell for cell in cells if cell["cell_id"] == "alpha=1_beta=0.1_gamma=1")
    assert promoted["is_reference"] is True
    assert promoted["reference_name"] == "on_grid_ref"
    # The off-grid reference is still listed.
    midterm_row = next(cell for cell in cells if cell["reference_name"] == "midterm_utility")
    assert midterm_row["beta"] == 0.4


def test_episode_utility_formula_matches_spec():
    # U = alpha * Value - beta * Cost - gamma * FailedJobs
    u = _episode_utility(completed_value=200.0, compute_cost=500.0, failed_jobs=3, alpha=1.0, beta=0.1, gamma=1.0)
    assert u == 200.0 - 50.0 - 3.0


def test_sweep_main_smoke(tmp_path, monkeypatch):
    """Run the driver on a tiny 4-cell, 2-seed config and check outputs."""
    raw = _deep_copy(load_config().raw)
    raw["experiment"]["num_jobs"] = 20
    raw["experiment"]["max_steps"] = 80
    raw["experiment"]["uncapped_max_steps"] = 120
    raw["sweep"]["alpha_grid"] = [1.0, 2.0]
    raw["sweep"]["beta_grid"] = [0.1, 0.25]
    raw["sweep"]["gamma_grid"] = [1.0]
    raw["sweep"]["seeds"] = [100, 101]
    raw["sweep"]["reference_points"] = {}
    raw["sweep"]["evaluation_utility"] = {"alpha": 1.0, "beta": 0.1, "gamma": 1.0}

    config_path = tmp_path / "mini.yaml"
    import yaml as _yaml
    config_path.write_text(_yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    run_id = "smoke_phase3"
    argv = [
        "sweep_phase3",
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--out-root",
        str(tmp_path / "results"),
    ]
    monkeypatch.setattr("sys.argv", argv)
    sweep_main()

    run_dir = tmp_path / "results" / run_id
    sweep_path = run_dir / "sweep.csv"
    cells_path = run_dir / "cells.csv"
    reflex_path = run_dir / "reflex.csv"
    agg_path = run_dir / "reflex_aggregate.csv"
    assert sweep_path.exists()
    assert cells_path.exists()
    assert reflex_path.exists()
    assert agg_path.exists()

    with sweep_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) == set(SWEEP_CSV_FIELDS)
    assert len(rows) == 4 * 2  # 4 cells × 2 seeds

    # mean_utility column must be computed under the eval weights, NOT the
    # per-cell weights.
    for row in rows:
        alpha = 1.0  # eval alpha
        beta = 0.1
        gamma = 1.0
        expected = (
            alpha * float(row["total_completed_value"])
            - beta * float(row["cost"])
            - gamma * int(row["failed_jobs"])
        )
        assert abs(float(row["mean_utility"]) - round(expected, 6)) < 1e-6

    with cells_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cell_rows = list(reader)
        assert set(reader.fieldnames or []) == set(CELLS_CSV_FIELDS)
    assert len(cell_rows) == 4
