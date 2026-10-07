from __future__ import annotations

"""Phase-3 (alpha, beta, gamma) sweep driver.

Iterates the cross product of ``cfg.sweep.alpha_grid``,
``cfg.sweep.beta_grid``, and ``cfg.sweep.gamma_grid`` plus every
``cfg.sweep.reference_points`` entry that is not already on the grid. For
each cell and every seed in ``cfg.sweep.seeds`` it runs the Utility-Based
baseline with those scoring weights and re-scores the episode under the
shared ``cfg.sweep.evaluation_utility`` so cross-cell comparisons are fair.

Reflex is also run once on the same seed pool. Reflex is weight-independent
but we want a paired reference point on every plot.

Outputs (written into ``<out_root>/<run_id>/``):
- ``sweep.csv``        — per-(cell, seed) rows with the Phase-3 schema.
- ``cells.csv``        — per-cell aggregate statistics (means across seeds).
- ``reflex.csv``        — per-seed Reflex rows + a one-row aggregate.
- ``config.yaml``       — resolved config.
- ``run_manifest.json`` — usual reproducibility block.
"""

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.config import (
    RunConfig,
    SweepReferencePoint,
    build_run_config,
    load_config,
    override_utility_weights,
)
from src.cost import cost as cost_fn
from src.reflex_agent import build_reflex_agent
from src.runner import run_many_episodes
from src.run_artifacts import (
    ensure_run_dir,
    generate_run_id,
    write_manifest,
    write_resolved_config,
)
from src.utility_agent import build_utility_agent


SWEEP_CSV_FIELDS: Tuple[str, ...] = (
    "alpha",
    "beta",
    "gamma",
    "cell_id",
    "is_reference",
    "reference_name",
    "seed",
    "agent_name",
    "mean_utility",              # U under the eval weights for this episode
    "cell_utility",              # U under the cell's (alpha, beta, gamma)
    "completion_rate",
    "value_weighted_completion_rate",
    "uncapped_completion_rate",
    "cost",
    "failure_rate",
    "total_completed_value",
    "total_job_value",
    "completed_jobs",
    "failed_jobs",
    "steps_executed",
    "hit_step_budget",
)

CELLS_CSV_FIELDS: Tuple[str, ...] = (
    "alpha",
    "beta",
    "gamma",
    "cell_id",
    "is_reference",
    "reference_name",
    "num_seeds",
    "mean_utility_mean",
    "mean_utility_std",
    "cell_utility_mean",
    "completion_rate_mean",
    "value_weighted_completion_rate_mean",
    "uncapped_completion_rate_mean",
    "cost_mean",
    "failure_rate_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-3 (alpha, beta, gamma) sweep for the Utility-Based baseline."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--include-uncapped",
        action="store_true",
        help=(
            "Also run the 900-step uncapped replay for every cell (slower). "
            "Off by default because Pareto analysis uses the capped metrics."
        ),
    )
    parser.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help="Optional cap on cell count (for smoke tests).",
    )
    return parser.parse_args()


def _cell_key(alpha: float, beta: float, gamma: float) -> str:
    return f"alpha={alpha:g}_beta={beta:g}_gamma={gamma:g}"


def _build_cells(cfg: RunConfig) -> List[Dict[str, Any]]:
    """Return the ordered list of cells to run.

    Grid cells come first; reference points that are not on the grid are
    appended at the end so plot legends can colour them distinctly.
    """
    sweep = cfg.sweep
    cells: List[Dict[str, Any]] = []
    grid_keys: set[str] = set()

    for alpha, beta, gamma in itertools.product(sweep.alpha_grid, sweep.beta_grid, sweep.gamma_grid):
        key = _cell_key(float(alpha), float(beta), float(gamma))
        grid_keys.add(key)
        cells.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "cell_id": key,
                "is_reference": False,
                "reference_name": "",
            }
        )

    for point in sweep.reference_points:
        key = _cell_key(point.alpha, point.beta, point.gamma)
        on_grid = key in grid_keys
        cells.append(
            {
                "alpha": float(point.alpha),
                "beta": float(point.beta),
                "gamma": float(point.gamma),
                "cell_id": key,
                "is_reference": True,
                "reference_name": point.name,
                "on_grid": on_grid,
            }
        )

    # Keep grid cells + append *only* off-grid reference points; on-grid
    # reference points are already represented, but we keep the metadata by
    # dropping the duplicate row and promoting the grid row to is_reference.
    final_cells: List[Dict[str, Any]] = []
    name_by_key: Dict[str, str] = {}
    for cell in cells:
        if cell["is_reference"] and cell.get("on_grid"):
            name_by_key[cell["cell_id"]] = cell["reference_name"]
    seen: set[str] = set()
    for cell in cells:
        if cell.get("on_grid"):
            continue
        if cell["cell_id"] in seen:
            continue
        if cell["cell_id"] in name_by_key and not cell["is_reference"]:
            cell = {**cell, "is_reference": True, "reference_name": name_by_key[cell["cell_id"]]}
        final_cells.append(cell)
        seen.add(cell["cell_id"])

    return final_cells


def _episode_utility(
    completed_value: float,
    compute_cost: float,
    failed_jobs: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    return (alpha * completed_value) - (beta * compute_cost) - (gamma * failed_jobs)


def _run_cell(
    cfg: RunConfig,
    alpha: float,
    beta: float,
    gamma: float,
    include_uncapped: bool,
) -> List[Any]:
    cell_cfg = override_utility_weights(cfg, alpha=alpha, beta=beta, gamma=gamma)
    metrics = run_many_episodes(
        cfg=cell_cfg,
        agent_factory=build_utility_agent,
        seeds=list(cfg.sweep.seeds),
        include_uncapped=include_uncapped,
    )
    return metrics


def _run_reflex(cfg: RunConfig, include_uncapped: bool) -> List[Any]:
    return run_many_episodes(
        cfg=cfg,
        agent_factory=build_reflex_agent,
        seeds=list(cfg.sweep.seeds),
        include_uncapped=include_uncapped,
    )


def _episode_row(
    cell: Dict[str, Any],
    metric,
    eval_weights,
) -> Dict[str, Any]:
    d = metric.as_dict()
    eval_utility = _episode_utility(
        completed_value=d["total_completed_value"],
        compute_cost=d["total_compute_cost"],
        failed_jobs=d["failed_jobs"],
        alpha=eval_weights.alpha,
        beta=eval_weights.beta,
        gamma=eval_weights.gamma,
    )
    return {
        "alpha": cell["alpha"],
        "beta": cell["beta"],
        "gamma": cell["gamma"],
        "cell_id": cell["cell_id"],
        "is_reference": bool(cell["is_reference"]),
        "reference_name": cell.get("reference_name", ""),
        "seed": d["seed"],
        "agent_name": d["agent_name"],
        "mean_utility": round(eval_utility, 6),
        "cell_utility": round(d["total_utility"], 6),
        "completion_rate": d["completion_rate"],
        "value_weighted_completion_rate": d["value_weighted_completion_rate"],
        "uncapped_completion_rate": d["uncapped_completion_rate"],
        "cost": d["total_compute_cost"],
        "failure_rate": d["failure_rate"],
        "total_completed_value": d["total_completed_value"],
        "total_job_value": d["total_job_value"],
        "completed_jobs": d["completed_jobs"],
        "failed_jobs": d["failed_jobs"],
        "steps_executed": d["steps_executed"],
        "hit_step_budget": d["hit_step_budget"],
    }


def _cell_aggregate(cell: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    mean_utility = np.array([row["mean_utility"] for row in rows], dtype=float)
    cell_utility = np.array([row["cell_utility"] for row in rows], dtype=float)
    completion = np.array([row["completion_rate"] for row in rows], dtype=float)
    vwc = np.array([row["value_weighted_completion_rate"] for row in rows], dtype=float)
    uncapped = np.array([row["uncapped_completion_rate"] for row in rows], dtype=float)
    cost = np.array([row["cost"] for row in rows], dtype=float)
    failure = np.array([row["failure_rate"] for row in rows], dtype=float)
    return {
        "alpha": cell["alpha"],
        "beta": cell["beta"],
        "gamma": cell["gamma"],
        "cell_id": cell["cell_id"],
        "is_reference": bool(cell["is_reference"]),
        "reference_name": cell.get("reference_name", ""),
        "num_seeds": len(rows),
        "mean_utility_mean": round(float(mean_utility.mean()), 6),
        "mean_utility_std": round(float(mean_utility.std(ddof=1)) if len(rows) > 1 else 0.0, 6),
        "cell_utility_mean": round(float(cell_utility.mean()), 6),
        "completion_rate_mean": round(float(completion.mean()), 6),
        "value_weighted_completion_rate_mean": round(float(vwc.mean()), 6),
        "uncapped_completion_rate_mean": round(float(uncapped.mean()), 6),
        "cost_mean": round(float(cost.mean()), 6),
        "failure_rate_mean": round(float(failure.mean()), 6),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_root = Path(args.out_root) if args.out_root else Path(cfg.experiment.out_root)
    run_id = args.run_id or generate_run_id("sweep_phase3")
    run_dir = ensure_run_dir(out_root, run_id)
    started = time.time()

    eval_weights = cfg.sweep.evaluation_utility
    cells = _build_cells(cfg)
    if args.limit_cells is not None:
        cells = cells[: args.limit_cells]
    num_cells = len(cells)

    print(
        f"Phase-3 sweep: {num_cells} cells x {len(cfg.sweep.seeds)} seeds "
        f"= {num_cells * len(cfg.sweep.seeds)} Utility-Based episodes"
    )
    print(f"Eval utility: alpha={eval_weights.alpha} beta={eval_weights.beta} gamma={eval_weights.gamma}")

    sweep_rows: List[Dict[str, Any]] = []
    cells_rows: List[Dict[str, Any]] = []

    for index, cell in enumerate(cells, start=1):
        metrics_list = _run_cell(
            cfg,
            alpha=cell["alpha"],
            beta=cell["beta"],
            gamma=cell["gamma"],
            include_uncapped=args.include_uncapped,
        )
        cell_rows = [_episode_row(cell, m, eval_weights) for m in metrics_list]
        sweep_rows.extend(cell_rows)
        cells_rows.append(_cell_aggregate(cell, cell_rows))
        print(
            f"  [{index:3d}/{num_cells}] {cell['cell_id']:<30s} "
            f"mean_U={cells_rows[-1]['mean_utility_mean']:+9.3f}  "
            f"comp={cells_rows[-1]['completion_rate_mean']:.3f}  "
            f"cost={cells_rows[-1]['cost_mean']:6.1f}  "
            f"fail={cells_rows[-1]['failure_rate_mean']:.3f}"
            + (" [ref]" if cell["is_reference"] else "")
        )

    reflex_metrics = _run_reflex(cfg, include_uncapped=args.include_uncapped)
    reflex_cell = {
        "alpha": float("nan"),
        "beta": float("nan"),
        "gamma": float("nan"),
        "cell_id": "reflex",
        "is_reference": True,
        "reference_name": "reflex",
    }
    reflex_rows = [_episode_row(reflex_cell, m, eval_weights) for m in reflex_metrics]
    reflex_aggregate = _cell_aggregate(reflex_cell, reflex_rows)

    # Write main sweep.csv (Utility-Based cells only).
    _write_csv(run_dir / "sweep.csv", sweep_rows, SWEEP_CSV_FIELDS)
    _write_csv(run_dir / "cells.csv", cells_rows, CELLS_CSV_FIELDS)
    _write_csv(run_dir / "reflex.csv", reflex_rows, SWEEP_CSV_FIELDS)
    _write_csv(run_dir / "reflex_aggregate.csv", [reflex_aggregate], CELLS_CSV_FIELDS)

    write_resolved_config(run_dir, cfg)
    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        seeds=list(cfg.sweep.seeds),
        extra={
            "entrypoint": "sweep_phase3",
            "num_cells": num_cells,
            "num_seeds": len(cfg.sweep.seeds),
            "include_uncapped": bool(args.include_uncapped),
            "eval_utility": {
                "alpha": eval_weights.alpha,
                "beta": eval_weights.beta,
                "gamma": eval_weights.gamma,
            },
        },
        wall_clock_start=started,
    )
    print(
        f"Reflex ref:   mean_U={reflex_aggregate['mean_utility_mean']:+9.3f}  "
        f"comp={reflex_aggregate['completion_rate_mean']:.3f}  "
        f"cost={reflex_aggregate['cost_mean']:6.1f}  "
        f"fail={reflex_aggregate['failure_rate_mean']:.3f}"
    )
    print(f"Wrote sweep run to: {run_dir}")


if __name__ == "__main__":
    main()
