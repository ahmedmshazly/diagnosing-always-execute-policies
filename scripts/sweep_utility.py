from __future__ import annotations

"""(alpha, beta, gamma) sweep scaffold.

This scaffold exists so that ``make sweep`` is wired end-to-end today and so
the output layout is pinned before the Pareto study lands. It reads the grid
from ``config.sweep``, iterates the cross-product, and writes per-cell
metrics under ``<out_root>/<run_id>/cells/alpha=..._beta=..._gamma=.../``.

The full Wilcoxon + Pareto analysis arrives in a follow-up commit. What this
script guarantees today:

- Every cell run produces a ``metrics.csv`` with the same schema the
  baseline driver uses.
- A top-level ``sweep_grid.csv`` indexes the cells.
- ``config.yaml`` and ``run_manifest.json`` still live at the run root.
"""

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import List

from src.compare_baselines import run_comparison
from src.config import load_config, override_utility_weights
from src.run_artifacts import (
    ensure_run_dir,
    generate_run_id,
    write_json,
    write_manifest,
    write_metrics_csv,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--seed-group",
        choices=("train", "test", "sweep"),
        default="sweep",
        help="'sweep' uses config.sweep.seeds; 'train'/'test' use the main split.",
    )
    return parser.parse_args()


def _seeds_for(cfg, seed_group: str):
    if seed_group == "train":
        return list(cfg.seeds.train)
    if seed_group == "test":
        return list(cfg.seeds.test)
    return list(cfg.sweep.seeds)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    seeds = _seeds_for(cfg, args.seed_group)
    out_root = Path(args.out_root) if args.out_root else Path(cfg.experiment.out_root)
    run_id = args.run_id or generate_run_id("sweep")
    run_dir = ensure_run_dir(out_root, run_id)

    started = time.time()
    write_resolved_config(run_dir, cfg)
    grid: List[dict] = []

    for alpha, beta, gamma in itertools.product(
        cfg.sweep.alpha_grid, cfg.sweep.beta_grid, cfg.sweep.gamma_grid
    ):
        cell_cfg = override_utility_weights(cfg, alpha=alpha, beta=beta, gamma=gamma)
        cell_id = f"alpha={alpha}_beta={beta}_gamma={gamma}"
        cell_dir = run_dir / "cells" / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        results = run_comparison(cfg=cell_cfg, seeds=seeds)
        write_resolved_config(cell_dir, cell_cfg)
        write_metrics_csv(cell_dir, results["all_rows"])
        write_json(
            cell_dir,
            "summary.json",
            {
                "reflex_summary": results["reflex_summary"],
                "utility_summary": results["utility_summary"],
                "head_to_head": results["comparison"],
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "seeds": seeds,
            },
        )
        grid.append(
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "cell_id": cell_id,
                "utility_mean": results["utility_summary"]["mean_total_utility"],
                "reflex_mean": results["reflex_summary"]["mean_total_utility"],
                "utility_mean_completion": results["utility_summary"]["mean_completion_rate"],
                "reflex_mean_completion": results["reflex_summary"]["mean_completion_rate"],
                "utility_mean_cost": results["utility_summary"]["mean_total_compute_cost"],
                "reflex_mean_cost": results["reflex_summary"]["mean_total_compute_cost"],
                "utility_mean_failure": results["utility_summary"]["mean_failure_rate"],
                "reflex_mean_failure": results["reflex_summary"]["mean_failure_rate"],
            }
        )
        print(f"[{cell_id}] done.")

    grid_path = run_dir / "sweep_grid.csv"
    if grid:
        with grid_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(grid[0].keys()))
            writer.writeheader()
            writer.writerows(grid)

    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        seeds=seeds,
        extra={
            "entrypoint": "sweep_utility",
            "seed_group": args.seed_group,
            "num_cells": len(grid),
        },
        wall_clock_start=started,
    )
    print(f"Wrote sweep run to: {run_dir}")


if __name__ == "__main__":
    main()
