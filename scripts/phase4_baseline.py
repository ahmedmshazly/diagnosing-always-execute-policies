from __future__ import annotations

"""Phase-4 extended baseline comparison: Reflex + full UB + stripped UB.

Extends the Phase-2 n=50 harness (scripts.aggregate) to include the
Stripped Utility-Based Agent as a third column. The RL agent joins as a
fourth column in Phase 5. This script writes the extended per-seed CSV and
then re-uses scripts.aggregate to produce matching aggregate.{csv,json,md}
and wilcoxon.csv artifacts so the §5.2 table can be updated in one step.

Outputs under ``<out_root>/<run_id>/``:
- ``metrics.csv``         : long-form per-seed rows for all three agents
                            (schema is a superset of Phase-2's metrics.csv
                            so the existing aggregate pipeline works).
- ``config.yaml``         : resolved config used for the run.
- ``run_manifest.json``   : commit SHA, seed list, wall-clock, lib versions.
- ``pairwise_wilcoxon.csv`` : per-metric paired Wilcoxon on the three
                              pairwise deltas (Utility−Reflex,
                              Stripped−Reflex, Utility−Stripped).
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

from src.config import load_config
from src.reflex_agent import build_reflex_agent
from src.runner import run_many_episodes
from src.run_artifacts import (
    ensure_run_dir,
    generate_run_id,
    write_manifest,
    write_resolved_config,
)
from src.utility_agent import build_stripped_utility_agent, build_utility_agent


DEFAULT_SEEDS = list(range(50))

PAIRWISE_METRICS = (
    "total_utility",
    "completion_rate",
    "value_weighted_completion_rate",
    "uncapped_completion_rate",
    "failure_rate",
    "total_completed_value",
    "total_compute_cost",
    "avg_compute_cost_per_step",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds (default 0..49, matching Phase-2).",
    )
    return parser.parse_args()


def _run_pool(cfg, factory, seeds):
    metrics = run_many_episodes(cfg=cfg, agent_factory=factory, seeds=seeds)
    return [m.as_dict() for m in metrics]


def _write_metrics_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_wilcoxon(
    rows_by_agent: Dict[str, List[Dict[str, Any]]],
    metrics: Sequence[str],
) -> List[Dict[str, Any]]:
    pairs = [
        ("Utility-Based Agent (Non-Learning Baseline)", "Reflex Agent"),
        ("Stripped Utility-Based Agent", "Reflex Agent"),
        ("Utility-Based Agent (Non-Learning Baseline)", "Stripped Utility-Based Agent"),
    ]
    results: List[Dict[str, Any]] = []
    for lhs, rhs in pairs:
        a = rows_by_agent[lhs]
        b = rows_by_agent[rhs]
        assert len(a) == len(b)
        for metric in metrics:
            lhs_arr = np.array([float(row[metric]) for row in a], dtype=float)
            rhs_arr = np.array([float(row[metric]) for row in b], dtype=float)
            diff = lhs_arr - rhs_arr
            nonzero = int((diff != 0).sum())
            if nonzero == 0:
                stat, pval = float("nan"), float("nan")
            else:
                test = wilcoxon(
                    diff[diff != 0],
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                stat, pval = float(test.statistic), float(test.pvalue)
            results.append(
                {
                    "lhs": lhs,
                    "rhs": rhs,
                    "metric": metric,
                    "n_pairs": len(a),
                    "n_nonzero": nonzero,
                    "mean_delta": round(float(diff.mean()), 6),
                    "median_delta": round(float(np.median(diff)), 6),
                    "statistic": round(stat, 6) if stat == stat else float("nan"),
                    "p_value": pval,
                }
            )
    return results


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_root = Path(args.out_root)
    run_id = args.run_id or generate_run_id("phase4_baseline")
    run_dir = ensure_run_dir(out_root, run_id)
    started = time.time()
    seeds = list(args.seeds)

    print(f"Phase-4 baseline run: {run_dir}")
    print(f"  seeds ({len(seeds)}): {seeds[:5]}...{seeds[-5:]}")

    reflex_rows = _run_pool(cfg, build_reflex_agent, seeds)
    utility_rows = _run_pool(cfg, build_utility_agent, seeds)
    stripped_rows = _run_pool(cfg, build_stripped_utility_agent, seeds)

    all_rows = [*reflex_rows, *utility_rows, *stripped_rows]
    _write_metrics_csv(run_dir / "metrics.csv", all_rows)

    rows_by_agent = {
        "Reflex Agent": reflex_rows,
        "Utility-Based Agent (Non-Learning Baseline)": utility_rows,
        "Stripped Utility-Based Agent": stripped_rows,
    }
    pairwise_rows = _pairwise_wilcoxon(rows_by_agent, PAIRWISE_METRICS)
    _write_metrics_csv(run_dir / "pairwise_wilcoxon.csv", pairwise_rows)

    summary = {agent: {
        "mean_total_utility": float(np.mean([float(r["total_utility"]) for r in rows])),
        "mean_completion_rate": float(np.mean([float(r["completion_rate"]) for r in rows])),
        "mean_total_compute_cost": float(np.mean([float(r["total_compute_cost"]) for r in rows])),
        "mean_failure_rate": float(np.mean([float(r["failure_rate"]) for r in rows])),
    } for agent, rows in rows_by_agent.items()}
    (run_dir / "summary.json").write_text(
        json.dumps({"seeds": seeds, "agents": summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_resolved_config(run_dir, cfg)
    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        seeds=seeds,
        extra={"entrypoint": "phase4_baseline", "num_agents": 3},
        wall_clock_start=started,
    )

    print("Agent means (quick look):")
    for agent, stats in summary.items():
        print(
            f"  {agent:<48s} U={stats['mean_total_utility']:+8.3f}  "
            f"comp={stats['mean_completion_rate']:.3f}  "
            f"cost={stats['mean_total_compute_cost']:6.1f}  "
            f"fail={stats['mean_failure_rate']:.3f}"
        )
    print(f"Wrote Phase-4 baseline to {run_dir}")
    print(f"Run `python -m scripts.aggregate --run-dir {run_dir}` for full table")


if __name__ == "__main__":
    main()
