from __future__ import annotations

"""Aggregate statistics for the Phase-4 3-agent baseline table.

Reads ``metrics.csv`` written by ``scripts.phase4_baseline`` and produces:

- ``phase4_aggregate.csv``: per-(agent, metric) n, mean, std, 95% bootstrap CI.
- ``phase4_aggregate.md``: human-readable §5.2-style table (3 agents).

Re-uses the bootstrap machinery from :mod:`scripts.aggregate`.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from scripts.aggregate import bootstrap_ci, summarise


AGENTS = (
    "Reflex Agent",
    "Utility-Based Agent (Non-Learning Baseline)",
    "Stripped Utility-Based Agent",
)
METRICS = (
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260418)
    return parser.parse_args()


def _read_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    rows = _read_metrics(run_dir / "metrics.csv")

    rng = np.random.default_rng(args.bootstrap_seed)
    per_agent: Dict[str, Dict[str, Dict[str, float]]] = {a: {} for a in AGENTS}
    num_seeds = 0
    for agent in AGENTS:
        agent_rows = [row for row in rows if row["agent_name"] == agent]
        if not agent_rows:
            raise ValueError(f"no rows for agent {agent!r} in {run_dir / 'metrics.csv'}")
        num_seeds = len(agent_rows)
        for metric in METRICS:
            values = np.array([float(row[metric]) for row in agent_rows], dtype=float)
            per_agent[agent][metric] = summarise(
                values, rng=rng, num_resamples=args.bootstrap_resamples
            )

    csv_rows: List[Dict[str, Any]] = []
    for agent in AGENTS:
        for metric in METRICS:
            stats = per_agent[agent][metric]
            csv_rows.append({"agent": agent, "metric": metric, **stats})
    with (run_dir / "phase4_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "agent",
                "metric",
                "n",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    lines: List[str] = []
    lines.append(f"# Phase-4 baseline aggregate — {run_dir}")
    lines.append("")
    lines.append(
        f"- Seeds: **{num_seeds}**; bootstrap: {args.bootstrap_resamples:,} resamples, "
        f"seed={args.bootstrap_seed}, 95% CI."
    )
    lines.append("")
    lines.append("## Per-agent means with 95% bootstrap CI")
    headers = ["Metric", "Reflex", "Utility-Based (full)", "Utility-Based (stripped)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for metric in METRICS:
        row = [f"`{metric}`"]
        for agent in AGENTS:
            stats = per_agent[agent][metric]
            row.append(
                f"{stats['mean']:.4f} ({stats['ci95_low']:.4f}, {stats['ci95_high']:.4f})"
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Append the pairwise Wilcoxon summary if produced by phase4_baseline.
    wilcoxon_path = run_dir / "pairwise_wilcoxon.csv"
    if wilcoxon_path.exists():
        wil_rows = _read_metrics(wilcoxon_path)
        lines.append("## Paired Wilcoxon signed-rank")
        lines.append(
            "Three pairs: (A) Utility − Reflex, (B) Stripped − Reflex, "
            "(C) Utility − Stripped. Negative Δ means the lhs agent "
            "underperforms the rhs on that metric."
        )
        lines.append("")
        lines.append(
            "| lhs | rhs | metric | mean Δ | median Δ | W | p |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for row in wil_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["lhs"].split(" ")[0],
                        row["rhs"].split(" ")[0],
                        f"`{row['metric']}`",
                        f"{float(row['mean_delta']):+.4f}",
                        f"{float(row['median_delta']):+.4f}",
                        (
                            f"{float(row['statistic']):.2f}"
                            if row["statistic"]
                            else "nan"
                        ),
                        (
                            f"{float(row['p_value']):.4g}"
                            if row["p_value"] not in ("nan", "")
                            else "nan"
                        ),
                    ]
                )
                + " |"
            )
        lines.append("")

    (run_dir / "phase4_aggregate.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {run_dir / 'phase4_aggregate.csv'}")
    print(f"Wrote {run_dir / 'phase4_aggregate.md'}")


if __name__ == "__main__":
    main()
