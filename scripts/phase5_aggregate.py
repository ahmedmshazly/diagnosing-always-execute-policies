from __future__ import annotations

"""Phase-5 held-out aggregate.

Reads the Phase-5 combined ``metrics.csv`` from ``scripts.phase5_heldout``
and writes:

- ``phase5_aggregate.csv``     per-(agent, metric) mean, std, 95% bootstrap CI.
- ``phase5_pairwise.csv``      paired Wilcoxon on (RL vs Reflex),
                                (RL vs Tuned), (Tuned vs Reflex), and any
                                cross-RL pair. Uses the ``--lhs`` agent
                                list from the CLI (defaults: everything
                                beginning with ``rl_``) vs the other two.
- ``phase5_aggregate.md``      §5-style table for the paper.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from scipy.stats import wilcoxon

from scripts.aggregate import bootstrap_ci, summarise


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

REFLEX_NAME = "Reflex Agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260418)
    parser.add_argument(
        "--rl-label-prefix",
        type=str,
        default="rl_",
        help="Any agent_name starting with this prefix is treated as an RL run.",
    )
    parser.add_argument(
        "--tuned-label-prefix",
        type=str,
        default="Tuned Utility-Based Agent",
        help="Identifies the tuned Utility-Based agent in the CSV.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_by_agent(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    by_agent: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_agent.setdefault(row["agent_name"], []).append(row)
    return by_agent


def _per_seed_values(
    rows: List[Dict[str, str]], metric: str
) -> np.ndarray:
    return np.array(
        [float(row[metric]) for row in sorted(rows, key=lambda r: int(r["seed"]))],
        dtype=float,
    )


def _paired_wilcoxon(lhs: np.ndarray, rhs: np.ndarray) -> Dict[str, float]:
    diff = lhs - rhs
    nonzero = int((diff != 0).sum())
    if nonzero == 0:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_nonzero": 0,
            "mean_delta": float(diff.mean()) if diff.size else float("nan"),
            "median_delta": float(np.median(diff)) if diff.size else float("nan"),
        }
    result = wilcoxon(
        diff[diff != 0],
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_nonzero": nonzero,
        "mean_delta": float(diff.mean()),
        "median_delta": float(np.median(diff)),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    rows = _read_rows(run_dir / "metrics.csv")
    by_agent = _group_by_agent(rows)

    rng = np.random.default_rng(args.bootstrap_seed)
    per_agent: Dict[str, Dict[str, Dict[str, float]]] = {}
    for agent, agent_rows in by_agent.items():
        per_agent[agent] = {}
        for metric in METRICS:
            values = _per_seed_values(agent_rows, metric)
            per_agent[agent][metric] = summarise(
                values, rng=rng, num_resamples=args.bootstrap_resamples
            )

    # Write aggregate.csv
    agg_rows: List[Dict[str, Any]] = []
    for agent in by_agent.keys():
        for metric in METRICS:
            stats = per_agent[agent][metric]
            agg_rows.append({"agent": agent, "metric": metric, **stats})
    with (run_dir / "phase5_aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["agent", "metric", "n", "mean", "std", "ci95_low", "ci95_high"],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    # Paired Wilcoxon. Build pairs:
    # - (RL_i, Reflex) for every RL run.
    # - (RL_i, Tuned) for every RL run.
    # - (Tuned, Reflex).
    rl_agents = [name for name in by_agent if name.startswith(args.rl_label_prefix)]
    tuned_agents = [
        name for name in by_agent if name.startswith(args.tuned_label_prefix)
    ]
    pairs: List[tuple] = []
    for rl in rl_agents:
        pairs.append((rl, REFLEX_NAME))
        for tuned in tuned_agents:
            pairs.append((rl, tuned))
    for tuned in tuned_agents:
        pairs.append((tuned, REFLEX_NAME))

    pair_rows: List[Dict[str, Any]] = []
    for lhs, rhs in pairs:
        if lhs not in by_agent or rhs not in by_agent:
            continue
        lhs_rows = by_agent[lhs]
        rhs_rows = by_agent[rhs]
        for metric in METRICS:
            lhs_values = _per_seed_values(lhs_rows, metric)
            rhs_values = _per_seed_values(rhs_rows, metric)
            result = _paired_wilcoxon(lhs_values, rhs_values)
            pair_rows.append(
                {
                    "lhs": lhs,
                    "rhs": rhs,
                    "metric": metric,
                    "n_pairs": len(lhs_values),
                    **result,
                }
            )
    with (run_dir / "phase5_pairwise.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lhs",
                "rhs",
                "metric",
                "n_pairs",
                "n_nonzero",
                "mean_delta",
                "median_delta",
                "statistic",
                "p_value",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    # Markdown table.
    md_lines: List[str] = []
    md_lines.append(f"# Phase-5 held-out aggregate — {run_dir}")
    md_lines.append("")
    num_seeds = len(next(iter(by_agent.values())))
    md_lines.append(
        f"- Seeds (held-out): **{num_seeds}** per agent."
    )
    md_lines.append(
        f"- Bootstrap: {args.bootstrap_resamples:,} resamples, seed={args.bootstrap_seed}, 95% CI."
    )
    md_lines.append("")
    md_lines.append("## Per-agent means with 95% bootstrap CI")
    md_lines.append("")
    agent_order = [REFLEX_NAME] + tuned_agents + rl_agents
    agent_order = [a for a in agent_order if a in by_agent]
    headers = ["Metric"] + agent_order
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for metric in METRICS:
        row = [f"`{metric}`"]
        for agent in agent_order:
            stats = per_agent[agent][metric]
            row.append(
                f"{stats['mean']:.4f} ({stats['ci95_low']:.4f}, {stats['ci95_high']:.4f})"
            )
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    md_lines.append("## Paired Wilcoxon (lhs vs rhs, per metric)")
    md_lines.append(
        "Negative Δ means lhs underperforms rhs on that metric. "
        "For compute_cost and failure_rate a negative Δ is an improvement."
    )
    md_lines.append("")
    md_lines.append(
        "| lhs | rhs | metric | mean Δ | median Δ | W | p | n_nonzero |"
    )
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in pair_rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["lhs"],
                    row["rhs"],
                    f"`{row['metric']}`",
                    f"{row['mean_delta']:+.4f}",
                    f"{row['median_delta']:+.4f}",
                    (
                        f"{row['statistic']:.2f}"
                        if row["statistic"] == row["statistic"]
                        else "nan"
                    ),
                    f"{row['p_value']:.4g}" if row["p_value"] == row["p_value"] else "nan",
                    f"{row['n_nonzero']}/{row['n_pairs']}",
                ]
            )
            + " |"
        )
    md_lines.append("")

    (run_dir / "phase5_aggregate.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {run_dir / 'phase5_aggregate.csv'}")
    print(f"Wrote {run_dir / 'phase5_pairwise.csv'}")
    print(f"Wrote {run_dir / 'phase5_aggregate.md'}")


if __name__ == "__main__":
    main()
