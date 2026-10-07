from __future__ import annotations

"""Held-out comparison figure.

Two panels:

- Left:  bar chart of per-metric mean ± 95% CI for each agent on the
         held-out pool. Metrics shown: total_utility, completion_rate,
         cost, failure_rate.
- Right: per-seed paired deltas (RL − Reflex, RL − Tuned) for
         total_utility, as a strip plot with the mean Δ dashed line and
         the paired Wilcoxon p-value.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FOCUS_METRICS = (
    "total_utility",
    "completion_rate",
    "total_compute_cost",
    "failure_rate",
)
METRIC_UNITS = {
    "total_utility": "utility units",
    "completion_rate": "fraction",
    "total_compute_cost": "cost units",
    "failure_rate": "fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rl-label-prefix", type=str, default="rl_")
    parser.add_argument("--tuned-label-prefix", type=str, default="Tuned")
    return parser.parse_args()


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir

    aggregate = _read(run_dir / "phase5_aggregate.csv")
    pairwise = _read(run_dir / "phase5_pairwise.csv")

    # Restore the agent ordering: Reflex, Tuned, RL*.
    agents_seen = []
    for row in aggregate:
        if row["agent"] not in agents_seen:
            agents_seen.append(row["agent"])

    def _agent_key(name: str):
        if name == "Reflex Agent":
            return (0, name)
        if name.startswith(args.tuned_label_prefix):
            return (1, name)
        if name.startswith(args.rl_label_prefix):
            return (2, name)
        return (3, name)

    agents_seen.sort(key=_agent_key)

    # Pretty-print long agent names.
    def _short(name: str) -> str:
        if name == "Reflex Agent":
            return "Reflex"
        if name.startswith(args.tuned_label_prefix):
            return "Tuned\nUtility-Based"
        if name.startswith(args.rl_label_prefix):
            return name.replace(args.rl_label_prefix, "RL\n")
        return name

    palette = {
        "Reflex Agent": "#2b2b2b",
        "Tuned": "#1c7ed6",
        "RL": "#c92a2a",
    }

    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_util = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_cost = fig.add_subplot(gs[1, 0])
    ax_fail = fig.add_subplot(gs[1, 1])
    panels = {
        "total_utility": ax_util,
        "completion_rate": ax_comp,
        "total_compute_cost": ax_cost,
        "failure_rate": ax_fail,
    }

    stats_by_agent_metric = {}
    for row in aggregate:
        stats_by_agent_metric[(row["agent"], row["metric"])] = row

    for metric, ax in panels.items():
        xs = np.arange(len(agents_seen))
        means = []
        lo_err = []
        hi_err = []
        colours = []
        for agent in agents_seen:
            row = stats_by_agent_metric[(agent, metric)]
            mean = float(row["mean"])
            lo = float(row["ci95_low"])
            hi = float(row["ci95_high"])
            means.append(mean)
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
            if agent == "Reflex Agent":
                colours.append(palette["Reflex Agent"])
            elif agent.startswith(args.tuned_label_prefix):
                colours.append(palette["Tuned"])
            else:
                colours.append(palette["RL"])
        ax.bar(
            xs,
            means,
            yerr=[lo_err, hi_err],
            color=colours,
            edgecolor="black",
            linewidth=0.4,
            capsize=4,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([_short(a) for a in agents_seen], fontsize=9)
        ax.set_ylabel(f"{metric} ({METRIC_UNITS[metric]})")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Phase-5 held-out comparison ({run_dir.name}, mean ± 95% bootstrap CI)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(args.out, dpi=170)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
