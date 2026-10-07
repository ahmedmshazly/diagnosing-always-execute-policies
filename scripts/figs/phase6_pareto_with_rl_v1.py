from __future__ import annotations

"""Phase-6 V1 Pareto overlay.

Extends :mod:`scripts.figs.phase5_pareto_with_rl` to place two sets of
RL points on the same Pareto projections:

- the three Phase-5 8-dim RL runs (plotted in orange); and
- the three Phase-6 V1 14-dim RL runs (plotted in purple).

Everything else (Phase-3 grid, Pareto frontier, Reflex, Tuned UB) is
identical to the Phase-5 figure. The point of the figure is to visualise
whether the V1 points move along the frontier (or above it) relative to
Phase 5.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pareto import pareto_front


PROJECTIONS = (
    ("completion_vs_cost",
     ("completion_rate_mean", "higher", "cost_mean", "lower"),
     ("completion_rate", "total_compute_cost"),
     "Completion vs Cost"),
    ("completion_vs_failure",
     ("completion_rate_mean", "higher", "failure_rate_mean", "lower"),
     ("completion_rate", "failure_rate"),
     "Completion vs Failure rate"),
    ("cost_vs_failure",
     ("cost_mean", "lower", "failure_rate_mean", "lower"),
     ("total_compute_cost", "failure_rate"),
     "Cost vs Failure rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--heldout-v1-dir", type=Path, required=True)
    parser.add_argument("--heldout-phase5-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rl-label-prefix-v1", type=str, default="rl_v1_")
    parser.add_argument("--rl-label-prefix-phase5", type=str, default="rl_seed")
    parser.add_argument("--tuned-label-prefix", type=str, default="Tuned")
    return parser.parse_args()


def _read_dicts(path: Path) -> List[Dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_cell(row: Dict) -> Dict:
    parsed = dict(row)
    for key in (
        "alpha",
        "beta",
        "gamma",
        "mean_utility_mean",
        "cell_utility_mean",
        "completion_rate_mean",
        "value_weighted_completion_rate_mean",
        "uncapped_completion_rate_mean",
        "cost_mean",
        "failure_rate_mean",
    ):
        if key in parsed:
            parsed[key] = float(parsed[key])
    parsed["num_seeds"] = int(float(parsed["num_seeds"]))
    parsed["is_reference"] = parsed.get("is_reference", "False") in ("True", "true", True)
    return parsed


def _held_out_means(run_dir: Path) -> Dict[str, Dict[str, float]]:
    rows = _read_dicts(run_dir / "metrics.csv")
    by_agent: Dict[str, List[Dict]] = {}
    for row in rows:
        by_agent.setdefault(row["agent_name"], []).append(row)
    out: Dict[str, Dict[str, float]] = {}
    for agent, agent_rows in by_agent.items():
        out[agent] = {
            "completion_rate": float(np.mean([float(r["completion_rate"]) for r in agent_rows])),
            "total_compute_cost": float(np.mean([float(r["total_compute_cost"]) for r in agent_rows])),
            "failure_rate": float(np.mean([float(r["failure_rate"]) for r in agent_rows])),
            "total_utility": float(np.mean([float(r["total_utility"]) for r in agent_rows])),
        }
    return out


def main() -> None:
    args = parse_args()
    cells = [_parse_cell(r) for r in _read_dicts(args.sweep_dir / "cells.csv")]
    reflex_row = _parse_cell(_read_dicts(args.sweep_dir / "reflex_aggregate.csv")[0])
    v1 = _held_out_means(args.heldout_v1_dir)
    phase5 = _held_out_means(args.heldout_phase5_dir) if args.heldout_phase5_dir else {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)

    v1_rl_agents = sorted([a for a in v1 if a.startswith(args.rl_label_prefix_v1)])
    # Phase-5 RL agents in the phase5 heldout dir use names like rl_seed7.
    phase5_rl_agents = sorted(
        [
            a for a in phase5
            if a.startswith(args.rl_label_prefix_phase5)
            and not a.startswith(args.rl_label_prefix_v1)
        ]
    )
    tuned_agents = sorted([a for a in v1 if a.startswith(args.tuned_label_prefix)])
    reflex_held = next((a for a in v1 if a == "Reflex Agent"), None)

    v1_palette = ["#845ef7", "#7048e8", "#5f3dc4"]
    phase5_palette = ["#fd7e14", "#f76707", "#d9480f"]

    for ax, (projection_name, pareto_axes, metric_pair, title) in zip(axes, PROJECTIONS):
        x_key_sweep, _, y_key_sweep, _ = pareto_axes
        x_key_held, y_key_held = metric_pair
        xs = np.array([c[x_key_sweep] for c in cells])
        ys = np.array([c[y_key_sweep] for c in cells])
        ax.scatter(xs, ys, c="lightgrey", s=26, edgecolor="black",
                   linewidths=0.3, label="Phase-3 grid cells (n=81)")
        front = pareto_front(
            cells,
            axes=(
                (x_key_sweep, pareto_axes[1]),
                (y_key_sweep, pareto_axes[3]),
            ),
        )
        if front:
            front_sorted = sorted(front, key=lambda c: c[x_key_sweep])
            ax.plot(
                [c[x_key_sweep] for c in front_sorted],
                [c[y_key_sweep] for c in front_sorted],
                color="#2b6cb0",
                linewidth=1.3,
                alpha=0.85,
                label=f"Pareto front (n={len(front)})",
            )
            ax.scatter(
                [c[x_key_sweep] for c in front_sorted],
                [c[y_key_sweep] for c in front_sorted],
                facecolors="#2b6cb0",
                s=32,
                edgecolor="black",
                linewidths=0.3,
            )

        ax.scatter(
            reflex_row[x_key_sweep],
            reflex_row[y_key_sweep],
            marker="s",
            color="black",
            s=110,
            label="Reflex (sweep pool)",
            zorder=6,
        )
        if reflex_held is not None:
            ax.scatter(
                v1[reflex_held][x_key_held],
                v1[reflex_held][y_key_held],
                marker="D",
                color="#495057",
                s=100,
                edgecolor="black",
                linewidths=0.5,
                label="Reflex (held-out)",
                zorder=6,
            )
        for agent in tuned_agents:
            ax.scatter(
                v1[agent][x_key_held],
                v1[agent][y_key_held],
                marker="X",
                color="#1c7ed6",
                s=150,
                edgecolor="black",
                linewidths=0.5,
                label="Tuned UB (held-out)",
                zorder=7,
            )

        for agent, colour in zip(phase5_rl_agents, phase5_palette):
            ax.scatter(
                phase5[agent][x_key_held],
                phase5[agent][y_key_held],
                marker="o",
                color=colour,
                s=130,
                edgecolor="black",
                linewidths=0.5,
                label=f"Phase-5 RL: {agent}",
                zorder=7,
            )

        for agent, colour in zip(v1_rl_agents, v1_palette):
            ax.scatter(
                v1[agent][x_key_held],
                v1[agent][y_key_held],
                marker="*",
                color=colour,
                s=260,
                edgecolor="black",
                linewidths=0.5,
                label=f"Phase-6 V1 RL: {agent}",
                zorder=8,
            )

        ax.set_xlabel(x_key_sweep.replace("_mean", ""))
        ax.set_ylabel(y_key_sweep.replace("_mean", ""))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "Phase-6 V1 vs Phase-5 RL points on the Phase-3 Pareto projections",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(args.out, dpi=170)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
