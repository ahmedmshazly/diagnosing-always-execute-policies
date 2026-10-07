from __future__ import annotations

"""Phase-3 Pareto-frontier plots + best fixed-weight selection.

Reads ``<run-dir>/cells.csv`` and ``<run-dir>/reflex_aggregate.csv``, then:

1. Plots three Pareto-frontier panels:
   - completion_rate vs cost
   - completion_rate vs failure_rate
   - cost vs failure_rate
   Markers:
   - grey dot = grid cell
   - blue star = midterm Utility reference (α=1.0, β=0.4, γ=0.8)
   - green star = Phase-1 default reference (α=1.0, β=0.1, γ=1.0)
   - red ✕ = the "best fixed-weight" cell chosen for the RL comparison
   - black square = Reflex baseline (weight-independent)
   Line = Pareto frontier in the 2D projection.

2. Writes ``best_fixed_weight.json`` with the selected cell + the decision
   rule used; that file is the contract the RL agent's evaluation harness
   will load.

3. Writes ``pareto_fronts.csv`` with every frontier membership for every
   projection, so a reviewer can audit the plot.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pareto import best_by_metric, cells_dominating, pareto_front


PROJECTIONS: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...] = (
    ("completion_vs_cost", (("completion_rate_mean", "higher"), ("cost_mean", "lower"))),
    ("completion_vs_failure", (("completion_rate_mean", "higher"), ("failure_rate_mean", "lower"))),
    ("cost_vs_failure", (("cost_mean", "lower"), ("failure_rate_mean", "lower"))),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (defaults to <run-dir>/phase3_pareto.png).",
    )
    parser.add_argument(
        "--selection-rule",
        type=str,
        default="mean_utility_then_completion",
        choices=("mean_utility", "mean_utility_then_completion", "completion", "dominator_of_midterm"),
        help="Rule used to pick the best fixed-weight cell for RL comparison.",
    )
    return parser.parse_args()


def _read_cells(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in (
                "alpha",
                "beta",
                "gamma",
                "mean_utility_mean",
                "mean_utility_std",
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
            parsed["is_reference"] = parsed["is_reference"] in ("True", "true", True)
            rows.append(parsed)
    return rows


def _reference_cells(cells: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        cell["reference_name"]: cell
        for cell in cells
        if cell["is_reference"] and cell.get("reference_name")
    }


def _choose_best_fixed_weight(
    cells: Sequence[Dict[str, Any]],
    rule: str,
    midterm_ref: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Select the fixed-weight cell the RL agent will be benchmarked against.

    Decision rules:
    - ``mean_utility``               : pick the cell with the highest
                                       ``mean_utility_mean`` under the shared
                                       evaluation utility.
    - ``mean_utility_then_completion``: same as above, with a tie-break on
                                       ``completion_rate_mean``.
    - ``completion``                 : pick by ``completion_rate_mean``.
    - ``dominator_of_midterm``       : pick a cell that Pareto-dominates the
                                       midterm reference on the (completion,
                                       cost, failure) triple, preferring the
                                       one with highest mean_utility_mean.
    """
    if not cells:
        raise ValueError("no cells to choose from")

    if rule == "mean_utility":
        best = best_by_metric(cells, "mean_utility_mean", "higher")
        return dict(best)

    if rule == "mean_utility_then_completion":
        best_utility = max(c["mean_utility_mean"] for c in cells)
        contenders = [c for c in cells if abs(c["mean_utility_mean"] - best_utility) < 1e-9]
        if len(contenders) == 1:
            return dict(contenders[0])
        return dict(best_by_metric(contenders, "completion_rate_mean", "higher"))

    if rule == "completion":
        return dict(best_by_metric(cells, "completion_rate_mean", "higher"))

    if rule == "dominator_of_midterm":
        if midterm_ref is None:
            raise ValueError("dominator_of_midterm requires a midterm reference in cells.csv")
        axes = (
            ("completion_rate_mean", "higher"),
            ("cost_mean", "lower"),
            ("failure_rate_mean", "lower"),
        )
        dominance = cells_dominating(cells, midterm_ref, axes=axes, reference_name="midterm_utility")
        if not dominance.dominators:
            raise RuntimeError(
                "No cell strictly dominates the midterm reference on (completion, cost, failure);"
                " fall back to a different selection rule."
            )
        return dict(best_by_metric(dominance.dominators, "mean_utility_mean", "higher"))

    raise ValueError(f"unknown rule: {rule}")


def _plot_projection(
    ax,
    cells: Sequence[Dict[str, Any]],
    x_key: str,
    x_dir: str,
    y_key: str,
    y_dir: str,
    reflex: Dict[str, Any] | None,
    midterm_ref: Dict[str, Any] | None,
    phase1_ref: Dict[str, Any] | None,
    best_cell: Dict[str, Any],
    title: str,
) -> None:
    xs = np.array([c[x_key] for c in cells])
    ys = np.array([c[y_key] for c in cells])
    ax.scatter(xs, ys, c="lightgrey", s=26, edgecolor="black", linewidths=0.3, label="grid cells")

    front = pareto_front(cells, axes=((x_key, x_dir), (y_key, y_dir)))
    if front:
        front_sorted = sorted(front, key=lambda c: c[x_key])
        ax.plot(
            [c[x_key] for c in front_sorted],
            [c[y_key] for c in front_sorted],
            color="#2b6cb0",
            linewidth=1.3,
            alpha=0.85,
            label=f"Pareto front (n={len(front)})",
        )
        ax.scatter(
            [c[x_key] for c in front_sorted],
            [c[y_key] for c in front_sorted],
            facecolors="#2b6cb0",
            s=32,
            edgecolor="black",
            linewidths=0.3,
            zorder=4,
        )

    if reflex is not None:
        ax.scatter(
            reflex[x_key],
            reflex[y_key],
            marker="s",
            color="black",
            s=90,
            label=f"Reflex (weight-independent)",
            zorder=6,
        )
    if midterm_ref is not None:
        ax.scatter(
            midterm_ref[x_key],
            midterm_ref[y_key],
            marker="*",
            color="#1c7ed6",
            s=230,
            edgecolor="black",
            linewidths=0.5,
            label=f"Midterm (α={midterm_ref['alpha']:g}, β={midterm_ref['beta']:g}, γ={midterm_ref['gamma']:g})",
            zorder=7,
        )
    if phase1_ref is not None:
        ax.scatter(
            phase1_ref[x_key],
            phase1_ref[y_key],
            marker="*",
            color="#2f9e44",
            s=180,
            edgecolor="black",
            linewidths=0.5,
            label=f"Phase-1 default (α={phase1_ref['alpha']:g}, β={phase1_ref['beta']:g}, γ={phase1_ref['gamma']:g})",
            zorder=7,
        )

    ax.scatter(
        best_cell[x_key],
        best_cell[y_key],
        marker="X",
        color="#c92a2a",
        s=180,
        edgecolor="black",
        linewidths=0.7,
        label=f"Best fixed-weight (α={best_cell['alpha']:g}, β={best_cell['beta']:g}, γ={best_cell['gamma']:g})",
        zorder=8,
    )

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="best", fontsize=7)


def _export_pareto_fronts(
    run_dir: Path,
    cells: Sequence[Dict[str, Any]],
) -> None:
    rows: List[Dict[str, Any]] = []
    for projection_name, axes in PROJECTIONS:
        front = pareto_front(cells, axes=axes)
        for cell in front:
            rows.append(
                {
                    "projection": projection_name,
                    "alpha": cell["alpha"],
                    "beta": cell["beta"],
                    "gamma": cell["gamma"],
                    "cell_id": cell["cell_id"],
                    "is_reference": cell["is_reference"],
                    "reference_name": cell.get("reference_name", ""),
                    "mean_utility_mean": cell["mean_utility_mean"],
                    "completion_rate_mean": cell["completion_rate_mean"],
                    "cost_mean": cell["cost_mean"],
                    "failure_rate_mean": cell["failure_rate_mean"],
                }
            )
    fieldnames = [
        "projection",
        "alpha",
        "beta",
        "gamma",
        "cell_id",
        "is_reference",
        "reference_name",
        "mean_utility_mean",
        "completion_rate_mean",
        "cost_mean",
        "failure_rate_mean",
    ]
    with (run_dir / "pareto_fronts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    cells = _read_cells(run_dir / "cells.csv")
    reflex_agg = _read_cells(run_dir / "reflex_aggregate.csv")
    reflex_row = reflex_agg[0] if reflex_agg else None

    refs = _reference_cells(cells)
    midterm_ref = refs.get("midterm_utility")
    phase1_ref = refs.get("phase1_default")

    best_cell = _choose_best_fixed_weight(
        cells, rule=args.selection_rule, midterm_ref=midterm_ref
    )

    # Compute dominance of the midterm reference to include in best.json.
    dominance_report: Dict[str, Any] = {}
    if midterm_ref is not None:
        for proj_name, axes in PROJECTIONS:
            dominance = cells_dominating(cells, midterm_ref, axes=axes, reference_name="midterm_utility")
            dominance_report[proj_name] = {
                "num_dominators": len(dominance.dominators),
                "dominator_cell_ids": [d["cell_id"] for d in dominance.dominators],
            }

    best_payload = {
        "selection_rule": args.selection_rule,
        "best_fixed_weight": {
            key: best_cell[key]
            for key in (
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
        },
        "midterm_reference": midterm_ref,
        "phase1_reference": phase1_ref,
        "reflex_reference": reflex_row,
        "midterm_dominance": dominance_report,
    }
    (run_dir / "best_fixed_weight.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True, default=float),
        encoding="utf-8",
    )
    _export_pareto_fronts(run_dir, cells)

    # Plot three separate figures (single-panel) so each is readable on its
    # own, plus one combined 3-panel figure.
    out_path = args.out or (run_dir / "phase3_pareto.png")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 5.3), constrained_layout=True)
    panels = [
        (axes[0], "completion_rate_mean", "higher", "cost_mean", "lower",
         "Completion vs Cost"),
        (axes[1], "completion_rate_mean", "higher", "failure_rate_mean", "lower",
         "Completion vs Failure rate"),
        (axes[2], "cost_mean", "lower", "failure_rate_mean", "lower",
         "Cost vs Failure rate"),
    ]
    for ax, x_key, x_dir, y_key, y_dir, title in panels:
        _plot_projection(
            ax=ax,
            cells=cells,
            x_key=x_key,
            x_dir=x_dir,
            y_key=y_key,
            y_dir=y_dir,
            reflex=reflex_row,
            midterm_ref=midterm_ref,
            phase1_ref=phase1_ref,
            best_cell=best_cell,
            title=title,
        )
    fig.suptitle(
        f"Phase-3 Pareto frontiers (80-cell sweep, n={cells[0]['num_seeds']} seeds each; best by {args.selection_rule})",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=170)

    # Also emit a separate 3-panel vertical variant for space-constrained
    # layouts.
    vertical_path = run_dir / "phase3_pareto_vertical.png"
    fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(7, 15), constrained_layout=True)
    for ax, (x_key, x_dir, y_key, y_dir, title) in zip(
        axes2,
        [
            ("completion_rate_mean", "higher", "cost_mean", "lower", "Completion vs Cost"),
            ("completion_rate_mean", "higher", "failure_rate_mean", "lower", "Completion vs Failure rate"),
            ("cost_mean", "lower", "failure_rate_mean", "lower", "Cost vs Failure rate"),
        ],
    ):
        _plot_projection(
            ax=ax,
            cells=cells,
            x_key=x_key,
            x_dir=x_dir,
            y_key=y_key,
            y_dir=y_dir,
            reflex=reflex_row,
            midterm_ref=midterm_ref,
            phase1_ref=phase1_ref,
            best_cell=best_cell,
            title=title,
        )
    fig2.savefig(vertical_path, dpi=170)
    print(f"Wrote {out_path}")
    print(f"Wrote {vertical_path}")
    print(f"Wrote {run_dir / 'best_fixed_weight.json'}")
    print(f"Wrote {run_dir / 'pareto_fronts.csv'}")


if __name__ == "__main__":
    main()
