from __future__ import annotations

"""Plot the three training runs on one figure.

Reads ``learning_curve.csv`` from each of the 3 run directories (init seeds
{7, 11, 13}) and overlays their per-update mean returns plus an optional
validation-best marker.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Path to a phase5/rl_seed_* training dir. Pass 3x.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoothing", type=int, default=10)
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override the top-panel title (default: Phase-5 style).",
    )
    return parser.parse_args()


def _load(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return np.array([]), np.array([]), np.array([])
    updates = np.array([int(r["update"]) for r in rows])
    returns = np.array([float(r["mean_undiscounted_return"]) for r in rows])
    env_steps = np.array([int(r["env_steps_cumulative"]) for r in rows])
    return updates, returns, env_steps


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(float)
    kernel = np.ones(window) / float(window)
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p) for p in args.run_dir]
    curves = [(run.name, _load(run / "learning_curve.csv")) for run in run_dirs]

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(10, 9), constrained_layout=True, sharex=False
    )

    palette = ["#2b6cb0", "#2f9e44", "#c92a2a"]
    for (label, (updates, returns, env_steps)), colour in zip(curves, palette):
        if updates.size == 0:
            continue
        ax_top.plot(updates, returns, color=colour, alpha=0.25, linewidth=1.0)
        smoothed = _smooth(returns, args.smoothing)
        ax_top.plot(
            updates[: len(smoothed)] + (args.smoothing // 2),
            smoothed,
            color=colour,
            linewidth=2.2,
            label=label,
        )

    ax_top.set_ylabel("mean episode return")
    ax_top.set_xlabel("policy-gradient update")
    default_title = (
        f"Phase-5 training curves (3 init seeds, {args.smoothing}-update moving avg.)"
    )
    ax_top.set_title(
        args.title or default_title,
        fontsize=12,
        fontweight="bold",
    )
    ax_top.axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
    ax_top.grid(True, alpha=0.25, linestyle=":")
    ax_top.legend(loc="best", fontsize=9)

    # Middle: same data against cumulative env steps for cross-run comparison.
    for (label, (_, returns, env_steps)), colour in zip(curves, palette):
        if env_steps.size == 0:
            continue
        smoothed = _smooth(returns, args.smoothing)
        ax_mid.plot(
            env_steps[: len(smoothed)],
            smoothed,
            color=colour,
            linewidth=2.2,
            label=label,
        )
    ax_mid.set_xlabel("env steps cumulative")
    ax_mid.set_ylabel("mean episode return (smoothed)")
    ax_mid.grid(True, alpha=0.25, linestyle=":")

    # Bottom: validation mean across runs if available.
    for (label, _), run, colour in zip(curves, run_dirs, palette):
        val_path = run / "validation_log.csv"
        if not val_path.exists():
            continue
        with val_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        updates = [int(r["update"]) for r in rows]
        means = [float(r["mean_return"]) for r in rows]
        best_so_far = [float(r["best_mean_return_so_far"]) for r in rows]
        ax_bot.plot(updates, means, color=colour, linestyle=":", alpha=0.8)
        ax_bot.plot(updates, best_so_far, color=colour, linewidth=2.2, label=f"{label} best-so-far")
    ax_bot.set_xlabel("policy-gradient update (validation)")
    ax_bot.set_ylabel("validation-pool mean return")
    ax_bot.set_title("Validation-pool eval (solid = best-so-far)", fontsize=10)
    ax_bot.grid(True, alpha=0.25, linestyle=":")
    ax_bot.legend(loc="best", fontsize=9)

    fig.savefig(args.out, dpi=170)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
