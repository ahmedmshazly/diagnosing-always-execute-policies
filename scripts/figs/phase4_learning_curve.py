from __future__ import annotations

"""Phase-4 smoke-test learning-curve figure.

Reads ``<run_dir>/learning_curve.csv`` and writes a 2-panel figure:
- top panel:  mean_undiscounted_return per update, with a smoothed trace.
- bottom:     total_loss per update, with a smoothed trace.

Both panels share the x-axis (update index). The title reports first-window
vs last-window improvements from ``summary.json``.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoothing-window", type=int, default=5)
    return parser.parse_args()


def _read_curve(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    updates = np.array([int(r["update"]) for r in rows])
    returns = np.array([float(r["mean_undiscounted_return"]) for r in rows])
    losses = np.array([float(r["total_loss"]) for r in rows])
    entropies = np.array([float(r["entropy"]) for r in rows])
    env_steps = np.array([int(r["env_steps_cumulative"]) for r in rows])
    return updates, returns, losses, entropies, env_steps


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(float)
    kernel = np.ones(window) / float(window)
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    updates, returns, losses, entropies, env_steps = _read_curve(run_dir / "learning_curve.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(9, 9), constrained_layout=True, sharex=True
    )

    axes[0].plot(updates, returns, color="#a5a5a5", alpha=0.6, label="per-update mean return")
    smoothed = _smooth(returns, args.smoothing_window)
    axes[0].plot(
        updates[: len(smoothed)] + (args.smoothing_window // 2),
        smoothed,
        color="#2b6cb0",
        linewidth=2.0,
        label=f"{args.smoothing_window}-update moving avg.",
    )
    axes[0].axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
    axes[0].set_ylabel("mean episode return")
    axes[0].set_title(
        f"RL smoke-test learning curve — {run_dir.name}\n"
        f"first-window return {summary.get('first_window_mean_return', float('nan')):+.3f} → "
        f"last-window {summary.get('last_window_mean_return', float('nan')):+.3f} "
        f"(Δ={summary.get('return_improvement', float('nan')):+.3f})",
        fontsize=11,
    )
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.25, linestyle=":")

    axes[1].plot(updates, losses, color="#d4a5a5", alpha=0.6)
    loss_smoothed = _smooth(losses, args.smoothing_window)
    axes[1].plot(
        updates[: len(loss_smoothed)] + (args.smoothing_window // 2),
        loss_smoothed,
        color="#c92a2a",
        linewidth=2.0,
    )
    axes[1].set_ylabel("total loss (policy − cH·entropy)")
    axes[1].grid(True, alpha=0.25, linestyle=":")
    axes[1].set_title(
        f"first-window loss {summary.get('first_window_mean_loss', float('nan')):+.3f} → "
        f"last-window {summary.get('last_window_mean_loss', float('nan')):+.3f} "
        f"(Δ={summary.get('loss_improvement', float('nan')):+.3f}, positive = loss decreased)",
        fontsize=10,
    )

    axes[2].plot(updates, entropies, color="#2f9e44", linewidth=1.5)
    axes[2].set_ylabel("policy entropy")
    axes[2].set_xlabel(
        f"policy-gradient update (env steps cumulative shown on twin axis below)"
    )
    axes[2].grid(True, alpha=0.25, linestyle=":")

    ax2 = axes[2].twiny()
    ax2.plot(env_steps, entropies, alpha=0)  # transparent, only to set x-limits
    ax2.set_xlim(env_steps.min(), env_steps.max())
    ax2.set_xlabel("env steps cumulative", fontsize=9)

    out_path = args.out or (run_dir / "learning_curve.png")
    fig.savefig(out_path, dpi=170)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
