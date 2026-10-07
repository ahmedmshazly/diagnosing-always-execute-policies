from __future__ import annotations

"""Performance-profile + per-seed figure for the n=250 power analysis
(rliable-style, Agarwal et al. 2021). Reads the committed CSVs so the figure
regenerates from data.
"""

from pathlib import Path

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    return list(csv.DictReader(open(path)))


def main() -> None:
    prof = _read("results/hardening/power_rliable.csv")
    per = _read("results/hardening/power_rliable_perseed.csv")
    tau = np.array([float(r["tau"]) for r in prof])
    rl_ge = np.array([float(r["rl_frac_ge"]) for r in prof])
    rf_ge = np.array([float(r["reflex_frac_ge"]) for r in prof])
    diffs = np.array([float(r["diff"]) for r in per])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(tau, rf_ge, label="Reflex", color="#1f77b4", lw=2)
    ax.plot(tau, rl_ge, label="RL (always-execute)", color="#d62728", lw=2)
    ax.set_xlabel("total utility threshold τ")
    ax.set_ylabel("fraction of seeds with score ≥ τ")
    ax.set_title("Performance profiles (n=250 workloads)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    order = np.argsort(diffs)
    colors = ["#2ca02c" if d > 0 else ("#7f7f7f" if d == 0 else "#d62728") for d in diffs[order]]
    ax.bar(range(len(diffs)), diffs[order], color=colors, width=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(diffs.mean(), color="purple", ls="--", lw=1.5,
               label=f"mean {diffs.mean():+.2f} (Wilcoxon p=2.3e-3)")
    ax.set_xlabel("seed (sorted by RL−Reflex)")
    ax.set_ylabel("RL − Reflex total utility")
    ax.set_title("Per-seed deterministic gap (n=250)\ngreen=RL wins, grey=tie, red=Reflex wins")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path("results/hardening/fig_power_rliable.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
