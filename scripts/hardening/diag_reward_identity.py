from __future__ import annotations

"""Diagnostic A1 — does Sum_t r_t equal the reported episode utility?

The paper's methodological centrepiece is a "three-role identification": the
same scalar U is (i) the evaluation metric, (ii) the non-learning agent's
scoring rule, and (iii) the RL reward. Paper Section 4.3.3 states explicitly that
"the cumulative sum Sum_t r_t equals the total episode utility reported
throughout, so the RL agent is trained on exactly the objective it is
scored against."

This script tests that claim directly. For each seed it runs ONE always-
execute trajectory two ways on an identical seed (so the trajectories are
byte-identical):

  PATH A (canonical metric): src.runner.run_episode with an always-execute
          agent -> EpisodeMetrics.total_utility, computed by
          src.metrics.summarize_episode as
              U = alpha*completed_value - beta*compute_cost - gamma*failed_jobs

  PATH B (RL reward): src.rl.env.OrchestrationEnv stepped with action 0
          every step -> sum of per-step rewards r_t, where
              r_t = alpha*dValue_t - beta*step_cost_t - gamma*dRisk_t
          and dRisk_t is the change in the normalised recent_failures counter.

If the identity holds, Sum_t r_t == total_utility for every seed. If the
risk term does not telescope (because recent_failures decays every step and
is normalised, whereas the metric counts failed *jobs*), the two disagree by
exactly gamma * (failed_jobs - final_normalised_recent_failures).

Outputs results/hardening/reward_identity.csv and prints a summary table.
No training; deterministic; runs in seconds.
"""

import csv
from pathlib import Path
from typing import List

import numpy as np

from src.config import load_config
from src.metrics import summarize_episode
from src.rl.env import OrchestrationEnv, evaluation_weights_from_config
from src.runner import run_episode
from src.sim_environment import EpisodeState


class AlwaysExecuteAgent:
    """Emits Execute_Ready_Job every step — the converged RL policy."""

    name = "Always-Execute (diagnostic)"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def choose_action(self, state: EpisodeState) -> str:
        return "Execute_Ready_Job"


def _canonical_utility(cfg, seed: int) -> dict:
    """PATH A: official EpisodeMetrics via the shared runner."""
    metrics = run_episode(
        cfg=cfg,
        agent_factory=lambda c: AlwaysExecuteAgent(c),
        seed=seed,
        include_uncapped=False,
    )
    return {
        "total_utility": metrics.total_utility,
        "completed_value": metrics.total_completed_value,
        "compute_cost": metrics.total_compute_cost,
        "failed_jobs": metrics.failed_jobs,
        "steps": metrics.steps_executed,
    }


def _reward_sum(cfg, seed: int) -> dict:
    """PATH B: sum of the RL per-step reward on the identical trajectory."""
    ew = evaluation_weights_from_config(cfg)
    env = OrchestrationEnv(
        cfg=cfg,
        num_jobs=cfg.experiment.num_jobs,
        max_steps=cfg.experiment.max_steps,
        memoryless=False,
    )
    env.reset(seed=seed)
    reward_sum = 0.0
    sum_dvalue = 0.0
    sum_step_cost = 0.0
    sum_drisk = 0.0
    steps = 0
    while True:
        _obs, reward, terminated, truncated, info = env.step(0)  # 0 == Execute_Ready_Job
        reward_sum += float(reward)
        sum_dvalue += float(info["delta_value"])
        sum_step_cost += float(info["step_cost"])
        sum_drisk += float(info["delta_risk"])
        steps += 1
        if terminated or truncated:
            break
    final_norm_rf = env._normalised_recent_failures()
    return {
        "reward_sum": reward_sum,
        "sum_dvalue": sum_dvalue,
        "sum_step_cost": sum_step_cost,
        "sum_drisk": sum_drisk,
        "final_norm_recent_failures": final_norm_rf,
        "alpha": ew.alpha,
        "beta": ew.beta,
        "gamma": ew.gamma,
        "steps": steps,
    }


def main() -> None:
    cfg = load_config()
    seeds = list(range(200, 210))  # held-out pool; diagnostic only
    rows: List[dict] = []
    print(
        f"{'seed':>5} {'U_metric':>10} {'reward_sum':>11} {'gap':>9} "
        f"{'failed':>7} {'final_rf':>9} {'pred_gap':>9} {'val_ok':>7} {'cost_ok':>8}"
    )
    for seed in seeds:
        a = _canonical_utility(cfg, seed)
        b = _reward_sum(cfg, seed)
        gap = b["reward_sum"] - a["total_utility"]
        # Predicted gap if only the risk term fails to telescope:
        #   reward_sum - U = -gamma*sum_drisk + gamma*failed_jobs
        #                  = gamma*(failed_jobs - final_norm_recent_failures)
        pred_gap = b["gamma"] * (a["failed_jobs"] - b["final_norm_recent_failures"])
        # Value term telescopes iff sum_dvalue == completed_value.
        val_ok = abs(b["sum_dvalue"] - a["completed_value"]) < 1e-6
        # Cost term telescopes iff sum_step_cost == compute_cost.
        cost_ok = abs(b["sum_step_cost"] - a["compute_cost"]) < 1e-6
        # Risk telescopes to the *counter*, not failed_jobs:
        risk_to_counter_ok = abs(b["sum_drisk"] - b["final_norm_recent_failures"]) < 1e-6
        row = {
            "seed": seed,
            "U_metric": round(a["total_utility"], 4),
            "reward_sum": round(b["reward_sum"], 4),
            "gap": round(gap, 4),
            "pred_gap": round(pred_gap, 4),
            "failed_jobs": a["failed_jobs"],
            "final_norm_recent_failures": round(b["final_norm_recent_failures"], 4),
            "completed_value": round(a["completed_value"], 4),
            "compute_cost": round(a["compute_cost"], 4),
            "sum_dvalue": round(b["sum_dvalue"], 4),
            "sum_step_cost": round(b["sum_step_cost"], 4),
            "sum_drisk": round(b["sum_drisk"], 4),
            "value_telescopes": val_ok,
            "cost_telescopes": cost_ok,
            "risk_telescopes_to_counter": risk_to_counter_ok,
            "alpha": b["alpha"],
            "beta": b["beta"],
            "gamma": b["gamma"],
        }
        rows.append(row)
        print(
            f"{seed:>5} {row['U_metric']:>10} {row['reward_sum']:>11} {row['gap']:>9} "
            f"{row['failed_jobs']:>7} {row['final_norm_recent_failures']:>9} "
            f"{row['pred_gap']:>9} {str(val_ok):>7} {str(cost_ok):>8}"
        )

    gaps = np.array([r["gap"] for r in rows], dtype=float)
    pred = np.array([r["pred_gap"] for r in rows], dtype=float)
    print("\n--- summary ---")
    print(f"seeds                        : {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print(f"mean gap (reward_sum - U)    : {gaps.mean():+.4f}")
    print(f"max |gap|                    : {np.abs(gaps).max():.4f}")
    print(f"max |gap - predicted_gap|    : {np.abs(gaps - pred).max():.2e}")
    print(f"value term telescopes (all)  : {all(r['value_telescopes'] for r in rows)}")
    print(f"cost  term telescopes (all)  : {all(r['cost_telescopes'] for r in rows)}")
    print(f"risk telescopes to METRIC    : {all(abs(r['gap']) < 1e-6 for r in rows)}")
    print(f"risk telescopes to COUNTER   : {all(r['risk_telescopes_to_counter'] for r in rows)}")

    out = Path("results/hardening/reward_identity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
