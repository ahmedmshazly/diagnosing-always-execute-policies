from __future__ import annotations

"""Diagnostic — does the TRAINING objective (discounted return under the RL
reward) actually prefer scaling on env_tight, or does discounting hide the
benefit?

This distinguishes two explanations for "REINFORCE stays at always-execute on
env_tight even though scale_when_blocked has higher undiscounted utility":

  (a) OPTIMISATION FAILURE — scale_when_blocked has higher *discounted* return
      too, so REINFORCE *should* prefer it but does not find it.
  (b) DISCOUNT/REWARD ARTIFACT — always-execute has higher *discounted* return
      (scaling's benefit is delayed and δ=0.99-discounted away), so REINFORCE
      *correctly* prefers always-execute for the objective it optimises; the
      discrepancy is the undiscounted metric vs the discounted training return.

For each policy we run the OrchestrationEnv and accumulate, per episode:
  - undiscounted Σ r_t   (== episode utility under failed_jobs_delta)
  - discounted   Σ δ^t r_t   (the REINFORCE objective, G_0)
on the same seeds, and compare the two policies on both.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.config import load_config
from src.rl.env import OrchestrationEnv
from src.sim_environment import EpisodeState


class AlwaysExecute:
    def choose_action(self, s: EpisodeState) -> int:
        return 0  # Execute_Ready_Job


class ScaleWhenBlocked:
    def __init__(self, cfg):
        self.cfg = cfg

    def choose_action(self, s: EpisodeState) -> int:
        ready = s.ready_tasks()
        if not ready:
            return 0
        acpu, aram = s.available_cpu(), s.available_ram()
        if any(t.cpu_demand <= acpu and t.ram_demand <= aram for t in ready):
            return 0
        if s.cluster.scale_boost_remaining == 0:
            return 2  # Scale_Up
        return 0


def _returns(cfg, policy, seeds, delta) -> tuple:
    und, dis = [], []
    for seed in seeds:
        env = OrchestrationEnv(cfg=cfg, num_jobs=cfg.experiment.num_jobs,
                               max_steps=cfg.experiment.max_steps, memoryless=False)
        env.reset(seed=seed)
        g_und, g_dis, t = 0.0, 0.0, 0
        while True:
            a = policy.choose_action(env._state)
            _o, r, term, trunc, _i = env.step(a)
            g_und += r
            g_dis += (delta ** t) * r
            t += 1
            if term or trunc:
                break
        und.append(g_und); dis.append(g_dis)
    return np.array(und), np.array(dis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/env_tight_fixedrisk.yaml"))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(200, 250)))
    args = parser.parse_args()

    cfg = load_config(args.config)
    delta = cfg.rl.delta_discount
    print(f"config={cfg.meta.get('config_name')} risk_mode={cfg.rl.reward_risk_mode} "
          f"delta={delta} | seeds {args.seeds[0]}..{args.seeds[-1]} (n={len(args.seeds)})\n")

    ae_u, ae_d = _returns(cfg, AlwaysExecute(), args.seeds, delta)
    sb_u, sb_d = _returns(cfg, ScaleWhenBlocked(cfg), args.seeds, delta)

    print(f"{'policy':<20} {'undisc Σr':>12} {'disc Σδ^t r':>13}")
    print(f"{'always_execute':<20} {ae_u.mean():>12.3f} {ae_d.mean():>13.3f}")
    print(f"{'scale_when_blocked':<20} {sb_u.mean():>12.3f} {sb_d.mean():>13.3f}")
    print(f"{'Δ (scale − exec)':<20} {sb_u.mean()-ae_u.mean():>+12.3f} "
          f"{sb_d.mean()-ae_d.mean():>+13.3f}")

    d_und = sb_u - ae_u
    d_dis = sb_d - ae_d
    from scipy.stats import wilcoxon
    pu = wilcoxon(sb_u, ae_u).pvalue if np.any(d_und != 0) else float("nan")
    pd = wilcoxon(sb_d, ae_d).pvalue if np.any(d_dis != 0) else float("nan")
    print(f"\npaired Wilcoxon p: undiscounted {pu:.2e} | discounted {pd:.2e}")

    print("\n=== VERDICT ===")
    if sb_d.mean() > ae_d.mean():
        print("scale_when_blocked has HIGHER discounted return too -> REINFORCE "
              "SHOULD prefer it. Staying at always-execute is an OPTIMISATION "
              "FAILURE (weak/noisy gradient toward the better policy), not a "
              "discounting artifact.")
    else:
        print("always-execute has higher DISCOUNTED return even though scaling "
              "wins on undiscounted utility -> REINFORCE correctly optimises its "
              "(discounted) objective; the gap is a DISCOUNT/EVAL MISMATCH, not "
              "an optimisation failure. The fix is the objective (or delta=1), "
              "not the optimiser.")


if __name__ == "__main__":
    main()
