from __future__ import annotations

"""PPO training driver (the stronger-method control). Saves a policy_final.pt /
policy_best_by_val.pt loadable by src.rl.agent.RLPolicyAgent, so the same
evaluation tooling (diag_policy_compare, diag_power_rliable) applies unchanged.
"""

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch

# One thread per process so N parallel runs use ~N cores, not N*4 (the sim loop
# is single-threaded; the MLP is tiny, so intra-op parallelism only thrashes).
torch.set_num_threads(1)

from src.config import load_config
from src.rl.ppo import PPOConfig, PPORecord, train_ppo
from src.run_artifacts import ensure_run_dir, write_manifest, write_resolved_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--run-id", required=True)
    p.add_argument("--init-seed", type=int, default=7)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--num-jobs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--validation-cap", type=int, default=20)
    p.add_argument("--out-root", type=Path, default=Path("results"))
    args = p.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    run_dir = ensure_run_dir(args.out_root, args.run_id)
    started = time.time()

    ppo = PPOConfig(
        total_iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        num_jobs=args.num_jobs,
        max_steps=args.max_steps,
        ppo_epochs=args.ppo_epochs,
        entropy_coef=args.entropy_coef,
        gamma=cfg.rl.delta_discount,
        lr=cfg.rl.learning_rate,
        init_seed=args.init_seed,
        hidden_sizes=tuple(cfg.rl.network.hidden_sizes),
        activation=cfg.rl.network.activation,
    )
    val_seeds = list(cfg.rl.validation_seeds)[: args.validation_cap]
    print(f"PPO: {run_dir} | config={cfg.meta.get('config_name')} "
          f"risk_mode={cfg.rl.reward_risk_mode} | iters={ppo.total_iterations} "
          f"N={ppo.num_jobs} | seed={ppo.init_seed}")

    log: List[PPORecord] = []
    actor, _critic, best_state, best_val = train_ppo(
        cfg, ppo, validation_seeds=val_seeds, log=log, log_every=10)

    torch.save(actor.state_dict(), run_dir / "policy_final.pt")
    torch.save(best_state, run_dir / "policy_best_by_val.pt")
    rows = [asdict(r) for r in log]
    if rows:
        with (run_dir / "learning_curve.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write_resolved_config(run_dir, cfg)
    write_manifest(run_dir=run_dir, cfg=cfg, seeds=val_seeds,
                   extra={"entrypoint": "train_ppo", "best_val": best_val,
                          "iterations": ppo.total_iterations}, wall_clock_start=started)
    print(f"Done. best_val={best_val:+.3f}  ({time.time()-started:.1f}s)")


if __name__ == "__main__":
    main()
