from __future__ import annotations

"""Phase-4 smoke-test trainer entry point.

Scope: 10 000 env steps on the stage-1 curriculum cell ((N, T_max) =
(20, 120)) using initialisation seed 7 (first of cfg.rl.initialisation_seeds
by default, overridable via --init-seed).

Full training is Phase 5. This driver deliberately stops when the
cumulative env-step budget is exhausted so the smoke test is a tight,
reproducible ~minutes-scale run.

Outputs under ``<out_root>/<run_id>/``:
- ``config.yaml``        resolved config used for the run.
- ``run_manifest.json``  commit SHA, seed list, wall-clock, lib versions.
- ``learning_curve.csv`` one row per policy-gradient update.
- ``summary.json``       compact stats summary used by the figure script.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.config import load_config
from src.rl.trainer import StageSpec, TrainConfig, UpdateRecord, train
from src.run_artifacts import (
    ensure_run_dir,
    generate_run_id,
    write_manifest,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=10_000,
        help="Phase-4 smoke-test budget (default 10k env steps).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help=(
            "1 = use only curriculum stage 1 (N=20, T_max=120) per the "
            "Phase-4 brief. 0 = use the full 3-stage curriculum (Phase 5)."
        ),
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Network init seed; default = cfg.rl.initialisation_seeds[0].",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override cfg.rl.batch_size (smaller batches = more updates per 10k steps).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print one line every N updates.",
    )
    parser.add_argument(
        "--fixed-arrival-seed",
        type=int,
        default=None,
        help=(
            "Use this seed as the arrival seed for every batch (makes the "
            "smoke run fully reproducible). Default: sample per-batch from "
            "cfg.rl.training_seeds."
        ),
    )
    parser.add_argument(
        "--random-baseline-episodes",
        type=int,
        default=32,
        help=(
            "After training, evaluate the random-action policy on this many "
            "stage-1 episodes for the 'exceeds random' check. 0 = skip."
        ),
    )
    return parser.parse_args()


def _write_learning_curve_csv(path: Path, records: List[UpdateRecord]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    rows = [asdict(r) for r in records]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarise(records: List[UpdateRecord]) -> dict:
    if not records:
        return {"records": 0}
    returns = np.array([r.mean_undiscounted_return for r in records], dtype=np.float64)
    losses = np.array([r.total_loss for r in records], dtype=np.float64)
    window = max(5, len(records) // 5)
    first_mean = float(returns[:window].mean())
    last_mean = float(returns[-window:].mean())
    return {
        "num_updates": len(records),
        "env_steps_total": int(records[-1].env_steps_cumulative),
        "first_window_mean_return": first_mean,
        "last_window_mean_return": last_mean,
        "return_improvement": last_mean - first_mean,
        "first_window_mean_loss": float(losses[:window].mean()),
        "last_window_mean_loss": float(losses[-window:].mean()),
        "loss_improvement": float(losses[:window].mean() - losses[-window:].mean()),
        "window": window,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_root = Path(args.out_root)
    run_id = args.run_id or generate_run_id("rl_smoke")
    run_dir = ensure_run_dir(out_root, run_id)
    started = time.time()

    training_cfg = TrainConfig.from_cfg(cfg, init_seed=args.init_seed)
    if args.batch_size is not None:
        training_cfg = TrainConfig(
            **{**training_cfg.__dict__, "batch_size": int(args.batch_size)}
        )
    if args.stage == 1:
        # Keep only stage 1 and crank up num_updates so the trainer doesn't
        # run out of budget before max_env_steps is reached.
        stage_one = training_cfg.stages[0]
        stage_one = StageSpec(
            num_jobs=stage_one.num_jobs,
            max_steps=stage_one.max_steps,
            # Enough to exhaust the 10k step budget at batch_size=16,
            # horizon≈120: ceil(10000 / (16*60)) ~ 11, take a large
            # safety margin.
            num_updates=max(stage_one.num_updates, 200),
        )
        training_cfg = TrainConfig(
            **{**training_cfg.__dict__, "stages": [stage_one]}
        )

    print(f"RL smoke-test run: {run_dir}")
    print(
        f"  stages: {[(s.num_jobs, s.max_steps, s.num_updates) for s in training_cfg.stages]}"
    )
    print(
        f"  batch_size={training_cfg.batch_size}  lr={training_cfg.learning_rate}  "
        f"delta={training_cfg.delta_discount}  cH={training_cfg.entropy_coef}  "
        f"init_seed={training_cfg.init_seed}  max_env_steps={args.max_env_steps}"
    )

    policy, records = train(
        cfg=cfg,
        training_cfg=training_cfg,
        max_env_steps=args.max_env_steps,
        checkpoint_dir=None,  # smoke run; no checkpoints
        log_every=args.log_every,
        fixed_arrival_seed=args.fixed_arrival_seed,
    )

    # Save the policy checkpoint at the end of the smoke run.
    torch.save(policy.state_dict(), run_dir / "policy_final.pt")
    _write_learning_curve_csv(run_dir / "learning_curve.csv", records)
    summary = _summarise(records)

    # Evaluate a uniform-random policy on the same stage so the smoke-test
    # check "mean episode utility exceeds random" is well-defined.
    if args.random_baseline_episodes > 0:
        from src.rl.env import OrchestrationEnv

        stage = training_cfg.stages[0]
        rng = np.random.default_rng(args.init_seed if args.init_seed is not None else 7)
        random_returns = []
        env_for_random = OrchestrationEnv(
            cfg=cfg,
            num_jobs=stage.num_jobs,
            max_steps=stage.max_steps,
            memoryless=training_cfg.memoryless,
            reset_rng_seed=int(rng.integers(0, 2**31 - 1)),
        )
        seeds_for_random = [
            int(rng.integers(0, 2**31 - 1))
            for _ in range(args.random_baseline_episodes)
        ]
        for seed in seeds_for_random:
            obs, _ = env_for_random.reset(seed=seed)
            total = 0.0
            while True:
                action = int(rng.integers(0, env_for_random.action_space.n))
                obs, reward, terminated, truncated, _ = env_for_random.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            random_returns.append(total)
        random_mean = float(np.mean(random_returns)) if random_returns else 0.0
        random_std = (
            float(np.std(random_returns, ddof=1)) if len(random_returns) > 1 else 0.0
        )
        summary["random_baseline_mean_return"] = random_mean
        summary["random_baseline_std_return"] = random_std
        summary["random_baseline_num_episodes"] = len(random_returns)
        summary["learned_exceeds_random"] = (
            summary.get("last_window_mean_return", float("-inf")) > random_mean
        )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    write_resolved_config(run_dir, cfg)
    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        seeds=list(cfg.rl.training_seeds),
        extra={
            "entrypoint": "train_rl",
            "scope": "phase4_smoke",
            "stage": args.stage,
            "max_env_steps": args.max_env_steps,
            "init_seed": training_cfg.init_seed,
            "batch_size": training_cfg.batch_size,
        },
        wall_clock_start=started,
    )

    print("---")
    print(
        f"Updates: {summary.get('num_updates')}  "
        f"env_steps: {summary.get('env_steps_total')}"
    )
    print(
        f"First-window return: {summary.get('first_window_mean_return', 0):+7.3f} | "
        f"Last-window return:  {summary.get('last_window_mean_return', 0):+7.3f} | "
        f"Δ: {summary.get('return_improvement', 0):+7.3f}"
    )
    print(
        f"First-window loss:   {summary.get('first_window_mean_loss', 0):+7.3f} | "
        f"Last-window loss:    {summary.get('last_window_mean_loss', 0):+7.3f} | "
        f"Δ: {summary.get('loss_improvement', 0):+7.3f} (positive = loss decreased)"
    )
    if "random_baseline_mean_return" in summary:
        print(
            f"Random baseline mean return: "
            f"{summary['random_baseline_mean_return']:+7.3f} "
            f"(n={summary['random_baseline_num_episodes']})"
        )
        print(
            f"Learned exceeds random: {summary['learned_exceeds_random']}"
        )


if __name__ == "__main__":
    main()
