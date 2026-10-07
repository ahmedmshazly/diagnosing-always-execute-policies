from __future__ import annotations

"""Phase-5 rollout dump — behavioural analysis.

Given an RL policy checkpoint and a small list of held-out seeds, emits a
long-form CSV with one row per simulator step containing:

- seed
- step index
- action name chosen by the RL policy
- full 8-dim state vector
- step reward, step cost, delta value, delta risk
- number of running tasks / completed jobs / failed jobs after the step
- raw logits + softmax probabilities for all six actions

Intended for the paper's §6.3 behavioural analysis — 3–5 seeds is
sufficient to show which actions the learned policy prefers in which
regimes, and how its choices differ from the Utility-Based Agent's.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.config import load_config
from src.rl.agent import load_policy
from src.rl.env import ACTIONS, STATE_FIELD_ORDER, OrchestrationEnv
from src.run_artifacts import ensure_run_dir, write_manifest, write_resolved_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy state_dict used for the rollouts.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[200, 203, 210, 225, 240],
        help="Held-out seeds to roll out. Default: 5 seeds from the test pool.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-jobs", type=int, default=None)
    return parser.parse_args()


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_dir = ensure_run_dir(args.out_root, args.run_id)
    started = time.time()
    policy = load_policy(cfg, args.checkpoint)

    state_fields = [f"state_{field}" for field in STATE_FIELD_ORDER]
    prob_fields = [f"prob_{action}" for action in ACTIONS]
    logit_fields = [f"logit_{action}" for action in ACTIONS]
    fieldnames = [
        "seed",
        "step",
        "action",
        *state_fields,
        "step_cost",
        "delta_value",
        "delta_risk",
        "reward",
        "cumulative_reward",
        "running_tasks",
        "completed_jobs",
        "failed_jobs",
        "cpu_in_use",
        "ram_in_use",
        "queue_depth",
        "spot_price",
        *prob_fields,
        *logit_fields,
    ]

    rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for seed in args.seeds:
        env = OrchestrationEnv(
            cfg=cfg,
            num_jobs=args.num_jobs,
            max_steps=args.max_steps,
            memoryless=False,
        )
        obs, _ = env.reset(seed=int(seed))
        cumulative = 0.0
        step_idx = 0
        while True:
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy.logits(state_tensor).squeeze(0)
                probs = _softmax(logits)
            action_idx = int(torch.argmax(logits).item())
            action_name = ACTIONS[action_idx]
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            cumulative += float(reward)
            row = {
                "seed": seed,
                "step": step_idx,
                "action": action_name,
                "step_cost": info["step_cost"],
                "delta_value": info["delta_value"],
                "delta_risk": info["delta_risk"],
                "reward": float(reward),
                "cumulative_reward": cumulative,
                "running_tasks": len(env._state.running_tasks),
                "completed_jobs": sum(1 for j in env._state.jobs if j.completed),
                "failed_jobs": sum(1 for j in env._state.jobs if j.failed),
                "cpu_in_use": info["cpu_in_use"],
                "ram_in_use": info["ram_in_use"],
                "queue_depth": info["queue_depth"],
                "spot_price": info["spot_price"],
            }
            for idx, field in enumerate(STATE_FIELD_ORDER):
                row[f"state_{field}"] = round(float(obs[idx]), 4)
            for idx, action in enumerate(ACTIONS):
                row[f"prob_{action}"] = round(float(probs[idx].item()), 4)
                row[f"logit_{action}"] = round(float(logits[idx].item()), 3)
            rows.append(row)
            step_idx += 1
            obs = next_obs
            if terminated or truncated:
                break
        summary_rows.append(
            {
                "seed": seed,
                "steps": step_idx,
                "cumulative_reward": round(cumulative, 4),
                "completed_jobs": sum(1 for j in env._state.jobs if j.completed),
                "failed_jobs": sum(1 for j in env._state.jobs if j.failed),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    trace_path = out_dir / "rollouts.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "rollouts_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Action histogram per seed.
    histogram_rows: List[Dict] = []
    for seed in args.seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        hist = {action: 0 for action in ACTIONS}
        for row in seed_rows:
            hist[row["action"]] += 1
        histogram_rows.append({"seed": seed, **hist})
    histogram_path = out_dir / "action_histograms.csv"
    with histogram_path.open("w", newline="", encoding="utf-8") as handle:
        if histogram_rows:
            writer = csv.DictWriter(handle, fieldnames=list(histogram_rows[0].keys()))
            writer.writeheader()
            writer.writerows(histogram_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "checkpoint": str(args.checkpoint),
                "num_steps_total": sum(row["steps"] for row in summary_rows),
                "action_histograms": histogram_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_resolved_config(out_dir, cfg)
    write_manifest(
        run_dir=out_dir,
        cfg=cfg,
        seeds=args.seeds,
        extra={
            "entrypoint": "phase5_rollout_dump",
            "checkpoint": str(args.checkpoint),
            "rollouts": len(args.seeds),
        },
        wall_clock_start=started,
    )
    print(f"Wrote {trace_path} ({len(rows)} step rows)")
    print(f"Wrote {summary_path} ({len(summary_rows)} rollout summaries)")
    print(f"Wrote {histogram_path} (action histograms)")


if __name__ == "__main__":
    main()
