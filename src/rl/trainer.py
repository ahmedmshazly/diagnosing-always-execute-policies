from __future__ import annotations

"""REINFORCE-with-baseline trainer for the Self-Learning Utility-Based Agent.

Implements paper §4.3.1–§4.3.5:
- Policy-gradient update per Eq. 5 / 6.
- Entropy regularisation with coefficient cH.
- Sequence-specific per-timestep baseline (§4.3.4).
- Memoryless termination (tau ~ Exp(mu)) inside each batch episode.
- Three-stage curriculum over (N, T_max) with per-stage update counts.
- No value-function head (paper §4.3.1).
- Gradient-clip norm, Adam optimiser.

This module exposes one public entry point, :func:`train`, and a handful of
small helpers used by the tests. Checkpoint I/O and validation-pool
evaluation are implemented here so Phase-5's full run is a matter of
wall-clock time rather than additional engineering.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim

from ..config import RunConfig
from .baseline import discounted_returns, sequence_specific_baseline
from .env import NUM_ACTIONS, OrchestrationEnv, observation_dim
from .policy import MLPPolicy


@dataclass
class StageSpec:
    num_jobs: int
    max_steps: int
    num_updates: int


@dataclass
class TrainConfig:
    learning_rate: float
    entropy_coef: float
    delta_discount: float
    batch_size: int
    grad_clip_norm: float
    stages: List[StageSpec]
    network_hidden_sizes: Tuple[int, ...]
    network_activation: str
    init_seed: int = 7
    eval_every_updates: int = 25
    checkpoint_every_updates: int = 50
    memoryless: bool = True

    @classmethod
    def from_cfg(cls, cfg: RunConfig, init_seed: Optional[int] = None) -> "TrainConfig":
        stages = [
            StageSpec(
                num_jobs=stage.num_jobs,
                max_steps=stage.max_steps,
                num_updates=stage.num_updates,
            )
            for stage in cfg.rl.curriculum.stages
        ]
        if init_seed is None:
            init_seed = int(cfg.rl.initialisation_seeds[0])
        return cls(
            learning_rate=cfg.rl.learning_rate,
            entropy_coef=cfg.rl.entropy_coef,
            delta_discount=cfg.rl.delta_discount,
            batch_size=cfg.rl.batch_size,
            grad_clip_norm=cfg.rl.grad_clip_norm,
            stages=stages,
            network_hidden_sizes=tuple(cfg.rl.network.hidden_sizes),
            network_activation=cfg.rl.network.activation,
            init_seed=init_seed,
            eval_every_updates=cfg.rl.eval_every_updates,
            checkpoint_every_updates=cfg.rl.checkpoint_every_updates,
            memoryless=cfg.rl.memoryless_termination,
        )


@dataclass
class UpdateRecord:
    """One row of the learning-curve CSV."""

    update: int
    stage: int
    num_jobs: int
    max_steps: int
    batch_arrival_seed: int
    mean_undiscounted_return: float
    mean_discounted_return: float
    policy_loss: float
    entropy: float
    total_loss: float
    grad_norm: float
    env_steps_this_update: int
    env_steps_cumulative: int


@dataclass
class ValidationRecord:
    """One row of the validation_log CSV."""

    update: int
    stage: int
    env_steps_cumulative: int
    num_seeds: int
    mean_return: float
    std_return: float
    is_new_best: bool
    best_mean_return_so_far: float


@dataclass
class EpisodeRollout:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)
    horizon: int = 0

    @property
    def length(self) -> int:
        return len(self.actions)


def _run_episode(
    env: OrchestrationEnv,
    policy: MLPPolicy,
    seed: int,
) -> EpisodeRollout:
    """Run one episode collecting log-probs and entropies for gradient use."""
    observation, info = env.reset(seed=seed)
    rollout = EpisodeRollout(horizon=int(info.get("horizon", env.max_steps)))
    state_tensor = torch.from_numpy(observation).float().unsqueeze(0)
    while True:
        dist = policy.distribution(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        action_item = int(action.item())
        next_obs, reward, terminated, truncated, _ = env.step(action_item)
        rollout.states.append(observation)
        rollout.actions.append(action_item)
        rollout.rewards.append(float(reward))
        rollout.log_probs.append(log_prob.squeeze(0))
        rollout.entropies.append(entropy.squeeze(0))
        if terminated or truncated:
            break
        observation = next_obs
        state_tensor = torch.from_numpy(observation).float().unsqueeze(0)
    return rollout


def _compute_update_loss(
    rollouts: List[EpisodeRollout],
    discount: float,
    entropy_coef: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the REINFORCE-with-baseline loss for a batch."""
    if not rollouts:
        raise ValueError("rollouts is empty")

    returns_per_episode = [
        discounted_returns(ep.rewards, discount) for ep in rollouts
    ]
    baseline_by_t = sequence_specific_baseline(returns_per_episode)

    policy_loss_terms: List[torch.Tensor] = []
    entropy_terms: List[torch.Tensor] = []
    total_steps = 0

    for ep, returns in zip(rollouts, returns_per_episode):
        if ep.length == 0:
            continue
        advantages = torch.tensor(
            returns - baseline_by_t[: ep.length],
            dtype=torch.float32,
        )
        log_probs = torch.stack(ep.log_probs)
        entropies = torch.stack(ep.entropies)
        policy_loss_terms.append(-(log_probs * advantages).sum())
        entropy_terms.append(entropies.sum())
        total_steps += ep.length

    if total_steps == 0:
        # Every episode terminated before taking a step; degenerate batch.
        return torch.tensor(0.0, requires_grad=True), {
            "policy_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "env_steps": 0,
        }

    policy_loss = torch.stack(policy_loss_terms).sum() / float(total_steps)
    entropy_term = torch.stack(entropy_terms).sum() / float(total_steps)
    total_loss = policy_loss - entropy_coef * entropy_term

    metrics = {
        "policy_loss": float(policy_loss.detach().item()),
        "entropy": float(entropy_term.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "env_steps": int(total_steps),
    }
    return total_loss, metrics


def train(
    cfg: RunConfig,
    training_cfg: TrainConfig,
    env_factory=None,
    max_env_steps: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    log_every: int = 1,
    rng_seed: Optional[int] = None,
    fixed_arrival_seed: Optional[int] = None,
    validation_seeds: Optional[Iterable[int]] = None,
    validation_num_jobs: Optional[int] = None,
    validation_max_steps: Optional[int] = None,
    validation_log: Optional[List["ValidationRecord"]] = None,
    best_checkpoint_path: Optional[Path] = None,
) -> Tuple[MLPPolicy, List[UpdateRecord]]:
    """REINFORCE-with-baseline training loop.

    Returns the final policy and the list of per-update learning-curve
    records. When ``max_env_steps`` is set the loop stops as soon as the
    cumulative env-step count crosses the threshold (used for smoke tests).
    """
    torch.manual_seed(int(training_cfg.init_seed))
    np.random.seed(int(training_cfg.init_seed))

    policy = MLPPolicy(
        state_dim=observation_dim(cfg),
        num_actions=NUM_ACTIONS,
        hidden_sizes=training_cfg.network_hidden_sizes,
        activation=training_cfg.network_activation,
    )
    optimiser = optim.Adam(policy.parameters(), lr=training_cfg.learning_rate)

    arrival_rng = np.random.default_rng(rng_seed or training_cfg.init_seed)
    training_seed_pool = list(cfg.rl.training_seeds)
    if not training_seed_pool:
        raise ValueError("rl.training_seeds is empty")

    records: List[UpdateRecord] = []
    cumulative_steps = 0
    update_counter = 0

    val_seed_list = list(validation_seeds) if validation_seeds is not None else None
    val_num_jobs = (
        validation_num_jobs
        if validation_num_jobs is not None
        else cfg.experiment.num_jobs
    )
    val_max_steps = (
        validation_max_steps
        if validation_max_steps is not None
        else cfg.experiment.max_steps
    )
    best_mean_val = -float("inf")

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for stage_idx, stage in enumerate(training_cfg.stages):
        for _ in range(stage.num_updates):
            # Pick one arrival seed per batch; all B episodes share it.
            if fixed_arrival_seed is not None:
                batch_seed = int(fixed_arrival_seed)
            else:
                batch_seed = int(
                    training_seed_pool[int(arrival_rng.integers(0, len(training_seed_pool)))]
                )
            env = OrchestrationEnv(
                cfg=cfg,
                num_jobs=stage.num_jobs,
                max_steps=stage.max_steps,
                memoryless=training_cfg.memoryless,
                reset_rng_seed=int(arrival_rng.integers(0, 2**31 - 1)),
            )
            rollouts = [
                _run_episode(env, policy, batch_seed)
                for _ in range(training_cfg.batch_size)
            ]
            if max_env_steps is not None and cumulative_steps >= max_env_steps:
                break

            total_loss, metrics = _compute_update_loss(
                rollouts=rollouts,
                discount=training_cfg.delta_discount,
                entropy_coef=training_cfg.entropy_coef,
            )
            optimiser.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy.parameters(), training_cfg.grad_clip_norm
            ).item()
            optimiser.step()

            undiscounted_returns = np.array(
                [sum(ep.rewards) for ep in rollouts], dtype=np.float64
            )
            discounted = np.array(
                [
                    float(discounted_returns(ep.rewards, training_cfg.delta_discount)[0])
                    if ep.length > 0
                    else 0.0
                    for ep in rollouts
                ],
                dtype=np.float64,
            )

            record = UpdateRecord(
                update=update_counter,
                stage=stage_idx,
                num_jobs=stage.num_jobs,
                max_steps=stage.max_steps,
                batch_arrival_seed=batch_seed,
                mean_undiscounted_return=float(undiscounted_returns.mean()),
                mean_discounted_return=float(discounted.mean()),
                policy_loss=metrics["policy_loss"],
                entropy=metrics["entropy"],
                total_loss=metrics["total_loss"],
                grad_norm=float(grad_norm),
                env_steps_this_update=metrics["env_steps"],
                env_steps_cumulative=cumulative_steps + metrics["env_steps"],
            )
            records.append(record)

            if log_every and update_counter % log_every == 0:
                print(
                    f"  [update {update_counter:4d}] stage={stage_idx} "
                    f"mean_return={record.mean_undiscounted_return:+7.3f}  "
                    f"loss={record.total_loss:+8.4f}  "
                    f"entropy={record.entropy:.4f}  "
                    f"steps={record.env_steps_cumulative}"
                )

            cumulative_steps = record.env_steps_cumulative
            update_counter += 1

            if (
                checkpoint_dir is not None
                and training_cfg.checkpoint_every_updates > 0
                and update_counter % training_cfg.checkpoint_every_updates == 0
            ):
                torch.save(
                    policy.state_dict(),
                    checkpoint_dir / f"policy_update_{update_counter:06d}.pt",
                )

            # Validation-pool evaluation on the configured cadence.
            if (
                val_seed_list is not None
                and training_cfg.eval_every_updates > 0
                and update_counter % training_cfg.eval_every_updates == 0
            ):
                val_stats = evaluate(
                    cfg=cfg,
                    policy=policy,
                    seeds=val_seed_list,
                    num_jobs=val_num_jobs,
                    max_steps=val_max_steps,
                )
                is_new_best = val_stats["mean_return"] > best_mean_val
                if is_new_best:
                    best_mean_val = float(val_stats["mean_return"])
                    if best_checkpoint_path is not None:
                        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(policy.state_dict(), best_checkpoint_path)
                if validation_log is not None:
                    validation_log.append(
                        ValidationRecord(
                            update=update_counter,
                            stage=stage_idx,
                            env_steps_cumulative=cumulative_steps,
                            num_seeds=val_stats["num_episodes"],
                            mean_return=val_stats["mean_return"],
                            std_return=val_stats["std_return"],
                            is_new_best=is_new_best,
                            best_mean_return_so_far=best_mean_val,
                        )
                    )
                if log_every:
                    marker = "*" if is_new_best else " "
                    print(
                        f"    [val @ update {update_counter:4d}] "
                        f"mean={val_stats['mean_return']:+8.3f} "
                        f"(std {val_stats['std_return']:.3f}, n={val_stats['num_episodes']}) "
                        f"best-so-far={best_mean_val:+8.3f} {marker}"
                    )

            if max_env_steps is not None and cumulative_steps >= max_env_steps:
                break
        if max_env_steps is not None and cumulative_steps >= max_env_steps:
            break

    return policy, records


def evaluate(
    cfg: RunConfig,
    policy: MLPPolicy,
    seeds: Iterable[int],
    num_jobs: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Deterministic eval on a seed pool: returns mean undiscounted reward."""
    policy.eval()
    env = OrchestrationEnv(
        cfg=cfg,
        num_jobs=num_jobs,
        max_steps=max_steps,
        memoryless=False,
    )
    results: List[float] = []
    with torch.no_grad():
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed))
            total_reward = 0.0
            while True:
                state_tensor = torch.from_numpy(observation).float().unsqueeze(0)
                logits = policy.logits(state_tensor)
                # Greedy at eval time to reduce variance.
                action = int(torch.argmax(logits, dim=-1).item())
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            results.append(total_reward)
    policy.train()
    return {
        "mean_return": float(np.mean(results)) if results else 0.0,
        "std_return": float(np.std(results, ddof=1)) if len(results) > 1 else 0.0,
        "num_episodes": len(results),
    }


__all__ = [
    "EpisodeRollout",
    "StageSpec",
    "TrainConfig",
    "UpdateRecord",
    "evaluate",
    "train",
]
