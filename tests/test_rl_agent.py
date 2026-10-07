from __future__ import annotations

"""Tests for :class:`src.rl.agent.RLPolicyAgent` — the bridge between a
trained policy and the project's evaluation harness."""

import numpy as np
import torch

from src.config import load_config
from src.rl.agent import RLPolicyAgent
from src.rl.env import NUM_ACTIONS, NUM_STATE_FEATURES
from src.rl.policy import MLPPolicy
from src.runner import run_many_episodes
from src.sim_environment import (
    ACTIONS,
    WorkloadGenerator,
    make_episode_rngs,
)


def _fresh_state(cfg, seed=123, num_jobs=10):
    workload_rng, _ = make_episode_rngs(seed)
    return WorkloadGenerator(
        rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed
    ).generate_episode()


def test_rl_agent_returns_action_name_from_canonical_list(default_config):
    policy = MLPPolicy(
        state_dim=NUM_STATE_FEATURES, num_actions=NUM_ACTIONS, hidden_sizes=(8, 8)
    )
    agent = RLPolicyAgent(default_config, policy)
    action = agent.choose_action(_fresh_state(default_config))
    assert action in ACTIONS


def test_rl_agent_is_deterministic_by_default(default_config):
    policy = MLPPolicy(
        state_dim=NUM_STATE_FEATURES, num_actions=NUM_ACTIONS, hidden_sizes=(8, 8)
    )
    agent = RLPolicyAgent(default_config, policy)
    state = _fresh_state(default_config)
    # Same state object -> same argmax -> same action.
    first = agent.choose_action(state)
    second = agent.choose_action(state)
    assert first == second


def test_rl_agent_runs_an_episode_via_the_project_runner(default_config):
    """The adapter must plug into src.runner.run_many_episodes."""
    policy = MLPPolicy(
        state_dim=NUM_STATE_FEATURES, num_actions=NUM_ACTIONS, hidden_sizes=(8, 8)
    )

    def factory(cfg):
        return RLPolicyAgent(cfg, policy)

    metrics = run_many_episodes(
        cfg=default_config, agent_factory=factory, seeds=[100, 101], include_uncapped=False
    )
    assert len(metrics) == 2
    for record in metrics:
        assert 0.0 <= record.completion_rate <= 1.0
        assert record.steps_executed >= 1


def test_load_policy_round_trip(default_config, tmp_path):
    from src.rl.agent import load_policy

    policy = MLPPolicy(
        state_dim=NUM_STATE_FEATURES,
        num_actions=NUM_ACTIONS,
        hidden_sizes=tuple(default_config.rl.network.hidden_sizes),
        activation=default_config.rl.network.activation,
    )
    checkpoint_path = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), checkpoint_path)
    restored = load_policy(default_config, checkpoint_path)

    x = torch.from_numpy(np.ones(NUM_STATE_FEATURES, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        assert torch.allclose(policy.logits(x), restored.logits(x))
