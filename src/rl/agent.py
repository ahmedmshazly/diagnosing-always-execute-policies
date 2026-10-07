from __future__ import annotations

"""RLPolicyAgent: adapts a trained :class:`MLPPolicy` to the project's
agent protocol so the held-out evaluation harness can treat it like any
other agent.

The adapter:
- reads the per-step observation from :class:`EpisodeState.state_vector`
  in the same field order the training environment used,
- runs a greedy argmax over the policy's logits (sampling is reserved for
  training; evaluation is deterministic),
- returns the action name from :data:`src.sim_environment.ACTIONS`.

No exploration, no extra stochasticity. The only randomness during
evaluation comes from the simulator's event stream, which is already
seeded through :func:`src.sim_environment.make_episode_rngs`.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import RunConfig
from ..sim_environment import ACTIONS, EpisodeState
from .env import observation_dim, state_field_order
from .policy import MLPPolicy


class RLPolicyAgent:
    name = "Self-Learning Utility-Based Agent (RL)"

    def __init__(
        self,
        cfg: RunConfig,
        policy: MLPPolicy,
        deterministic: bool = True,
        label: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.policy = policy.eval()
        self.deterministic = bool(deterministic)
        self._state_field_order = state_field_order(cfg)
        if label is not None:
            self.name = label

    def _observation(self, state: EpisodeState) -> np.ndarray:
        sv = state.state_vector()
        arr = np.array(
            [getattr(sv, name) for name in self._state_field_order],
            dtype=np.float32,
        )
        return np.clip(arr, 0.0, 1.0)

    def choose_action(self, state: EpisodeState) -> str:
        obs = torch.from_numpy(self._observation(state)).float().unsqueeze(0)
        with torch.no_grad():
            logits = self.policy.logits(obs)
            if self.deterministic:
                action_index = int(torch.argmax(logits, dim=-1).item())
            else:
                distribution = torch.distributions.Categorical(logits=logits)
                action_index = int(distribution.sample().item())
        return ACTIONS[action_index]


def load_policy(
    cfg: RunConfig,
    checkpoint_path: Path,
    hidden_sizes=None,
    activation: Optional[str] = None,
    state_dim: Optional[int] = None,
) -> MLPPolicy:
    """Rebuild the policy network from a state_dict on disk.

    The input layer width is taken from :func:`observation_dim(cfg)` unless
    ``state_dim`` is passed explicitly (useful for loading a Phase-5
    checkpoint under a Phase-6 config).
    """
    from .env import NUM_ACTIONS

    sizes = tuple(hidden_sizes) if hidden_sizes is not None else tuple(cfg.rl.network.hidden_sizes)
    act = activation or cfg.rl.network.activation
    dim = int(state_dim) if state_dim is not None else observation_dim(cfg)
    policy = MLPPolicy(
        state_dim=dim,
        num_actions=NUM_ACTIONS,
        hidden_sizes=sizes,
        activation=act,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def make_rl_agent_factory(
    cfg: RunConfig,
    checkpoint_path: Path,
    deterministic: bool = True,
    label: Optional[str] = None,
):
    """Return a factory callable usable by src.runner.run_many_episodes."""
    policy = load_policy(cfg, checkpoint_path)

    def _factory(run_cfg: RunConfig) -> RLPolicyAgent:
        return RLPolicyAgent(run_cfg, policy, deterministic=deterministic, label=label)

    return _factory


__all__ = ["RLPolicyAgent", "load_policy", "make_rl_agent_factory"]
