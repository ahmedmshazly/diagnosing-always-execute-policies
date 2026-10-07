from __future__ import annotations

"""Feed-forward policy network for the Self-Learning agent.

Paper §4.3.2: two hidden layers of 64 units with tanh, softmax output over
the six discrete actions. Value-function head intentionally omitted — the
sequence-specific baseline (§4.3.4) is the sole variance-reduction mechanism.
"""

from typing import Sequence, Tuple

import torch
from torch import nn


ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


class MLPPolicy(nn.Module):
    """Softmax policy over a discrete action set."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported activation: {activation}")
        activation_cls = ACTIVATIONS[activation]
        layers = []
        prev = state_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, int(size)))
            layers.append(activation_cls())
            prev = int(size)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_actions)

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(state))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(state), dim=-1)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits(state))

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, entropy) for a single state batch.

        ``state`` must have shape ``(N, state_dim)``; outputs are shape ``(N,)``.
        """
        dist = self.distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


__all__ = ["MLPPolicy"]
