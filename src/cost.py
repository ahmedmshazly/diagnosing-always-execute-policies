from __future__ import annotations

"""Explicit cost function.

The midterm cost accounting was a branchy implicit formula split across
``estimate_step_cost`` and the utility agent's ``_resource_cost``. This
module makes it a single pure function with a written-out formula so a
reviewer can read it end-to-end.

Formula (see SPECIFICATION.md §4):

    cost(state, action) = step_cost(state) + action_cost(action, state)

    step_cost(state)       = state.cluster.spot_price
                             * (cost.cpu_weight * cpu_in_use
                                + cost.ram_weight * ram_in_use)

    action_cost(a, state)  = state.cluster.spot_price * cost.action_costs[a]

Phase-1 defaults keep every ``action_costs`` entry at 0.0 so the episode
total ``Σ_t cost(state_t, action_t)`` is exactly the old ``Σ_t step_cost``
— this preserves the ``canonical_midterm`` numerics bit-for-bit.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from .sim_environment import EpisodeState


def step_cost(state: "EpisodeState") -> float:
    """Operational (resource-usage) cost for the current step.

    Depends only on CPU / RAM usage and spot price. Does **not** include the
    action tariff (see :func:`action_cost`).
    """
    cost_cfg = state.cfg.simulator.cost
    usage = (
        cost_cfg.cpu_weight * state.cpu_in_use()
        + cost_cfg.ram_weight * state.ram_in_use()
    )
    return state.cluster.spot_price * usage


def action_cost(action: str, state: "EpisodeState") -> float:
    """Action tariff for ``action`` in the current state.

    Uses the single ``cost.action_costs`` table in the config. Raises
    ``KeyError`` on unknown actions — the loader enforces completeness, so a
    missing entry here is a programming error, not a config typo.
    """
    cost_cfg = state.cfg.simulator.cost
    tariff = cost_cfg.action_costs[action]
    return state.cluster.spot_price * tariff


def cost(state: "EpisodeState", action: str) -> float:
    """Total per-step cost charged after ``action`` is applied.

    This is the pure function the episode loop accumulates into
    ``total_compute_cost`` and that the utility's ``β * Cost`` term uses.
    """
    return step_cost(state) + action_cost(action, state)


__all__ = ["action_cost", "cost", "step_cost"]
