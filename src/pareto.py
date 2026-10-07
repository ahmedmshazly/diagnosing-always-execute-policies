from __future__ import annotations

"""Pareto-frontier utilities and "best fixed-weight" selection.

The sweep produces one aggregate point per (alpha, beta, gamma) cell. For
each of the three 2D projections used in the paper this module computes:

1. The Pareto frontier — the set of cells that no other cell dominates.
2. A single "best fixed-weight" cell, chosen under an explicit decision
   rule. That cell becomes the RL agent's comparison baseline.

Sign conventions
----------------
Every metric in the sweep has a natural direction:

- ``completion_rate``, ``value_weighted_completion_rate``, ``mean_utility``:
  higher is better.
- ``cost``, ``failure_rate``: lower is better.

The helpers below take a list of cells and a list of ``(metric, direction)``
pairs where ``direction`` is ``"higher"`` or ``"lower"``, and return the
subset of cells that are Pareto-optimal in that projection. No row can
dominate itself; strict dominance wins ties the natural way (if two cells
are byte-for-byte identical on every axis, both survive the frontier).
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple


Direction = str  # "higher" or "lower"


def _as_better(value: float, reference: float, direction: Direction) -> bool:
    if direction == "higher":
        return value > reference
    if direction == "lower":
        return value < reference
    raise ValueError(f"unknown direction: {direction}")


def _as_not_worse(value: float, reference: float, direction: Direction) -> bool:
    if direction == "higher":
        return value >= reference
    if direction == "lower":
        return value <= reference
    raise ValueError(f"unknown direction: {direction}")


def dominates(
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float],
    axes: Sequence[Tuple[str, Direction]],
) -> bool:
    """Return True iff ``candidate`` strictly Pareto-dominates ``incumbent``.

    Strict dominance requires ``candidate`` to be no worse on every axis
    and strictly better on at least one.
    """
    strictly_better = False
    for metric, direction in axes:
        if not _as_not_worse(candidate[metric], incumbent[metric], direction):
            return False
        if _as_better(candidate[metric], incumbent[metric], direction):
            strictly_better = True
    return strictly_better


def pareto_front(
    points: Sequence[Mapping[str, float]],
    axes: Sequence[Tuple[str, Direction]],
) -> List[Mapping[str, float]]:
    """Return every ``point`` that is not dominated by any other point."""
    front: List[Mapping[str, float]] = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            if dominates(other, candidate, axes):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


@dataclass(frozen=True)
class DominanceResult:
    """Which sweep cells strictly dominate a reference point."""

    reference_name: str
    reference_point: Mapping[str, float]
    axes: Tuple[Tuple[str, Direction], ...]
    dominators: Tuple[Mapping[str, float], ...]


def cells_dominating(
    cells: Sequence[Mapping[str, float]],
    reference: Mapping[str, float],
    axes: Sequence[Tuple[str, Direction]],
    reference_name: str = "reference",
) -> DominanceResult:
    """Return the cells that strictly Pareto-dominate ``reference``."""
    dominators = tuple(cell for cell in cells if dominates(cell, reference, axes))
    return DominanceResult(
        reference_name=reference_name,
        reference_point=dict(reference),
        axes=tuple(axes),
        dominators=dominators,
    )


def best_by_metric(
    cells: Sequence[Mapping[str, float]],
    metric: str,
    direction: Direction,
) -> Mapping[str, float]:
    """Return the single cell with the best value of ``metric``.

    Ties are broken by returning the first maximiser in input order.
    """
    if not cells:
        raise ValueError("cells is empty")
    sign = 1.0 if direction == "higher" else -1.0
    best = cells[0]
    for cell in cells[1:]:
        if sign * cell[metric] > sign * best[metric]:
            best = cell
    return best


__all__ = [
    "DominanceResult",
    "best_by_metric",
    "cells_dominating",
    "dominates",
    "pareto_front",
]
