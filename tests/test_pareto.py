from __future__ import annotations

"""Pareto utility tests.

Covers strict dominance, frontier extraction, and the "best by metric"
helper against small hand-authored fixtures.
"""

import pytest

from src.pareto import best_by_metric, cells_dominating, dominates, pareto_front


AXES_CC = (("completion", "higher"), ("cost", "lower"))
AXES_CCF = (
    ("completion", "higher"),
    ("cost", "lower"),
    ("failure", "lower"),
)


def _p(completion: float, cost: float, failure: float = 0.0, **extra):
    return {"completion": completion, "cost": cost, "failure": failure, **extra}


def test_dominates_strict_when_better_on_one_and_tied_on_others():
    a = _p(completion=0.6, cost=100.0)
    b = _p(completion=0.6, cost=120.0)
    assert dominates(a, b, AXES_CC) is True
    assert dominates(b, a, AXES_CC) is False


def test_dominates_false_when_worse_on_any_axis():
    a = _p(completion=0.7, cost=150.0)
    b = _p(completion=0.6, cost=100.0)
    assert dominates(a, b, AXES_CC) is False
    assert dominates(b, a, AXES_CC) is False  # a is better on completion, b on cost


def test_dominates_requires_strict_improvement():
    a = _p(completion=0.6, cost=100.0)
    b = _p(completion=0.6, cost=100.0)
    assert dominates(a, b, AXES_CC) is False
    assert dominates(b, a, AXES_CC) is False


def test_pareto_front_filters_dominated_points():
    points = [
        _p(completion=0.7, cost=300.0, label="high_comp_high_cost"),
        _p(completion=0.5, cost=100.0, label="low_comp_low_cost"),
        _p(completion=0.6, cost=200.0, label="interior_not_dominated"),
        _p(completion=0.55, cost=250.0, label="strictly_dominated"),
    ]
    front = pareto_front(points, axes=AXES_CC)
    labels = {cell["label"] for cell in front}
    assert "high_comp_high_cost" in labels
    assert "low_comp_low_cost" in labels
    assert "interior_not_dominated" in labels
    assert "strictly_dominated" not in labels


def test_pareto_front_keeps_tied_points_if_not_strictly_dominated():
    points = [
        _p(completion=0.6, cost=100.0, label="a"),
        _p(completion=0.6, cost=100.0, label="b"),
    ]
    front = pareto_front(points, axes=AXES_CC)
    labels = sorted(cell["label"] for cell in front)
    assert labels == ["a", "b"]


def test_best_by_metric_returns_maximiser():
    points = [_p(completion=0.5, cost=100.0), _p(completion=0.7, cost=120.0)]
    best = best_by_metric(points, "completion", "higher")
    assert best["completion"] == 0.7


def test_best_by_metric_returns_minimiser_on_lower():
    points = [_p(completion=0.5, cost=100.0), _p(completion=0.7, cost=120.0)]
    best = best_by_metric(points, "cost", "lower")
    assert best["cost"] == 100.0


def test_cells_dominating_reference():
    reference = _p(completion=0.5, cost=500.0, failure=0.15)
    cells = [
        _p(completion=0.6, cost=400.0, failure=0.10, label="dominator_1"),
        _p(completion=0.5, cost=500.0, failure=0.10, label="dominator_2"),
        _p(completion=0.4, cost=400.0, failure=0.10, label="not_dominator"),  # worse on completion
    ]
    result = cells_dominating(cells, reference, axes=AXES_CCF, reference_name="test")
    labels = sorted(cell["label"] for cell in result.dominators)
    assert labels == ["dominator_1", "dominator_2"]


def test_unknown_direction_raises():
    with pytest.raises(ValueError):
        dominates({"x": 1.0}, {"x": 0.0}, (("x", "neither"),))
