import numpy as np
import pytest
from scripts.sncs_report import holm, family, read_csv, probe_report


def test_holm_keeps_identical_comparisons_in_family():
    p = [0.001, 0.02, 0.03, 1.0]
    np.testing.assert_allclose(holm(p), [0.004, 0.06, 0.06, 1.0])


def test_holm_is_order_invariant():
    p = np.array([.04, .001, .01, 1, .3])
    order = [3, 2, 4, 0, 1]
    np.testing.assert_allclose(holm(p[order]), holm(p)[order])


def test_holm_rejects_unexplained_missing_p():
    with pytest.raises(ValueError):
        holm([.02, np.nan, 1])


def test_full_family_and_duplicate_reference():
    rows = family(read_csv("results/hardening/taxonomy_n250.csv"))
    assert len(rows) == 34
    assert sum(r["algo"] == "hand" for r in rows) == 4
    benign = next(r for r in rows if r["algo"] == "hand" and r["env"] == "default")
    assert benign["p_holm_34"] > .05
    zeros = [r for r in rows if float(r["delta_vs_ae"]) == 0]
    assert zeros and all(r["p"] == 1 for r in zeros)


def test_probe_clusters_are_episodes():
    probe = probe_report("results/hardening/local_optimality.csv")
    assert len(probe) == 5
    assert all(r["source_episodes"] == 5 and r["sampled_states"] == 75 for r in probe)
    assert all(r["cluster_ci_low"] <= r["mean"] <= r["cluster_ci_high"] for r in probe)
