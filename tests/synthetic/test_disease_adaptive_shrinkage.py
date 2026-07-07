"""P2 — adaptive (empirical-Bayes) disease-tree τ² shrinkage (§III.3).

The fixed per-scale precisions shrink genuinely-different diseases toward each other at a hardcoded
strength. Estimate that strength per block FROM THE DATA so a divergent block overrides the prior,
flag heavily-shrunk blocks, and verify the operator only smooths cross-disease structure — never a
variable's own level. The non-adaptive default must reproduce the original operator byte-for-byte.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.disease_prior import (
    estimate_disease_scale_precisions,
    heavily_shrunk_blocks,
    hierarchical_disease_laplacians_from_affinity,
    sum_of_scales_disease_operator,
)


class _Graph:
    """Minimal structural DiseaseGraph over 4 category-siblings, so variable_affinity keys by id."""

    legality_class = "structural"

    def __init__(self, codes):
        self.codes = tuple(codes)
        n = len(codes)
        self._A = np.ones((n, n)) - np.eye(n)  # all same-category (affinity 1.0)

    def adjacency(self):
        import scipy.sparse as sp
        return sp.csr_matrix(self._A)

    def as_prior(self):
        raise AssertionError("structural graph must not be refused")


def _similar_and_divergent():
    rng = np.random.default_rng(0)
    similar = ("S0", "S1"), _Graph(("S0", "S1"))
    divergent = ("D0", "D1"), _Graph(("D0", "D1"))
    # similar block: two diseases sharing one level, only sampling noise separates them
    sim_vals = {"S0": rng.normal(3.0, 0.05, 400), "S1": rng.normal(3.0, 0.05, 400)}
    # divergent block: genuinely different levels, dispersion ≫ noise
    div_vals = {"D0": rng.normal(0.0, 0.05, 400), "D1": rng.normal(9.0, 0.05, 400)}
    return similar, sim_vals, divergent, div_vals


def test_adaptive_tau2_lets_data_override_the_prior():
    (sim_vars, sim_g), sim_vals, (div_vars, div_g), div_vals = _similar_and_divergent()

    tau_sim = estimate_disease_scale_precisions(sim_vals, sim_vars, sim_g)["category"]
    tau_div = estimate_disease_scale_precisions(div_vals, div_vars, div_g)["category"]

    # genuinely-similar diseases → strong shrinkage; genuinely-different → the data wins (τ²→0)
    assert tau_sim > 0.9
    assert tau_div < 0.1
    assert tau_sim > tau_div + 0.5


def test_adaptive_operator_differs_from_fixed_on_divergent_data():
    _, _, (div_vars, div_g), div_vals = _similar_and_divergent()
    fixed = sum_of_scales_disease_operator(div_vars, div_g, {"category": 1.0})
    adaptive = sum_of_scales_disease_operator(
        div_vars, div_g, adaptive=True, field_values_by_variable=div_vals
    )
    # the fixed prior shrinks the divergent pair hard; adaptive nearly releases it
    assert not np.allclose(fixed, adaptive)
    assert abs(adaptive[0, 1]) < abs(fixed[0, 1])


def test_default_path_is_byte_identical_to_original():
    # original operator = Σ τ·L over the per-scale Laplacians; reproduce it directly and compare.
    (sim_vars, sim_g), _, _, _ = _similar_and_divergent()
    precisions = {"category": 2.0, "block": 0.5, "chapter": 0.1}
    got = sum_of_scales_disease_operator(sim_vars, sim_g, precisions)

    from pegasus.ldo.disease_prior import variable_affinity
    laps = hierarchical_disease_laplacians_from_affinity(variable_affinity(sim_vars, sim_g))
    expect = sum(precisions.get(name, 0.0) * L for name, L in laps.items())
    assert np.array_equal(got, expect)


def test_operator_touches_structure_not_level():
    # requirement (1): xᵀ G_D x is invariant to adding a constant to every disease's level, so the
    # prior smooths cross-disease structure and never shrinks a variable's own level.
    (sim_vars, sim_g), _, _, _ = _similar_and_divergent()
    G = sum_of_scales_disease_operator(sim_vars, sim_g, {"category": 1.0})
    assert np.allclose(G.sum(axis=1), 0.0)                       # zero row sums
    x = np.array([1.3, -0.7])
    for c in (0.0, 5.0, -2.0):
        assert np.isclose((x + c) @ G @ (x + c), x @ G @ x)      # level-shift invariant


def test_heavily_shrunk_blocks_are_flagged():
    (sim_vars, sim_g), sim_vals, (div_vars, div_g), div_vals = _similar_and_divergent()
    flagged_sim = heavily_shrunk_blocks(sim_vals, sim_vars, sim_g, threshold=0.75)
    flagged_div = heavily_shrunk_blocks(div_vals, div_vars, div_g, threshold=0.75)

    assert len(flagged_sim) == 1 and flagged_sim[0]["scale"] == "category"
    assert set(flagged_sim[0]["variables"]) == {"S0", "S1"} and flagged_sim[0]["tau2"] > 0.75
    assert flagged_div == []                                     # divergent block is NOT over-shrunk
