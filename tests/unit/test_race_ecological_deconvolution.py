"""Proof-of-capability for the ecological race deconvolution (MSD §4.6).

One focused test per capability (not a battery, per project guidance). This is the planted-signal
study the user required *before* implementation, promoted from scratchpad/racebridge_validation_probe.py:
the estimator must recover a genuine racial rate signal that a naive fixed crosswalk ERASES, must not
itself erase it under a reasonable prior, and must degrade honestly when the prior is over-strong or
the cells are not contextually identified.
"""
from __future__ import annotations

import numpy as np
import pytest

from pegasus.measurement.race_ecological import (
    EcologicalRaceError,
    EcologicalRaceProblem,
    _neg_log_post_and_grad,
    _softmax_cols,
    contextual_identifiability,
    emission_from_reclassification,
    fit_ecological_race_deconvolution,
    reclassification_from_emission,
)

CATS = ["branca", "preta", "amarela", "parda", "indigena"]
BASE_COMP = np.array([0.43, 0.10, 0.45, 0.01, 0.01])
R_TRUE = np.array([1.0, 1.5, 1.2, 0.9, 1.3])  # planted genuine race rate-ratios; branca = reference
J = K = 5


def _C_whitening() -> np.ndarray:
    C = np.zeros((K, J))
    C[:, 0] = [0.95, 0.01, 0.03, 0.005, 0.005]
    C[:, 1] = [0.12, 0.68, 0.18, 0.01, 0.01]  # 32% of true preta mislabelled whiter
    C[:, 2] = [0.25, 0.02, 0.70, 0.02, 0.01]  # 25% of true parda -> admin branca
    C[:, 3] = [0.05, 0.01, 0.04, 0.88, 0.02]
    C[:, 4] = [0.05, 0.02, 0.10, 0.03, 0.80]
    return C / C.sum(axis=0, keepdims=True)


def _C_mild() -> np.ndarray:
    C = np.eye(K) * 0.9 + 0.025
    C[0, 1] += 0.03
    C[0, 2] += 0.05
    return C / C.sum(axis=0, keepdims=True)


def _gen(S, popmean, alpha, C_list, seed, base_rate=0.02):
    r = np.random.default_rng(seed)
    comp = r.dirichlet(alpha * BASE_COMP, size=S)
    pop = r.uniform(popmean * 0.5, popmean * 1.5, size=S)
    N = comp * pop[:, None]
    a_s = np.log(base_rate) + 0.3 * r.standard_normal(S)
    m = (np.exp(a_s)[:, None] * R_TRUE[None, :]) * N
    Ys = [r.poisson(m @ C.T).astype(float) for C in C_list]
    return Ys, N


def _problem(N, Ys, labels):
    return EcologicalRaceProblem(
        N=N,
        Y_by_stratum=Ys,
        self_declared_categories=CATS,
        admin_categories=CATS,
        stratum_labels=labels,
    )


def test_analytic_gradient_matches_finite_difference():
    # Small cells so the Poisson gradient magnitude (~ total counts) stays modest; central
    # differences (O(eps^2) truncation) + a relative threshold make this a rigorous check.
    Ys, N = _gen(20, 4000, 40.0, [_C_whitening()], seed=5)
    Z0 = np.log(np.eye(K) * 0.85 + 0.03)
    t = np.random.default_rng(0).normal(0, 0.3, K * J + 20 + J - 1)
    _, g = _neg_log_post_and_grad(t, Ys, N, Z0, 2.0)
    num = np.zeros_like(t)
    eps = 1e-6
    for i in range(len(t)):
        tp, tm = t.copy(), t.copy()
        tp[i] += eps
        tm[i] -= eps
        num[i] = (_neg_log_post_and_grad(tp, Ys, N, Z0, 2.0)[0]
                  - _neg_log_post_and_grad(tm, Ys, N, Z0, 2.0)[0]) / (2 * eps)
    rel = np.abs(num - g).max() / (np.abs(g).max() + 1e-8)
    assert rel < 1e-6, f"relative gradient error {rel:.2e}"


def test_recovers_planted_signal_that_naive_crosswalk_erases():
    # High contextual variation (alpha=5): the identifiability regime.
    Ys, N = _gen(300, 60000, 5.0, [_C_whitening()], seed=0)

    # The naive fixed-crosswalk estimand (admin counts over census) erases/inverts the signal.
    naive = Ys[0].sum(axis=0) / N.sum(axis=0)
    naive = naive / naive[0]
    assert naive[1] < 1.2, f"naive should erase the 1.5 preta signal, got {naive[1]:.2f}"

    res = fit_ecological_race_deconvolution(_problem(N, Ys, ["SIM-DO"]), prior_strength=2.0)
    assert res.identifiable
    assert res.converged
    # The deconvolution recovers the planted 1.50 that the naive estimator lost.
    assert 1.30 <= res.rate_ratios["preta"] <= 1.70
    assert res.rate_ratios["preta"] > naive[1] + 0.3
    assert res.rate_ratios["branca"] == pytest.approx(1.0)  # reference pinned
    # latent per-capita rate is emitted per cell x self-declared race
    assert res.latent_rate.shape == (300, J)
    assert np.all(res.latent_rate > 0)


def test_over_strong_identity_prior_re_erases_signal():
    # kappa=200 forces C -> identity (assumes no confusion), so the model cannot deconvolve:
    # this is the failure mode the user fears; it must be demonstrable, not hidden.
    Ys, N = _gen(300, 60000, 5.0, [_C_whitening()], seed=0)
    weak = fit_ecological_race_deconvolution(_problem(N, Ys, ["SIM-DO"]), prior_strength=2.0)
    strong = fit_ecological_race_deconvolution(_problem(N, Ys, ["SIM-DO"]), prior_strength=200.0)
    assert strong.rate_ratios["preta"] < weak.rate_ratios["preta"] - 0.2
    assert strong.rate_ratios["preta"] < 1.30  # pulled back toward the erased naive value


def test_two_strata_shared_rates_different_confusion():
    # Covariate-dependent confusion in its identifiable form: two strata (e.g. two systems / periods)
    # share the rate process but confuse differently. Cross-stratum consistency aids recovery and
    # each stratum gets its own emission matrix.
    Ys, N = _gen(300, 60000, 5.0, [_C_whitening(), _C_mild()], seed=0)
    res = fit_ecological_race_deconvolution(_problem(N, Ys, ["SIM-DO", "SIH"]), prior_strength=2.0)
    assert 1.30 <= res.rate_ratios["preta"] <= 1.70
    assert set(res.emission_by_stratum) == {"SIM-DO", "SIH"}
    for C in res.emission_by_stratum.values():
        assert C.shape == (K, J)
        np.testing.assert_allclose(C.sum(axis=0), np.ones(J), atol=1e-6)  # column-stochastic


def test_identifiability_flag_tracks_contextual_variation():
    # High composition variance -> identifiable; near-constant composition -> flagged non-identifiable.
    _, N_hi = _gen(300, 60000, 5.0, [_C_whitening()], seed=1)
    _, N_lo = _gen(300, 60000, 5000.0, [_C_whitening()], seed=1)
    hi = contextual_identifiability(N_hi)
    lo = contextual_identifiability(N_lo)
    assert hi > lo
    assert lo < 0.005 <= hi


def test_bootstrap_reports_finite_positive_cv():
    # Exercise the parametric-bootstrap uncertainty path on a small problem.
    Ys, N = _gen(40, 60000, 5.0, [_C_whitening()], seed=3)
    res = fit_ecological_race_deconvolution(
        _problem(N, Ys, ["SIM-DO"]), prior_strength=2.0, bootstrap_replicates=6
    )
    assert res.uncertainty_method == "parametric_bootstrap_6"
    assert res.rate_ratio_cv is not None
    assert set(res.rate_ratio_cv) == set(CATS)
    for c, v in res.rate_ratio_cv.items():
        assert np.isfinite(v) and v >= 0.0


def test_column_softmax_is_column_stochastic():
    Z = np.random.default_rng(0).normal(0, 1.0, (4, 3))
    C = _softmax_cols(Z)
    np.testing.assert_allclose(C.sum(axis=0), np.ones(3), atol=1e-12)
    assert np.all(C > 0)


def test_emission_from_reclassification_roundtrips_exactly():
    # Bayes bridge: construct R (row-stochastic P(self|admin)) + p_admin FROM a known emission C_true,
    # then recover C_true. This is the W-RACE-2-wire reconciliation (registry direction -> estimator
    # direction); it must invert exactly.
    rng = np.random.default_rng(0)
    Kd = Jd = 5
    C_true = rng.dirichlet(np.ones(Kd), size=Jd).T          # (K, J), each column sums to 1
    p_self = rng.dirichlet(np.ones(Jd) * 2.0)               # P(self=j)
    joint = C_true * p_self[None, :]                        # P(admin=k, self=j)
    p_admin = joint.sum(axis=1)                             # P(admin=k)
    R = joint / p_admin[:, None]                            # P(self=j | admin=k), row-stochastic
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-12)

    C_rec = emission_from_reclassification(R, p_admin)
    np.testing.assert_allclose(C_rec, C_true, atol=1e-9)          # exact inversion
    np.testing.assert_allclose(C_rec.sum(axis=0), 1.0, atol=1e-9)  # column-stochastic


def test_emission_from_reclassification_accepts_unnormalised_counts():
    # p_admin may be raw admin counts; it is normalised internally.
    R = np.array([[0.8, 0.2], [0.3, 0.7]])
    C_from_counts = emission_from_reclassification(R, np.array([4000.0, 1000.0]))
    C_from_probs = emission_from_reclassification(R, np.array([0.8, 0.2]))
    np.testing.assert_allclose(C_from_counts, C_from_probs, atol=1e-12)


def test_emission_from_reclassification_rejects_non_row_stochastic():
    bad = np.array([[0.8, 0.3], [0.3, 0.7]])  # first row sums to 1.1
    with pytest.raises(EcologicalRaceError):
        emission_from_reclassification(bad, np.array([0.5, 0.5]))


def test_reclassification_from_emission_is_the_inverse_bridge():
    # C (emission P(admin|self)) + census self marginal -> R (reclassification P(self|admin), the
    # registry direction). Then round-trip back through emission_from_reclassification with the
    # implied admin marginal must recover C. This is the write-back a calibrated prior uses.
    rng = np.random.default_rng(1)
    Kd = Jd = 5
    C_true = rng.dirichlet(np.ones(Kd), size=Jd).T   # column-stochastic
    p_self = rng.dirichlet(np.ones(Jd) * 2.0)
    R = reclassification_from_emission(C_true, p_self)
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-9)     # row-stochastic P(self|admin)
    p_admin = (C_true * p_self[None, :]).sum(axis=1)              # implied P(admin=k)
    C_back = emission_from_reclassification(R, p_admin)
    np.testing.assert_allclose(C_back, C_true, atol=1e-9)         # converters compose to identity


def test_reclassification_from_emission_rejects_non_column_stochastic():
    bad = np.array([[0.9, 0.2], [0.3, 0.7]])  # first column sums to 1.2
    with pytest.raises(EcologicalRaceError):
        reclassification_from_emission(bad, np.array([0.5, 0.5]))
