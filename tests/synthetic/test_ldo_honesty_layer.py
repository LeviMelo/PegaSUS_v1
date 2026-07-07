"""WP6 / honesty layer — propagated uncertainty, coverage manifest, exact-certifies-approximate.

MSD-III §III.8 / §V.6 / §VIII: an LDO result is honest only if every promoted edge carries
propagated uncertainty (statistical + numerical), what was NOT searched is a typed region,
and the national approximation is certified against a state-scale exact run.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.coverage import build_coverage_manifest
from pegasus.ldo.exact_certify import ApproximationRejectedError, certify_exact_vs_approx
from pegasus.ldo.records import LinkRecord


def test_coverage_manifest_records_searched_and_unsearched():
    m = build_coverage_manifest(resolution="year", K=3, requested_K=8,
                                ran_residual_scan=True, residual_error=None)
    d = m.as_manifest()
    assert d["resolution_searched"] == "year"
    assert d["lag_orders_searched"] == [0, 1, 2, 3]
    assert "linear_conditional_precision" in d["functional_forms_searched"]
    assert "nonlinear_residual_hsic" in d["functional_forms_searched"]
    kinds = {u["kind"] for u in d["unsearched"]}
    # adaptive-K truncated lags 4..8, finer resolution shrunk, interactions not enumerated
    assert {"conditioning", "resolution", "interaction"} <= kinds
    assert d["n_unsearched_regions"] >= 3


def test_coverage_manifest_flags_nonlinear_when_scan_refused():
    m = build_coverage_manifest(resolution="year", K=2, requested_K=2,
                                ran_residual_scan=True, residual_error="scan_exceeds_memory")
    forms = m.as_manifest()["functional_forms_searched"]
    assert "nonlinear_residual_hsic" not in forms  # it did not actually run
    assert any(u["kind"] == "functional_form" for u in m.as_manifest()["unsearched"])


def _edge(s, t, w, unc, lag=0):
    return LinkRecord(source_var=s, target_var=t, edge_type="lagged_directed",
                      lag_k=lag, weight=w, uncertainty=unc)


def test_exact_certifies_approximate_agreement_and_loud_rejection():
    # overlapping edges within the propagated band → certified
    approx = [_edge("A", "B", 0.50, 0.05, lag=2), _edge("C", "D", 0.30, 0.04)]
    exact = [_edge("B", "A", 0.52, 0.03, lag=2), _edge("C", "D", 0.29, 0.02)]  # A-B endpoints unordered
    rep = certify_exact_vs_approx(approx, exact, tolerance_sigma=3.0)
    assert rep["n_overlap"] == 2 and rep["certified"] and rep["n_disagree"] == 0

    # a gross disagreement beyond the band → rejected, and strict mode raises loudly
    approx_bad = [_edge("A", "B", 0.50, 0.02, lag=2)]
    exact_bad = [_edge("A", "B", 0.95, 0.02, lag=2)]
    rep_bad = certify_exact_vs_approx(approx_bad, exact_bad, tolerance_sigma=3.0)
    assert not rep_bad["certified"] and rep_bad["n_disagree"] == 1
    try:
        certify_exact_vs_approx(approx_bad, exact_bad, tolerance_sigma=3.0, strict=True)
        assert False, "strict mode must raise on disagreement"
    except ApproximationRejectedError:
        pass


def test_numerical_error_propagates_into_edge_uncertainty():
    from pegasus.ldo.lowrank import fit_sparse_plus_lowrank
    rng = np.random.default_rng(0)
    # a wider problem so the randomized-SVD path (p > 2*rank_cap) is taken
    p = 60
    A = rng.standard_normal((p, p))
    C = A @ A.T / p + np.eye(p)
    exact = fit_sparse_plus_lowrank(C, lambda1=0.1, lambda2=0.1, randomized_factors=False)
    approx = fit_sparse_plus_lowrank(C, lambda1=0.1, lambda2=0.1, randomized_factors=True, factor_rank_cap=4)
    assert exact.numerical_error == 0.0            # exact eigh → no readout error
    assert approx.numerical_error >= 0.0           # randomized readout carries a truncation bound
