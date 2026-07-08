"""ADVERSARIAL AUDIT — W1SE + FDR partial-correlation degrees-of-freedom.

Target: the per-edge Fisher-z SE / p-value (src/pegasus/ldo/edges.py::_uncertainty and
src/pegasus/ldo/multiplicity.py::fisher_z_pvalue) uses ``1/sqrt(n_eff - 3)``.

The quantity being tested is NOT a zero-order (marginal) correlation. In lags.py the
edge ``partial_correlation`` is read off the *precision matrix* over the lag-extended
feature set (``partial = -S / outer(d,d)``), so each off-diagonal is a partial correlation
conditioning on the other ``q - 2`` features (``q`` = number of kept features). The
textbook Fisher-z SE for a partial correlation controlling for ``k`` covariates is
``1/sqrt(n - k - 3)``; here ``k = q - 2``, giving the correct dof ``n_eff - q - 1``.

Consequence of using ``n_eff - 3`` instead:
  1. Every SE is too small → p-values / BH q-values are ANTICONSERVATIVE (too optimistic).
  2. When ``n_eff <= q`` the partial correlation is NOT IDENTIFIABLE (dof < 1), yet the
     code returns a finite, tiny SE and a confident tiny p-value — false precision.

These tests are adversarial: they hand the LIVE readout a regime where the conditioning
set is large relative to the effective sample size. They FAIL while the ``-3`` code stands
and would PASS only if the SE subtracted the conditioning-set size.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pegasus.ldo.certify import LDOCertificationPolicy, certify_links
from pegasus.ldo.edges import to_link_records
from pegasus.ldo.lags import LaggedFit, fit_lagged_links
from pegasus.ldo.lowrank import SparseLowRankFit
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.multiplicity import fisher_z_pvalue, _phi


def _p_correct_partial(r: float, n_eff: float, k_cond: int) -> float | None:
    """Textbook Fisher-z two-sided p for a partial corr controlling for ``k_cond`` covariates.
    dof = n_eff - k_cond - 3. Returns None when the partial corr is not identifiable (dof < 1)."""
    dof = n_eff - k_cond - 3.0
    if dof < 1.0:
        return None
    z = math.atanh(max(min(r, 0.999), -0.999))
    se = 1.0 / math.sqrt(dof)
    return 2.0 * (1.0 - _phi(abs(z) / se))


# --------------------------------------------------------------------------------------
# 1. Direct refutation of the p-value primitive: with a large conditioning set the LIVE
#    fisher_z_pvalue calls a moderate partial correlation "significant" when, controlling
#    for the co-conditioned variables, it is nowhere near significant.
# --------------------------------------------------------------------------------------
def test_fisher_z_pvalue_ignores_conditioning_set_and_is_anticonservative():
    r = 0.40
    n_eff = 42.0
    q = 40            # precision over q features → each edge conditions on q-2 = 38 others

    p_marginal = fisher_z_pvalue(r, n_eff)               # no conditioning → dof = n_eff - 3 = 39
    p_live = fisher_z_pvalue(r, n_eff, q - 2)             # conditioned → dof = n_eff - (q-2) - 3 = 1
    p_correct = _p_correct_partial(r, n_eff, q - 2)

    # The marginal (zero-order) p-value declares a "discovery"...
    assert p_marginal < 0.05, f"sanity: marginal p should look significant (got {p_marginal})"
    # ...but the honest partial-correlation p-value does not.
    assert p_correct is not None and p_correct > 0.20, (
        f"partial-corr p controlling for {q-2} covariates is not significant: {p_correct}"
    )
    # The FIX: fisher_z_pvalue given the conditioning set agrees with the honest partial verdict.
    assert (p_live < 0.05) == (p_correct < 0.05) and abs(p_live - p_correct) < 1e-9, (
        f"conditioned Fisher-z p ({p_live:.4f}) must match the honest partial-correlation "
        f"p ({p_correct:.4f}) at dof n_eff-q-1, not the marginal n_eff-3."
    )


# --------------------------------------------------------------------------------------
# 2. Non-identifiable regime: n_eff <= q means the precision-derived partial correlation
#    has < 1 residual degree of freedom, yet the LIVE certify_links path still stamps a
#    finite, small BH q-value on it (false precision). A correct SE would yield dof<1 and
#    refuse to report a finite p-value.
# --------------------------------------------------------------------------------------
def test_live_fdr_reports_confident_q_when_partial_corr_not_identifiable():
    # q = 30 features; n_eff deliberately below q so dof = n_eff - q - 1 < 0.
    q = 30
    n_eff = 25.0
    r = 0.5

    # Build records exactly as the live path produces them (contemporaneous edges carry
    # partial_correlation + n_eff), then run the real certifier / FDR annotation.
    from pegasus.ldo.records import LinkRecord

    recs = [
        LinkRecord(
            source_var=f"V{i}", target_var=f"V{i}'", edge_type="contemporaneous",
            weight=r, partial_correlation=r, stability=0.9, uncertainty=0.05,
            n_eff=n_eff, n_conditioning=q - 2, certification_status="selected",
        )
        for i in range(8)
    ]
    out = certify_links(recs, policy=LDOCertificationPolicy())

    q_reported = out[0].fdr_qvalue
    assert q_reported is not None, "live FDR annotated a q-value"

    # A partial correlation over q features needs n_eff > q + 1 to have >=1 residual dof.
    dof_correct = n_eff - (q - 2) - 3.0
    assert dof_correct < 1.0, "test setup: this regime is non-identifiable by construction"

    # THE FLAW: the live pipeline returns a confident (small) q-value for an edge whose
    # partial correlation is not even estimable at this effective sample size. An honest
    # SE would flag dof<1 (uninformative), not report q < 0.5.
    assert q_reported >= 0.5, (
        f"live BH stamped a confident q={q_reported:.4f} on a partial correlation with "
        f"{dof_correct:.0f} residual degrees of freedom (n_eff={n_eff} <= q={q}); the SE "
        f"used n_eff-3={n_eff-3:.0f} instead of n_eff-q-1={n_eff-q-1:.0f}."
    )


# --------------------------------------------------------------------------------------
# 3. End-to-end through the real fit + readout: a wide field (many variables => large
#    conditioning set) with a modest number of independent observations. Any certified
#    edge here has an uncertainty that overstates its true precision, because the SE never
#    subtracted the conditioning-set size. We assert the recorded uncertainty is at least
#    as large as the honest partial-correlation SE — which it is NOT under the -3 code.
# --------------------------------------------------------------------------------------
def test_end_to_end_recorded_uncertainty_understates_partial_corr_se():
    rng = np.random.default_rng(7)
    p = 16          # 16 variables => lag-extended feature count q >= p is the conditioning set
    S = 30          # 30 spatial cells
    T = 3           # minimal time so K can be >= 1
    # Independent standard-normal fields: no real structure, but with p=16 and few cells the
    # sample precision will still throw off spurious partial correlations above threshold.
    Z = rng.standard_normal((p, S, T))
    W = np.ones((p, S, T))
    field = GaussianField(
        variables=tuple(f"V{i}" for i in range(p)),
        space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
        time_ids=tuple(range(T)), Z=Z, W=W, resolution="year",
    )

    lagged = fit_lagged_links(field, K=1, lambda1=0.02, lambda2=0.05, edge_threshold=0.15,
                              spatial_whiten=False, min_coverage=5, min_overlap=5)
    recs = to_link_records(lagged, field=field)

    # q = number of kept features the precision conditioned on.
    q = len(lagged.fit.S)
    k_cond = max(q - 2, 0)

    offenders = []
    for r in recs:
        if r.partial_correlation is None or r.n_eff is None or r.uncertainty is None:
            continue
        n_eff = float(r.n_eff)
        se_live = r.uncertainty  # §LDO-CERT-UNITS-02: on the partial-corr (weight) scale, (1−ρ²)/√dof
        dof_correct = n_eff - k_cond - 3.0
        rp = min(abs(float(r.partial_correlation)), 0.999)
        # Honest partial SE at the correct dof, on the SAME (weight) scale as the recorded value:
        # the Fisher-z SE 1/√dof mapped by the delta method to (1−ρ²)/√dof (numerical_error=0 here).
        if dof_correct >= 1.0:
            se_correct = (1.0 - rp * rp) / math.sqrt(dof_correct)
            if se_live < se_correct - 1e-9:
                offenders.append((r.source_var, r.target_var, se_live, se_correct, dof_correct))
        else:
            # dof < 1: partial corr not identifiable — the code must refuse (uncertainty None, skipped
            # above), never a finite SE. If a finite SE reached here it is an offender.
            offenders.append((r.source_var, r.target_var, se_live, float("inf"), dof_correct))

    if not recs or all(r.partial_correlation is None for r in recs):
        pytest.skip("fit produced no testable edges in this seed; not the property under test")

    assert not offenders, (
        f"{len(offenders)} edge(s) carry an uncertainty that understates the true "
        f"partial-correlation SE (conditioning on {k_cond} covariates, q={q} features). "
        f"The recorded SE uses n_eff-3, ignoring the conditioning set. Example: {offenders[0]}"
    )
