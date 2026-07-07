"""O12 / §3.12.3 / §II.3 — the Q-tensor state (n_eff / fragility / provenance) weights the LDO W.

A field with a large design effect (low n_eff), a fragile denominator, or unofficial provenance
must be DOWN-WEIGHTED in the precision fit — a normalized/uncertain quantity is not modelled as if
exact. This pins that field_weights threads through run_ldo into the assembled reliability tensor.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import assemble_ldo_tensor
from pegasus.she.panel import CommonPanel
import polars as pl


def _panel():
    rows = []
    for s in range(6):
        for t in (2020, 2021, 2022, 2023):
            rows.append({"municipality_cod6": f"27000{s}", "year": t, "A": 1.0 * s + t, "B": 2.0 * s})
    values = pl.DataFrame(rows)
    manifest = pl.DataFrame({"field_id": [], "municipality_cod6": [], "year": [], "state": []},
                            schema={"field_id": pl.Utf8, "municipality_cod6": pl.Utf8, "year": pl.Int64, "state": pl.Utf8})
    index = values.select(["municipality_cod6", "year"]).unique()
    return CommonPanel(resolution="year", cell_keys=("municipality_cod6", "year"), index=index,
                       values=values, manifest=manifest, fields=("A", "B"))


def test_field_weights_scale_the_reliability_tensor():
    panel = _panel()
    base = assemble_ldo_tensor(panel)
    weighted = assemble_ldo_tensor(panel, field_weights={"A": 0.25})
    ia = base.variables.index("A")
    ib = base.variables.index("B")
    # A's weights are scaled by 0.25 (relative to base); B's are unchanged
    obs = np.isfinite(base.X[ia])
    assert np.allclose(weighted.W[ia][obs], 0.25 * base.W[ia][obs])
    assert np.allclose(weighted.W[ib], base.W[ib])


def test_run_ldo_reports_state_weighted_count():
    from pegasus.ldo.orchestrator import run_ldo
    from pegasus.ldo.margins import GaussianField
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((3, 40, 6))
    field = GaussianField(variables=("A", "B", "C"),
                          space_ids=tuple(f"27{i:05d}"[:7] for i in range(40)),
                          time_ids=tuple(range(6)), Z=Z, W=np.ones((3, 40, 6)), resolution="year")
    # GaussianField source has no raw panel to weight, but the param must thread without error
    run = run_ldo(field, K=1, n_subsamples=2, seed=0)
    assert "n_state_weighted" in run.diagnostics


# --- W1 / §II.6.1 ObservationReliability contract: the reliability tensor W must actually
# reach the covariance estimator and change the estimate. The tests above only pin the PLUMBING
# (W is scaled, the diagnostic exists); these pin the EFFECT (the fit consumes W). ---

def test_reliability_weighting_is_byte_identical_when_absent():
    """weights=None ⇒ W=binary-mask ⇒ every moment product collapses to the unweighted form.

    A supplied all-ones weight (or the finite-mask itself) must reproduce the unweighted
    estimate EXACTLY (not merely approximately) for BOTH estimators — the safety guarantee
    that turning the contract on is a strict generalization, so no existing run drifts unless
    a cell is genuinely sub-unity reliable.
    """
    from pegasus.ldo.covariance import pairwise_correlation, whitened_lagged_correlation
    from pegasus.ldo.precision import build_spatial_precision_sparse

    rng = np.random.default_rng(1)
    samples = rng.standard_normal((5, 200))
    samples[0, ::7] = np.nan  # some missingness so the mask is nontrivial
    base = pairwise_correlation(samples, min_coverage=10, min_overlap=10)
    ones = pairwise_correlation(samples, min_coverage=10, min_overlap=10,
                                weights=np.ones_like(samples))
    mask = pairwise_correlation(samples, min_coverage=10, min_overlap=10,
                                weights=np.isfinite(samples).astype(float))
    assert np.array_equal(base.correlation, ones.correlation)
    assert np.array_equal(base.correlation, mask.correlation)

    Z = rng.standard_normal((3, 12, 8))
    Q = build_spatial_precision_sparse(tuple(f"27{i:05d}"[:7] for i in range(12)), kappa=1.0)
    wb = whitened_lagged_correlation(Z, Q, K=2, min_coverage=5)
    ww = whitened_lagged_correlation(Z, Q, K=2, min_coverage=5, weights=np.ones_like(Z))
    assert np.array_equal(wb.correlation, ww.correlation)


def test_unreliable_cells_cannot_manufacture_an_edge():
    """The core epistemic guarantee (A1/M6/LDO-B01): a correlation carried ENTIRELY by
    low-reliability (reconstructed/broadcast) cells is suppressed once those cells are
    down-weighted — so imputed demography can't be certified as a discovery.

    Construct A,B independent on reliable cells but perfectly correlated on unreliable ones.
    Unweighted, the spurious signal survives; weighted by reliability, it collapses to the
    reliable-only (≈0) association.
    """
    from pegasus.ldo.covariance import pairwise_correlation

    rng = np.random.default_rng(0)
    n_rel, n_unrel = 300, 300
    A = np.concatenate([rng.standard_normal(n_rel), rng.standard_normal(n_unrel)])
    B = np.concatenate([rng.standard_normal(n_rel), A[n_rel:].copy()])  # spurious on unreliable half
    samples = np.vstack([A, B])
    w = np.concatenate([np.ones(n_rel), np.full(n_unrel, 0.02)])
    weights = np.vstack([w, w])

    r_unw = pairwise_correlation(samples, min_coverage=10, min_overlap=10).correlation[0, 1]
    r_w = pairwise_correlation(samples, min_coverage=10, min_overlap=10, weights=weights).correlation[0, 1]
    r_reliable_only = pairwise_correlation(samples[:, :n_rel], min_coverage=10, min_overlap=10).correlation[0, 1]

    assert abs(r_unw) > 0.3, f"spurious edge should be present unweighted; got {r_unw:.3f}"
    assert abs(r_w) < 0.1, f"reliability weighting should suppress the spurious edge; got {r_w:.3f}"
    assert abs(r_w - r_reliable_only) < 0.1, "weighted estimate should track the reliable-only association"


def test_fit_lagged_links_threads_field_reliability():
    """End-to-end: field.W reaches fit_lagged_links and changes the fit, and the switch is a
    clean no-op when reliability is trivial (W=ones) — so default-on never drifts an all-observed
    run, only a run with genuinely sub-unity cells.

    (The *suppression* semantics — an unreliable-only tie collapsing — are proven directly at the
    estimator in ``test_unreliable_cells_cannot_manufacture_an_edge``; here we only pin that W is
    actually consumed by the fit and that the ON switch is reversible.)"""
    from pegasus.ldo.lags import fit_lagged_links
    from pegasus.ldo.margins import GaussianField

    rng = np.random.default_rng(2)
    p, S, T = 3, 30, 7
    Z = rng.standard_normal((p, S, T))
    Z[1, S // 2:, :] = Z[0, S // 2:, :]          # a tie carried only by the second half of munis
    space_ids = tuple(f"27{i:05d}"[:7] for i in range(S))

    # Nontrivial reliability on exactly those cells → the fit must change vs unweighted.
    W = np.ones((p, S, T)); W[:, S // 2:, :] = 0.02
    field = GaussianField(variables=("A", "B", "C"), space_ids=space_ids,
                          time_ids=tuple(range(T)), Z=Z, W=W, resolution="year")
    weighted = fit_lagged_links(field, K=1, spatial_whiten=False, use_reliability_weights=True)
    unweighted = fit_lagged_links(field, K=1, spatial_whiten=False, use_reliability_weights=False)
    assert not np.allclose(weighted.fit.S, unweighted.fit.S), "field.W must reach the estimator"

    # Trivial reliability (W=ones) → the ON switch is a byte-identical no-op.
    field_ones = GaussianField(variables=("A", "B", "C"), space_ids=space_ids,
                               time_ids=tuple(range(T)), Z=Z, W=np.ones((p, S, T)), resolution="year")
    on = fit_lagged_links(field_ones, K=1, spatial_whiten=False, use_reliability_weights=True)
    off = fit_lagged_links(field_ones, K=1, spatial_whiten=False, use_reliability_weights=False)
    assert np.array_equal(on.fit.S, off.fit.S), "W=ones must make the contract a strict no-op"
