"""LDO spatial GMRF whitening — §III.4(3/5) / §V.2 wired into the live estimator.

The variable precision Ω_var must be estimated *net of space*: without whitening,
spatial autocorrelation inflates apparent variable dependence (MSD-III §III.4). The
whitening ``Σ_space^{-1/2} = (κI + L_W)^{1/2}`` existed in precision.py but had zero
runtime callers and ``kappa`` was an inert threaded knob. These tests pin (1) the
mechanism — whitening a spatially-autocorrelated field removes its autocorrelation —
and (2) that it is now wired into ``fit_lagged_links`` so ``κ``/``L_W`` actually act.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.lags import fit_lagged_links
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.precision import build_spatial_precision, _matrix_sqrt_psd, _whiten_spatial

# a real connected municipality (cod6) subgraph — so κI + L_W has genuine Laplacian
# structure (these codes are present in structural_cod6_adjacency).
_REAL_CODES = (
    "110001", "110014", "110028", "110029", "110032", "110037", "110050", "110149",
    "110025", "110034", "110090", "110004", "110018", "110148", "110145", "110010",
    "110100", "110150", "110146", "110008", "110012", "110120", "110170", "110130",
)


def _moran(field: np.ndarray, adjacency: np.ndarray) -> float:
    S = field.shape[0]
    w = adjacency.sum()
    vals = []
    for t in range(field.shape[1]):
        x = field[:, t] - field[:, t].mean()
        den = float((x * x).sum())
        num = float((adjacency * np.outer(x, x)).sum())
        vals.append((S / w) * (num / den) if den > 0 else 0.0)
    return float(np.mean(vals))


def test_whitening_removes_spatial_autocorrelation():
    space_ids = _REAL_CODES
    S, T, kappa = len(space_ids), 200, 0.3
    Q = build_spatial_precision(space_ids, kappa=kappa)
    adjacency = (np.abs(Q - np.diag(np.diag(Q))) > 0).astype(float)
    assert adjacency.sum() > 0, "subgraph must have real adjacency edges"
    Q_half = _matrix_sqrt_psd(Q)
    Q_inv_half = _matrix_sqrt_psd(np.linalg.inv(Q))
    rng = np.random.default_rng(0)
    # GMRF sample: spatially smooth field, x = (κI+L_W)^{-1/2} z
    X = Q_inv_half @ rng.standard_normal((S, T))
    whitened = _whiten_spatial(X[None, :, :], Q_half)[0]

    moran_raw = _moran(X, adjacency)
    moran_white = _moran(whitened, adjacency)
    assert moran_raw > 0.1, f"GMRF field should be spatially autocorrelated, got {moran_raw:.3f}"
    assert abs(moran_white) < 0.08, f"whitening should remove autocorrelation, got {moran_white:.3f}"
    assert moran_white < moran_raw - 0.1, "whitening must substantially reduce Moran's I"


def test_spatial_whiten_is_wired_into_fit_lagged_links():
    """With real adjacency and a spatially-autocorrelated field, the whitened fit
    differs from the unwhitened one — proving κ/L_W now act (previously inert)."""
    space_ids = _REAL_CODES
    S, T, kappa = len(space_ids), 60, 0.3
    Q = build_spatial_precision(space_ids, kappa=kappa)
    Q_inv_half = _matrix_sqrt_psd(np.linalg.inv(Q))
    rng = np.random.default_rng(1)
    # A and B share a smooth spatial trend (a spatial confounder) + independent noise —
    # a real A-B correlation whose spatial component whitening transforms away.
    trend = Q_inv_half @ rng.standard_normal((S, T))
    Z = np.stack([
        trend + 0.4 * rng.standard_normal((S, T)),
        trend + 0.4 * rng.standard_normal((S, T)),
    ])
    field = GaussianField(
        variables=("A", "B"), space_ids=space_ids, time_ids=tuple(range(T)),
        Z=Z, W=np.ones_like(Z), resolution="year",
    )
    fit_raw = fit_lagged_links(field, K=1, kappa=kappa, spatial_whiten=False, min_coverage=10, min_overlap=8)
    fit_white = fit_lagged_links(field, K=1, kappa=kappa, spatial_whiten=True, min_coverage=10, min_overlap=8)
    # the estimated precision must change once spatial autocorrelation is whitened out
    delta = float(np.linalg.norm(fit_raw.fit.precision - fit_white.fit.precision))
    assert delta > 1e-6, f"spatial whitening had no effect on the precision (Δ={delta:.2e}) — knob still inert"
    # and the A-B contemporaneous partial correlation is attenuated (space explained some of it)
    raw_ab = abs(dict(((a, b), v) for a, b, v in fit_raw.contemporaneous).get(("A", "B"), 0.0))
    white_ab = abs(dict(((a, b), v) for a, b, v in fit_white.contemporaneous).get(("A", "B"), 0.0))
    assert white_ab <= raw_ab + 1e-9, f"whitening should not amplify the spatial-confounded edge (raw={raw_ab:.3f}, white={white_ab:.3f})"


def test_sparse_metric_whitening_equals_dense_and_scales():
    """The matrix-free sparse-metric whitened correlation (§V.2/§V.4) must equal the
    dense Σ_space^{-1/2}-whitening to machine precision, and must NOT densify S×S — so
    it scales to national S≈5570 where the dense O(S³) eigh is infeasible."""
    import time
    import numpy as np
    import scipy.sparse as sp
    from pegasus.ldo.covariance import pairwise_correlation, whitened_lagged_correlation
    from pegasus.ldo.lags import _build_lagged_feature_matrix
    from pegasus.ldo.precision import (
        build_spatial_precision, build_spatial_precision_sparse, _matrix_sqrt_psd, _whiten_spatial,
    )
    sids = _REAL_CODES[:15]
    S = len(sids)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((5, S, 30))
        dense = pairwise_correlation(
            _build_lagged_feature_matrix(_whiten_spatial(Z, _matrix_sqrt_psd(build_spatial_precision(sids, kappa=0.5))), 1),
            min_coverage=5, min_overlap=5,
        )
        metric = whitened_lagged_correlation(Z, build_spatial_precision_sparse(sids, kappa=0.5), 1, min_coverage=5)
        assert dense.kept == metric.kept
        assert np.allclose(dense.correlation, metric.correlation, atol=1e-9), \
            f"seed {seed}: metric whitening differs from dense by {np.abs(dense.correlation - metric.correlation).max():.2e}"
    # national scale: 5570 munis must finish quickly and never form a dense S×S matrix
    Qbig = sp.eye(5570, format="csr") * 0.5 + sp.random(5570, 5570, density=6 / 5570, format="csr", random_state=0)
    Zbig = np.random.default_rng(0).standard_normal((6, 5570, 25))
    t = time.time()
    out = whitened_lagged_correlation(Zbig, Qbig, 1, min_coverage=5)
    assert time.time() - t < 10.0, "national-scale metric whitening must be fast (matrix-free)"
    assert out.correlation.shape[0] <= 12
