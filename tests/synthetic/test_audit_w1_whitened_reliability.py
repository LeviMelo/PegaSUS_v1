"""ADVERSARIAL AUDIT — W1: reliability tensor W wired into the LDO covariance (commit 59e8226).

The fix claims: "the per-cell reliability tensor W now weights the LDO covariance so
reconstructed/low-reliability cells cannot manufacture a certified edge."

The existing proof test (`test_unreliable_cells_cannot_manufacture_an_edge`) exercises ONLY
`pairwise_correlation` — the `spatial_whiten=False` branch. But the LIVE national run uses
`spatial_whiten=True` (the orchestrator default), which routes through
`whitened_lagged_correlation`. This audit attacks the LIVE path.

The whitened path applies reliability as a *feature scaling* (`Ft *= W`) but mean-centers with
`H = R @ h` divided by the CONSTANT `n_cells = S * T_eff` (the RAW cell count) — NOT the sum of
weights. When W is NON-UNIFORM, the subtracted "mean" is `(weighted sum)/(raw count)`, a biased,
under-scaled mean. For data with a nonzero mean this leaves a large uncentered positive component
in every feature, which the cross-moment then reads as a spurious positive covariance. The
consequence is the OPPOSITE of the fix's guarantee: two INDEPENDENT variables acquire a spurious,
high-magnitude, would-be-CERTIFIED edge purely because the reliability tensor is non-uniform.

These tests assert the fix's stated contract on the whitened path. They FAIL if the flaw is real.
"""

from __future__ import annotations

import numpy as np
import pytest

from pegasus.ldo.covariance import pairwise_correlation, whitened_lagged_correlation
from pegasus.ldo.precision import build_spatial_precision_sparse


def _corr01(pw) -> float | None:
    if 0 in pw.kept and 1 in pw.kept:
        return float(pw.correlation[pw.kept.index(0), pw.kept.index(1)])
    return None


def test_whitened_nonuniform_W_does_not_manufacture_edge_between_independents():
    """Two INDEPENDENT variables must not acquire a spurious edge when down-weighted.

    A valid reliability down-weight is (at worst) scale-invariant on the correlation of two
    independent variables — it cannot turn r≈0 into r≈1. Here A and B are independent with a
    nonzero mean; a non-uniform W (high on a few munis, tiny elsewhere — exactly the real
    provenance pattern: `observed`=1.0 on a subset, `broadcast`≈0.25-0.5 elsewhere) is applied.

    Ground truth: both the unweighted whitened estimate and an honest hard-drop (properly
    centered) estimate put the correlation near 0. If the weighted whitened estimator instead
    reports a large-magnitude edge, the reliability weighting is MANUFACTURING a certifiable
    discovery out of noise — refuting the fix's central claim.
    """
    rng = np.random.default_rng(7)
    p, S, T = 2, 60, 5
    mean = 2.0  # a modest nonzero mean (count-with-exposure margins are far from zero-mean)
    A = rng.standard_normal((S, T)) + mean
    B = rng.standard_normal((S, T)) + mean  # independent of A
    Z = np.stack([A, B])
    space_ids = tuple(f"99{i:05d}"[:7] for i in range(S))
    Q = build_spatial_precision_sparse(space_ids, kappa=1.0)

    # Real provenance pattern: reliable on a subset of munis, low-reliability broadcast elsewhere.
    W = np.full((p, S, T), 0.05)
    W[:, :10, :] = 1.0

    unw = whitened_lagged_correlation(Z, Q, K=1, min_coverage=3)
    wtd = whitened_lagged_correlation(Z, Q, K=1, min_coverage=3, weights=W)

    r_unw = _corr01(unw)
    r_wtd = _corr01(wtd)
    assert r_unw is not None and r_wtd is not None
    # Sanity: the honest, properly-centered reference agrees the truth is ≈0.
    assert abs(r_unw) < 0.15, f"unweighted whitened corr of independents should be ~0; got {r_unw:.3f}"

    # The fix's contract: down-weighting must not manufacture a certifiable edge (threshold 0.05,
    # certification well below the |r| we are guarding against). A valid weighted correlation of
    # two independent variables stays near 0 regardless of the weight profile.
    assert abs(r_wtd) < 0.2, (
        f"reliability weighting manufactured a spurious edge between INDEPENDENT variables: "
        f"weighted whitened corr = {r_wtd:.3f} (unweighted {r_unw:.3f}). The whitened path "
        f"mean-centers with the raw cell count, not the weight sum, so a non-uniform W leaks a "
        f"large uncentered mean into the cross-moment — the OPPOSITE of the fix's guarantee."
    )


def test_whitened_weighted_matches_honest_harddrop_reference():
    """The weighted whitened estimate should track the honest 'drop the unreliable cells' estimate.

    If reliability weighting means anything, driving W→0 on the unreliable cells should approach
    the estimate you'd get by actually removing them. We compare the whitened weighted correlation
    against a hard-drop pairwise (properly centered) reference on the SAME retained cells.
    """
    rng = np.random.default_rng(11)
    p, S, T = 2, 60, 5
    mean = 2.0
    A = rng.standard_normal((S, T)) + mean
    B = rng.standard_normal((S, T)) + mean
    Z = np.stack([A, B])
    space_ids = tuple(f"99{i:05d}"[:7] for i in range(S))
    Q = build_spatial_precision_sparse(space_ids, kappa=1.0)

    W = np.full((p, S, T), 1e-6)
    W[:, :10, :] = 1.0  # essentially only the first 10 munis carry weight
    wtd = whitened_lagged_correlation(Z, Q, K=1, min_coverage=3, weights=W)
    r_wtd = _corr01(wtd)

    # Honest reference: hard-drop the near-zero-weight cells and estimate on the retained subset.
    Zdrop = Z.copy()
    Zdrop[:, 10:, :] = np.nan
    feat = np.stack([Zdrop[0].reshape(-1), Zdrop[1].reshape(-1)])
    honest = pairwise_correlation(feat, min_coverage=3, min_overlap=3)
    r_honest = float(honest.correlation[0, 1]) if honest.kept == (0, 1) else None
    assert r_honest is not None

    assert abs(r_wtd - r_honest) < 0.2, (
        f"weighted whitened corr {r_wtd:.3f} diverges from the honest hard-drop reference "
        f"{r_honest:.3f}: the feature-scaling reliability form does not implement a valid "
        f"down-weight in the whitened path."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
