"""Missing-aware covariance for the LDO (MII-LDO robustness fix).

Real epidemiological panels are sparse: no cell has every variable observed
(complete-case is empty), and imputing missing→0 fabricates a spurious shared
factor that collapses every relationship into ``latent_shared``. The correct
estimator is the **pairwise-complete** covariance — each entry from the cells where
*both* variables are observed — projected to the nearest correlation matrix so the
sparse+low-rank precision estimator receives a valid PSD input.

Variables whose coverage or pairwise overlap is too small are dropped (degenerate
columns), and the surviving index is returned so edges map back to variable names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairwiseCovariance:
    correlation: np.ndarray          # (q x q) PSD correlation over kept variables
    kept: tuple[int, ...]            # indices into the original variable axis
    min_overlap: int
    coverage: np.ndarray             # per-kept-variable observed-sample count


def _nearest_correlation(C: np.ndarray, *, floor: float = 1e-3) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD correlation matrix."""
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, floor, None)
    psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    return psd / np.outer(d, d)


def pairwise_correlation(
    samples: np.ndarray,
    *,
    min_coverage: int = 30,
    min_overlap: int = 20,
) -> PairwiseCovariance:
    """Pairwise-complete correlation of a (features × samples) matrix with NaNs.

    ``samples[i]`` is variable ``i`` over all cells (NaN where unobserved).

    Fully vectorized: the per-pair complete-case moments are assembled as a handful
    of BLAS matrix products over the mask ``M`` and the NaN-zeroed data ``X0`` rather
    than a Python double-loop over the ``O(q^2)`` feature pairs (which dominated the
    LDO's per-fit cost, run once per stability subsample). For a pair ``(a,b)`` with
    overlap mask, ``mean_a = (X0 @ M^T)/n``, ``var_a = (X0^2 @ M^T)/n - mean_a^2``,
    ``cov = (X0 @ X0^T)/n - mean_a mean_b`` — the 1/n (or 1/(n-1)) cancels in the
    correlation ratio, so this is numerically identical to per-pair ``np.corrcoef``.
    """
    F, n = samples.shape
    finite = np.isfinite(samples)
    coverage = finite.sum(axis=1)
    # nanstd only over covered rows (avoid all-NaN warnings); std==0 columns are dropped.
    stds = np.zeros(F, dtype=np.float64)
    covered = coverage > 0
    if covered.any():
        stds[covered] = np.nanstd(np.where(finite[covered], samples[covered], np.nan), axis=1)
    kept = [i for i in range(F) if coverage[i] >= min_coverage and stds[i] > 1e-9]
    q = len(kept)
    if q == 0:
        return PairwiseCovariance(correlation=np.zeros((0, 0)), kept=(), min_overlap=min_overlap, coverage=coverage[kept])

    X = samples[kept]                              # (q, n), may contain NaN
    M = np.isfinite(X).astype(np.float64)          # (q, n) overlap mask
    X0 = np.where(M > 0.0, X, 0.0)                  # (q, n) NaN -> 0

    n_ab = M @ M.T                                 # (q, q) pairwise overlap counts
    Sx = X0 @ M.T                                  # [a,b] = sum_t x_a * M_b  (sum of a over overlap)
    Sxx = (X0 * X0) @ M.T                           # [a,b] = sum_t x_a^2 * M_b
    Cxy = X0 @ X0.T                                # [a,b] = sum_t x_a * x_b  (overlap only; x=0 off-mask)

    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(n_ab > 0, 1.0 / n_ab, 0.0)
        mean_a = Sx * inv                          # mean of a over overlap-with-b
        mean_b = Sx.T * inv                        # mean of b over overlap-with-a
        var_a = Sxx * inv - mean_a * mean_a
        var_b = Sxx.T * inv - mean_b * mean_b
        cov = Cxy * inv - mean_a * mean_b
        denom = np.sqrt(np.clip(var_a, 0.0, None) * np.clip(var_b, 0.0, None))
        corr = np.where(denom > 1e-18, cov / denom, 0.0)

    # Enforce the guards the loop applied: enough overlap, finite, degenerate-variance -> 0.
    valid = (n_ab >= min_overlap) & (var_a > 1e-18) & (var_b > 1e-18) & np.isfinite(corr)
    C = np.where(valid, corr, 0.0)
    np.fill_diagonal(C, 1.0)
    C = 0.5 * (C + C.T)

    return PairwiseCovariance(
        correlation=_nearest_correlation(C),
        kept=tuple(kept),
        min_overlap=min_overlap,
        coverage=coverage[kept],
    )


__all__ = ["PairwiseCovariance", "pairwise_correlation"]
