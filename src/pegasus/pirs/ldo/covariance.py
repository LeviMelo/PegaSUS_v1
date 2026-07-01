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
    """
    F, n = samples.shape
    finite = np.isfinite(samples)
    coverage = finite.sum(axis=1)
    kept = [i for i in range(F) if coverage[i] >= min_coverage and np.nanstd(samples[i]) > 1e-9]
    q = len(kept)
    C = np.eye(q)
    for a in range(q):
        ia = kept[a]
        for b in range(a + 1, q):
            ib = kept[b]
            mask = finite[ia] & finite[ib]
            k = int(mask.sum())
            if k < min_overlap:
                continue
            xa = samples[ia, mask]
            xb = samples[ib, mask]
            sa, sb = xa.std(), xb.std()
            if sa > 1e-9 and sb > 1e-9:
                C[a, b] = C[b, a] = float(np.corrcoef(xa, xb)[0, 1])
    return PairwiseCovariance(
        correlation=_nearest_correlation(C),
        kept=tuple(kept),
        min_overlap=min_overlap,
        coverage=coverage[kept],
    )


__all__ = ["PairwiseCovariance", "pairwise_correlation"]
