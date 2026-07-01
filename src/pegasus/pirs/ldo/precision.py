"""LDO Layer 1 — structured precision at K=0 (MSD-II §II.6.1-2, MII-LDO-02).

Model the latent Gaussian field with a separable precision
``Prec(Z) ≈ Ω_var ⊗ Σ_space^{-1} ⊗ Σ_time^{-1}``. At ``K=0`` (contemporaneous
only) the links are the off-diagonal support of the variable-dependency operator
``Ω_var``. The spatial prior is the GMRF precision ``Σ_space^{-1} = κI + L_W`` from
the SpatialWeightGraph (§II.4).

Estimation strategy (the separable backbone): GMRF-*whiten* ``Z`` across space so
cells become approximately iid draws of the ``p``-variable vector, then recover a
sparse ``Ω_var`` by graphical lasso. Whitening by ``Σ_space^{-1/2}`` removes the
spatial autocorrelation that would otherwise inflate apparent variable
dependence — so a link in ``Ω_var`` is dependence *net of* space.

(The K=0 estimator uses sklearn's graphical_lasso; the GPU proximal + sparse–
low-rank split arrives with Layer 2, MII-LDO-03. The interface here is what is
locked; the numerical backend is a Mutable Solver.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import graphical_lasso

from pegasus.geo.spatial_graph import structural_cod6_adjacency
from pegasus.pirs.ldo.margins import GaussianField


@dataclass
class PrecisionFit:
    variables: tuple[str, ...]
    precision: np.ndarray            # Ω_var (p x p)
    partial_correlation: np.ndarray  # -Ω_ij / sqrt(Ω_ii Ω_jj)
    n_samples: int
    edges: list[tuple[int, int, float]] = field(default_factory=list)  # (i, j, partial_corr)

    def edge_names(self) -> list[tuple[str, str, float]]:
        return [(self.variables[i], self.variables[j], r) for i, j, r in self.edges]


def build_spatial_precision(space_ids: tuple[str, ...], *, kappa: float = 1.0) -> np.ndarray:
    """GMRF spatial precision ``κ I + L_W`` (cod6) over the field's municipalities."""
    adjacency = structural_cod6_adjacency()
    S = len(space_ids)
    idx = {s: i for i, s in enumerate(space_ids)}
    L = np.zeros((S, S), dtype=np.float64)
    for s in space_ids:
        i = idx[s]
        neighbours = [nb for nb in adjacency.get(s, ()) if nb in idx]
        L[i, i] = float(len(neighbours))
        for nb in neighbours:
            L[i, idx[nb]] -= 1.0
    return kappa * np.eye(S) + L


def _matrix_sqrt_psd(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _whiten_spatial(Z: np.ndarray, Q_half: np.ndarray) -> np.ndarray:
    """Whiten each variable/time slice across space by ``Σ_space^{-1/2}``.

    Missing cells are imputed with the Gaussian margin mean (0) before whitening.
    """
    p, S, T = Z.shape
    Zc = np.where(np.isfinite(Z), Z, 0.0)
    out = np.empty_like(Zc)
    for j in range(p):
        # (S x T) → whitened (S x T) = Q_half @ slice
        out[j] = Q_half @ Zc[j]
    return out


def fit_contemporaneous_precision(
    field: GaussianField,
    *,
    kappa: float = 1.0,
    alpha: float = 0.05,
    edge_threshold: float = 0.05,
    spatial_whiten: bool = True,
) -> PrecisionFit:
    """Estimate the sparse contemporaneous variable precision ``Ω_var`` (K=0)."""
    p, S, T = field.shape
    Z = field.Z
    if spatial_whiten and S > 1:
        Q = build_spatial_precision(field.space_ids, kappa=kappa)
        Q_half = _matrix_sqrt_psd(Q)
        Zw = _whiten_spatial(Z, Q_half)
    else:
        Zw = np.where(np.isfinite(Z), Z, 0.0)

    # Samples = (s, t) cells; features = p variables.
    D = Zw.reshape(p, S * T).T  # (n_samples, p)
    # Standardize features so graphical_lasso's single alpha is comparable.
    mu = D.mean(axis=0)
    sd = D.std(axis=0)
    sd[sd == 0] = 1.0
    Dz = (D - mu) / sd
    n = Dz.shape[0]

    emp_cov = np.cov(Dz, rowvar=False)
    emp_cov += 1e-4 * np.eye(p)  # ridge for numerical PD
    try:
        _, precision = graphical_lasso(emp_cov, alpha=alpha, max_iter=200)
    except Exception:
        precision = np.linalg.pinv(emp_cov)

    d = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)

    edges: list[tuple[int, int, float]] = []
    for i in range(p):
        for j in range(i + 1, p):
            r = float(partial[i, j])
            if abs(r) >= edge_threshold:
                edges.append((i, j, r))
    edges.sort(key=lambda e: abs(e[2]), reverse=True)
    return PrecisionFit(
        variables=field.variables,
        precision=precision,
        partial_correlation=partial,
        n_samples=n,
        edges=edges,
    )


__all__ = ["PrecisionFit", "fit_contemporaneous_precision", "build_spatial_precision"]
