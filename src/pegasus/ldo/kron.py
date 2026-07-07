"""Kronecker-factored joint precision operator (MSD-III §V.2, §III.4(3)).

The LDO's joint latent-field precision is *separable* across its three axes:

    Ω_joint  =  Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹

with three small factors — the ``p×p`` variable precision (LVGLASSO ``Ω_var``), the
sparse ``S×S`` spatial GMRF precision ``Σ_space⁻¹ = κI + L_W``
(:func:`pegasus.ldo.precision.build_spatial_precision_sparse`), and a ``τ×τ`` temporal
precision ``Σ_time⁻¹`` (AR/temporal-smoothness over the ``τ = K+1`` lag positions). The
dense joint object is ``(p·S·T)²`` — at national scale ``(130·5570·25)² ≈ 3·10¹⁴`` — and
must **never** be materialized. Every operation factors axis-by-axis instead:

* **matvec** — the 3-factor vec-trick: reshape the vector to a ``(p,S,τ)`` tensor and
  contract one mode at a time (``O(p²Sτ + pS²τ_nnz + pSτ²)``), never forming the joint.
* **log-det** — ``logdet(A⊗B⊗C) = Sτ·logdet(A) + pτ·logdet(B) + pS·logdet(C)`` (each on a
  small factor; the sparse ``B`` via a sparse Cholesky, or matrix-free SLQ at scale).
* **solve** — ``(A⊗B⊗C)⁻¹ = A⁻¹ ⊗ B⁻¹ ⊗ C⁻¹``, applied mode-wise (dense solves on the
  var/time factors, a sparse solve on the space factor). Exact, no iteration — the
  ``(pST)² → p²+S²+τ²`` collapse §V.2 prescribes. A matrix-free CG is also provided for
  the non-separable (Kronecker + correction) case.

This is the structural-decomposition axis of §III.4(3); §V.6 (exact-certifies-approximate)
governs its use where separability is an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_temporal_precision_ar1(n: int, phi: float = 0.5) -> np.ndarray:
    """The exact ``n×n`` AR(1) precision ``Σ_time⁻¹`` over ``n`` lag positions (tridiagonal).

    For a stationary AR(1) with lag-1 correlation ``φ``, the precision is
    ``(1/(1-φ²))·tridiag(-φ, [1, 1+φ², …, 1+φ², 1], -φ)`` — a temporal-smoothness prior
    tying adjacent lags, replacing the pure lag-stacking with a genuine time factor. At
    ``φ=0`` it is the identity (lags independent); ``n=1`` is ``[[1]]``.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not -1.0 < phi < 1.0:
        raise ValueError("AR(1) phi must be in (-1, 1)")
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)
    P = np.zeros((n, n), dtype=np.float64)
    diag = np.full(n, 1.0 + phi * phi)
    diag[0] = diag[-1] = 1.0
    np.fill_diagonal(P, diag)
    idx = np.arange(n - 1)
    P[idx, idx + 1] = -phi
    P[idx + 1, idx] = -phi
    return P / (1.0 - phi * phi)


@dataclass
class KroneckerPrecision:
    """A separable joint precision ``Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹``.

    ``omega_var`` (``p×p`` dense) and ``sigma_time_inv`` (``τ×τ`` dense) are small SPD
    factors; ``sigma_space_inv`` is the sparse (``scipy.sparse``) ``S×S`` GMRF precision —
    never densified. The axis order of the implied vectorization is ``(variable, space,
    time)`` with the *time* axis fastest (C-order flatten of a ``(p,S,τ)`` tensor).
    """

    omega_var: np.ndarray            # (p, p) dense SPD
    sigma_space_inv: object          # (S, S) scipy.sparse SPD
    sigma_time_inv: np.ndarray       # (τ, τ) dense SPD

    @property
    def p(self) -> int:
        return int(self.omega_var.shape[0])

    @property
    def S(self) -> int:
        return int(self.sigma_space_inv.shape[0])

    @property
    def tau(self) -> int:
        return int(self.sigma_time_inv.shape[0])

    @property
    def dim(self) -> int:
        return self.p * self.S * self.tau

    def _as_tensor(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64).reshape(self.p, self.S, self.tau)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """``(Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹) x`` by mode-wise contraction (never the joint)."""
        X = self._as_tensor(x)
        # mode 0 (variable): Y[i,s,t] = Σ_i' Ω[i,i'] X[i',s,t]
        Y = np.tensordot(self.omega_var, X, axes=([1], [0]))          # (p, S, τ)
        # mode 1 (space): apply the sparse S×S factor along axis 1 for all (variable,time)
        Ysp = Y.transpose(1, 0, 2).reshape(self.S, self.p * self.tau)  # (S, p·τ)
        Ysp = self.sigma_space_inv @ Ysp                              # sparse matvec
        Y = Ysp.reshape(self.S, self.p, self.tau).transpose(1, 0, 2)  # (p, S, τ)
        # mode 2 (time): Z[i,s,t] = Σ_t' Σt[t,t'] Y[i,s,t']
        Z = np.tensordot(Y, self.sigma_time_inv, axes=([2], [1]))     # (p, S, τ)
        return Z.reshape(self.dim)

    def logdet(self, *, space_logdet: float | None = None) -> float:
        """``logdet(Ω_var⊗Σ_space⁻¹⊗Σ_time⁻¹) = Sτ·logdet(Ω)+pτ·logdet(Σs⁻¹)+pS·logdet(Σt⁻¹)``.

        The two dense factors use ``slogdet``; the sparse space factor uses an exact sparse
        Cholesky by default (:func:`sparse_spd_logdet`), or a caller-supplied ``space_logdet``
        (e.g. a matrix-free stochastic-Lanczos estimate from :mod:`pegasus.ldo.stochastic`
        at national ``S``).
        """
        sign_v, ld_v = np.linalg.slogdet(self.omega_var)
        sign_t, ld_t = np.linalg.slogdet(self.sigma_time_inv)
        if sign_v <= 0 or sign_t <= 0:
            raise ValueError("Kronecker factors must be SPD (positive determinant)")
        ld_s = sparse_spd_logdet(self.sigma_space_inv) if space_logdet is None else float(space_logdet)
        return (self.S * self.tau) * ld_v + (self.p * self.tau) * ld_s + (self.p * self.S) * ld_t

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Exact ``(Ω_var⊗Σ_space⁻¹⊗Σ_time⁻¹)⁻¹ b`` mode-wise ``= (Ω⁻¹⊗Σs⁻¹⁻¹⊗Σt⁻¹⁻¹) b``.

        Applies each factor's inverse to its axis — dense ``np.linalg.solve`` on the var/time
        factors, a sparse ``spsolve`` on the space factor — so the joint system is solved in
        ``O(p³+τ³+ nnz(S))`` per right-hand side, never the ``O((pST)³)`` dense inverse.
        """
        from scipy.sparse.linalg import spsolve

        B = self._as_tensor(b)
        # invert mode 0 (variable)
        Y = np.tensordot(np.linalg.inv(self.omega_var), B, axes=([1], [0]))   # (p,S,τ)
        # invert mode 1 (space): solve Σs⁻¹ · Ysp = rhs  → Ysp = (Σs⁻¹)⁻¹ rhs
        rhs = Y.transpose(1, 0, 2).reshape(self.S, self.p * self.tau)
        Ysp = spsolve(self.sigma_space_inv.tocsc(), rhs)
        Ysp = np.asarray(Ysp).reshape(self.S, self.p, self.tau).transpose(1, 0, 2)
        # invert mode 2 (time)
        Z = np.tensordot(Ysp, np.linalg.inv(self.sigma_time_inv), axes=([2], [1]))
        return Z.reshape(self.dim)

    def to_dense(self) -> np.ndarray:
        """Materialize the joint operator (TEST/small-scale only — ``(pSτ)²`` memory)."""
        import scipy.sparse as sp

        B = self.sigma_space_inv.toarray() if sp.issparse(self.sigma_space_inv) else np.asarray(self.sigma_space_inv)
        return np.kron(np.kron(self.omega_var, B), self.sigma_time_inv)


def kronecker_from_ldo(
    omega_var: np.ndarray, space_ids: tuple[str, ...], *,
    kappa: float = 1.0, phi: float = 0.5, tau: int = 1,
) -> KroneckerPrecision:
    """Assemble the separable joint precision from an LDO fit's factors (§V.2).

    ``omega_var`` is the fitted ``p×p`` lag-0 variable precision; the spatial factor is the
    same sparse GMRF ``κI+L_W`` used to whiten (:func:`build_spatial_precision_sparse`); the
    temporal factor is an AR(1) precision over ``tau`` lag positions. This is the separable
    joint model whose log-det/solve feed §V.5/§V.6 — the whole ``(pSτ)²`` object stays implicit.
    """
    from pegasus.ldo.precision import build_spatial_precision_sparse

    space = build_spatial_precision_sparse(space_ids, kappa=kappa)
    time = build_temporal_precision_ar1(max(1, int(tau)), phi=phi)
    return KroneckerPrecision(omega_var=np.asarray(omega_var, dtype=np.float64),
                              sigma_space_inv=space, sigma_time_inv=time)


def joint_logdet(op: KroneckerPrecision, *, exact_space_max: int = 20000,
                 n_probes: int = 16, seed: int = 0) -> tuple[float, str]:
    """Joint ``logdet`` of the factored operator, exact for moderate ``S`` else matrix-free.

    Returns ``(logdet, method)`` where ``method`` is ``"sparse_cholesky"`` (exact, ``S ≤
    exact_space_max``) or ``"stochastic_lanczos"`` (§V.3 SLQ estimate on the sparse space
    matvec — the space factor never densifies). The var/time factors are always exact.
    """
    if op.S <= exact_space_max:
        return op.logdet(), "sparse_cholesky"
    from pegasus.ldo.stochastic import stochastic_logdet

    space = op.sigma_space_inv
    ld_s = stochastic_logdet(lambda x: space @ x, op.S, n_probes=n_probes, seed=seed)
    return op.logdet(space_logdet=ld_s), "stochastic_lanczos"


def sparse_spd_logdet(A) -> float:
    """Exact ``logdet`` of a sparse SPD matrix via a sparse LU (``|det|``, SPD ⇒ det>0).

    ``logdet = Σ log|diag(L)| + Σ log|diag(U)|``. Row/column permutations only flip the sign,
    which is ``+`` for an SPD matrix, so the magnitude is exact. Prefer a matrix-free SLQ
    estimate (:mod:`pegasus.ldo.stochastic`) when ``S`` is too large to factor.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    A = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
    lu = splu(A)
    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()
    return float(np.sum(np.log(np.abs(diagL))) + np.sum(np.log(np.abs(diagU))))


def kron_cg_solve(op: KroneckerPrecision, b: np.ndarray, *, correction=None,
                  tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """Matrix-free CG for ``(Ω_joint + correction) x = b`` using :meth:`KroneckerPrecision.matvec`.

    For the *pure* separable operator prefer the exact :meth:`KroneckerPrecision.solve`. CG is
    for the non-separable case (a Kronecker operator plus a low-rank/diagonal correction
    ``correction(x) → A_corr·x``), preconditioned by the exact Kronecker inverse — the
    Kronecker-factor preconditioner §V.4 names. Returns the solution vector.
    """
    b = np.asarray(b, dtype=np.float64).reshape(op.dim)

    def A(v):
        y = op.matvec(v)
        return y + correction(v) if correction is not None else y

    # Preconditioner M⁻¹ = exact Kronecker inverse (drops the correction — the standard
    # separable preconditioner for a Kronecker-plus-correction system).
    def Minv(v):
        return op.solve(v)

    x = np.zeros(op.dim)
    r = b - A(x)
    z = Minv(r)
    d = z.copy()
    rz = float(r @ z)
    bnorm = max(float(np.linalg.norm(b)), 1e-30)
    for _ in range(max_iter):
        if np.linalg.norm(r) / bnorm < tol:
            break
        Ad = A(d)
        alpha = rz / max(float(d @ Ad), 1e-30)
        x = x + alpha * d
        r = r - alpha * Ad
        z = Minv(r)
        rz_new = float(r @ z)
        beta = rz_new / max(rz, 1e-30)
        d = z + beta * d
        rz = rz_new
    return x


__all__ = [
    "KroneckerPrecision",
    "build_temporal_precision_ar1",
    "kronecker_from_ldo",
    "joint_logdet",
    "sparse_spd_logdet",
    "kron_cg_solve",
]
