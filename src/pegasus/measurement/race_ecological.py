"""Ecological deconvolution of self-declared race rates from confused administrative race.

MSD §4.6 (RaceBridge, ecological reformalization). This is a *distinct* estimator from the
per-cell forward crosswalk in :mod:`pegasus.measurement.race`; both are legitimate and this
module does not touch the wired crosswalk path. It is the aggregate (ecological) model whose
capability was validated by planted-signal recovery before implementation
(scratchpad/racebridge_validation_probe.py -> tests/unit/test_race_ecological_deconvolution.py).

The problem it solves
---------------------
Administrative race/color (a doctor filling a death certificate, a clerk a hospital record) is a
*confused* observation of a person's self-declared race. Applying a fixed reclassification crosswalk
cell-by-cell provably **erases** a genuine racial rate signal when the confusion is non-trivial
(the naive ``Y.sum / N.sum`` estimator inverts a planted 1.50 rate-ratio to 0.91 in validation).
Individual reclassification is unidentifiable; the distortion is only recoverable in aggregate, by
pooling across cells whose self-declared composition varies (Goodman/King ecological inference).

Generative model
-----------------
Cells ``s = 1..S`` (e.g. municipality x year), self-declared races ``j = 1..J`` (KNOWN census
population ``N[s, j]``), administrative races ``k = 1..K`` (OBSERVED counts ``Y[s, k]``). Confusion
may vary across *strata* ``g = 1..G`` (a data system, an age band, a period) -- this is the
covariate-dependent confusion the user asked for, in its identifiable (piecewise) form::

    m[s, j]        = exp(a[s] + b[j]) * N[s, j]      # latent true-race event mass; b[0] := 0 (reference)
    mu^(g)[s, k]   = sum_j C^(g)[k|j] * m[s, j]      # confused into admin categories, per stratum
    Y^(g)[s, k]   ~ Poisson(mu^(g)[s, k])

``b[j]`` are the shared genuine race log-rate-ratios (the signal to recover WITHOUT erasing);
``a[s]`` are nuisance cell baselines; ``C^(g)`` (K x J, column-stochastic) is the per-stratum
emission matrix ``P(admin = k | self-declared = j)`` -- the SAME direction as the whitening the
deconvolution inverts, and the OPPOSITE direction of the row-stochastic reclassification prior
stored in the race-bridge registry (do not confuse the two; see ``emission_prior``).

Identifiability comes from (a) cross-cell variation in the self-declared composition ``N[s, .]`` and
(b) optionally multiple strata that share the rate process but confuse differently. A smooth
logit-space Gaussian prior ``0.5 * kappa * ||Z - Z0||^2`` (C = column-softmax(Z)) anchors C to a
literature prior; validation shows a *too-strong* prior (kappa ~ 200) re-erases the signal, so kappa
is a genuine modeling knob (trust in the prior), defaulted weak and never auto-forced to identity.

The optimiser is L-BFGS-B on (Z per stratum, a, free b) with the analytic gradient validated by
finite-difference gradcheck (max abs error ~1e-6).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # SciPy is a hard dependency of the inference stack; guard only for import-time clarity.
    from scipy.optimize import minimize
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "pegasus.measurement.race_ecological requires scipy.optimize.minimize"
    ) from exc


__all__ = [
    "EcologicalRaceProblem",
    "EcologicalRaceResult",
    "fit_ecological_race_deconvolution",
    "contextual_identifiability",
    "emission_from_reclassification",
    "reclassification_from_emission",
]

# Below this per-cell RMS composition-share variation (in the least-varying non-degenerate direction)
# the ecological system is too near-collinear across cells to separate the races, and the recovered
# rate ratios are prior-dominated; the result is emitted but flagged rather than silently trusted
# (CLAUDE.md IV/V: never silently degrade). Calibrated against planted-signal recovery: alpha=80
# (~0.010) still recovers a 1.5 ratio as 1.30; alpha=5000 (~0.0015) is prior-dominated.
IDENTIFIABILITY_FLOOR = 0.005
DEFAULT_PRIOR_STRENGTH = 2.0
DEFAULT_MAX_ITER = 2000


class EcologicalRaceError(ValueError):
    """Raised when an ecological race-deconvolution problem is malformed."""


def emission_from_reclassification(
    reclassification: np.ndarray,
    admin_marginal: np.ndarray,
) -> np.ndarray:
    """Convert a reclassification prior ``P(self-declared=j | admin=k)`` to the emission prior
    ``P(admin=k | self-declared=j)`` the ecological deconvolution consumes.

    The race-bridge registry stores a **row-stochastic** reclassification matrix ``R[k, j] =
    P(self-declared=j | admin=k)`` (each admin row sums to 1 over self-declared). The ecological
    model (MSD §4.6) instead needs the **column-stochastic** emission ``C[k|j] = P(admin=k |
    self-declared=j)`` (each self-declared column sums to 1 over admin) — the forward whitening it
    inverts. These are opposite conditionals, related by Bayes with the admin marginal ``p_admin[k] =
    P(admin=k)`` (directly observable by counting admin race in the data):

        C[k|j] = P(admin=k | self=j)
               = P(self=j | admin=k) P(admin=k) / P(self=j)
               = R[k, j] p_admin[k] / Σ_{k'} R[k', j] p_admin[k']

    The self marginal ``P(self=j)`` is exactly the normaliser ``Σ_{k'} R[k',j] p_admin[k']`` and so
    cancels — only ``R`` and the observed ``p_admin`` are required. This is the reconciliation the
    W-RACE-2 wiring needs; it is a pure transform (no live-path effect) so it can be validated in
    isolation before the estimator is wired into the denominator path.

    Args:
        reclassification: (K, J) row-stochastic ``R[k, j] = P(self=j | admin=k)``.
        admin_marginal: (K,) observed ``p_admin[k] = P(admin=k)``; normalised internally if it does
            not already sum to 1.

    Returns:
        (K, J) column-stochastic emission ``C[k|j]``. A self-declared category that no admin category
        maps onto (a structurally zero column) falls back to that category's admin marginal, so the
        column is always a proper distribution rather than silently zero.
    """
    R = np.asarray(reclassification, dtype=float)
    p = np.asarray(admin_marginal, dtype=float)
    if R.ndim != 2:
        raise EcologicalRaceError("reclassification must be a 2-D (K, J) matrix.")
    K, J = R.shape
    if p.shape != (K,):
        raise EcologicalRaceError(f"admin_marginal shape {p.shape} != expected ({K},).")
    if np.any(R < 0) or not np.all(np.isfinite(R)):
        raise EcologicalRaceError("reclassification has negative or non-finite entries.")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise EcologicalRaceError("admin_marginal has negative or non-finite entries.")
    row_sums = R.sum(axis=1)
    if np.max(np.abs(row_sums - 1.0)) > 1e-6:
        raise EcologicalRaceError("reclassification must be row-stochastic P(self|admin) (rows sum to 1).")
    total = p.sum()
    if total <= 0:
        raise EcologicalRaceError("admin_marginal must have positive total mass.")
    p = p / total

    joint = R * p[:, None]  # joint[k, j] = P(admin=k, self=j)
    col = joint.sum(axis=0, keepdims=True)  # P(self=j)
    C = np.divide(joint, col, out=np.zeros_like(joint), where=col > 0)
    # A structurally empty self column (no admin maps onto it) → fall back to the admin marginal so
    # the column is a valid distribution, never silently all-zero.
    empty = (col.ravel() <= 0)
    if np.any(empty):
        C[:, empty] = p[:, None]
    return C


def reclassification_from_emission(
    emission: np.ndarray,
    self_marginal: np.ndarray,
) -> np.ndarray:
    """Convert an emission ``P(admin=k | self-declared=j)`` to the reclassification ``P(self-declared=j
    | admin=k)`` the race-bridge registry stores (rows = admin, sum to 1).

    Exact mirror of :func:`emission_from_reclassification` (swap the admin/self roles); Bayes with the
    census self-declared marginal ``p_self[j] = P(self-declared=j)``:

        R[k, j] = P(self=j | admin=k)
                = C[k, j] p_self[j] / Σ_{j'} C[k, j'] p_self[j']

    This is the direction a calibrated ecological confusion estimate must be written back in so the
    existing local-pi bridge (which consumes the row-stochastic reclassification matrix) uses it
    unchanged.

    Args:
        emission: (K, J) column-stochastic ``C[k|j] = P(admin=k | self=j)``.
        self_marginal: (J,) census ``p_self[j] = P(self=j)``; normalised internally.

    Returns:
        (K, J) row-stochastic reclassification ``R[k, j] = P(self=j | admin=k)``. An admin category
        that nothing maps onto (a structurally zero row) falls back to the self marginal so the row is
        a proper distribution rather than silently zero.
    """
    C = np.asarray(emission, dtype=float)
    q = np.asarray(self_marginal, dtype=float)
    if C.ndim != 2:
        raise EcologicalRaceError("emission must be a 2-D (K, J) matrix.")
    K, J = C.shape
    if q.shape != (J,):
        raise EcologicalRaceError(f"self_marginal shape {q.shape} != expected ({J},).")
    if np.any(C < 0) or not np.all(np.isfinite(C)):
        raise EcologicalRaceError("emission has negative or non-finite entries.")
    if np.any(q < 0) or not np.all(np.isfinite(q)):
        raise EcologicalRaceError("self_marginal has negative or non-finite entries.")
    col_sums = C.sum(axis=0)
    if np.max(np.abs(col_sums - 1.0)) > 1e-6:
        raise EcologicalRaceError("emission must be column-stochastic P(admin|self) (columns sum to 1).")
    total = q.sum()
    if total <= 0:
        raise EcologicalRaceError("self_marginal must have positive total mass.")
    q = q / total

    joint = C * q[None, :]  # joint[k, j] = P(admin=k, self=j)
    row = joint.sum(axis=1, keepdims=True)  # P(admin=k)
    R = np.divide(joint, row, out=np.zeros_like(joint), where=row > 0)
    empty = (row.ravel() <= 0)
    if np.any(empty):
        R[empty, :] = q[None, :]
    return R


@dataclass(frozen=True)
class EcologicalRaceProblem:
    """Inputs for one ecological deconvolution.

    ``N`` is the KNOWN census self-declared population per cell (S x J). ``Y_by_stratum`` holds one
    (S x K) administrative-race count array per stratum, all sharing the same latent rate process and
    the same cell/self-declared axes but each with its own emission matrix. ``emission_prior`` is a
    (K x J) *column-stochastic* ``P(admin = k | self-declared = j)`` prior (NOT the registry's
    row-stochastic reclassification matrix); when omitted a soft-identity prior is used.
    """

    N: np.ndarray
    Y_by_stratum: list[np.ndarray]
    self_declared_categories: list[str]
    admin_categories: list[str]
    stratum_labels: list[str]
    emission_prior: np.ndarray | None = None

    def __post_init__(self) -> None:
        N = np.asarray(self.N, dtype=float)
        if N.ndim != 2:
            raise EcologicalRaceError("N must be a 2-D (S, J) array of census self-declared population.")
        S, J = N.shape
        if J != len(self.self_declared_categories):
            raise EcologicalRaceError(
                f"N has J={J} columns but {len(self.self_declared_categories)} self_declared_categories."
            )
        if not self.Y_by_stratum:
            raise EcologicalRaceError("Y_by_stratum must contain at least one stratum observation.")
        if len(self.Y_by_stratum) != len(self.stratum_labels):
            raise EcologicalRaceError("Y_by_stratum and stratum_labels length mismatch.")
        K = len(self.admin_categories)
        for label, Y in zip(self.stratum_labels, self.Y_by_stratum):
            Ya = np.asarray(Y, dtype=float)
            if Ya.shape != (S, K):
                raise EcologicalRaceError(
                    f"stratum {label!r}: Y shape {Ya.shape} != expected ({S}, {K})."
                )
            if np.any(Ya < 0) or not np.all(np.isfinite(Ya)):
                raise EcologicalRaceError(f"stratum {label!r}: Y has negative or non-finite counts.")
        if np.any(N < 0) or not np.all(np.isfinite(N)):
            raise EcologicalRaceError("N has negative or non-finite population.")
        if self.emission_prior is not None:
            C0 = np.asarray(self.emission_prior, dtype=float)
            if C0.shape != (K, J):
                raise EcologicalRaceError(f"emission_prior shape {C0.shape} != expected ({K}, {J}).")
            colsums = C0.sum(axis=0)
            if np.any(colsums <= 0) or np.max(np.abs(colsums - 1.0)) > 1e-6:
                raise EcologicalRaceError(
                    "emission_prior must be column-stochastic P(admin=k|self=j) (each column sums to 1)."
                )

    @property
    def n_cells(self) -> int:
        return int(np.asarray(self.N).shape[0])

    @property
    def n_self(self) -> int:
        return len(self.self_declared_categories)

    @property
    def n_admin(self) -> int:
        return len(self.admin_categories)

    @property
    def n_strata(self) -> int:
        return len(self.stratum_labels)


@dataclass(frozen=True)
class EcologicalRaceResult:
    rate_ratios: dict[str, float]
    log_rate_ratios: dict[str, float]
    cell_baseline_log_rate: np.ndarray
    latent_rate: np.ndarray  # lambda[s, j] = exp(a[s] + b[j]), per-capita true rate by self-declared race
    emission_by_stratum: dict[str, np.ndarray]
    contextual_identifiability: float
    identifiable: bool
    converged: bool
    n_iter: int
    reference_category: str
    uncertainty_method: str
    rate_ratio_cv: dict[str, float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _softmax_cols(Z: np.ndarray) -> np.ndarray:
    """Column softmax: each column (a self-declared race j) sums to 1 over admin categories k."""
    Zs = Z - Z.max(axis=0, keepdims=True)
    E = np.exp(Zs)
    return E / E.sum(axis=0, keepdims=True)


def _unpack(theta: np.ndarray, S: int, J: int, K: int, G: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    Zs = [theta[g * K * J:(g + 1) * K * J].reshape(K, J) for g in range(G)]
    off = G * K * J
    a = theta[off:off + S]
    b = np.concatenate([[0.0], theta[off + S:off + S + J - 1]])
    return Zs, a, b


def _neg_log_post_and_grad(
    theta: np.ndarray,
    Ys: list[np.ndarray],
    N: np.ndarray,
    Z0: np.ndarray,
    kappa: float,
) -> tuple[float, np.ndarray]:
    """Poisson negative log-posterior and its analytic gradient (gradcheck-validated ~1e-6)."""
    S, J = N.shape
    K = Z0.shape[0]
    G = len(Ys)
    Zs, a, b = _unpack(theta, S, J, K, G)
    m = np.exp(a[:, None] + b[None, :]) * N  # (S, J) latent true-race event mass
    nll = 0.0
    dm = np.zeros_like(m)
    gZs: list[np.ndarray] = []
    for Y, Z in zip(Ys, Zs):
        C = _softmax_cols(Z)  # (K, J)
        mu = np.clip(m @ C.T, 1e-9, None)  # (S, K)
        nll += float((mu - Y * np.log(mu)).sum())
        g = 1.0 - Y / mu  # (S, K)
        dm += g @ C  # (S, J)
        dC = g.T @ m  # (K, J)
        gZ = C * (dC - (dC * C).sum(axis=0, keepdims=True)) + kappa * (Z - Z0)
        gZs.append(gZ)
    nll += sum(0.5 * kappa * float(((Z - Z0) ** 2).sum()) for Z in Zs)
    da = (dm * m).sum(axis=1)  # (S,)
    db = (dm * m).sum(axis=0)[1:]  # (J-1,)
    grad = np.concatenate([gz.ravel() for gz in gZs] + [da, db])
    return nll, grad


def contextual_identifiability(N: np.ndarray) -> float:
    """Per-cell RMS composition variation in the least-varying non-degenerate direction.

    The ecological deconvolution can only separate the self-declared races when their *shares* vary
    across cells; if every cell has the same composition the system is rank-deficient and b is
    unidentified. Compositions live on a (J-1)-simplex, so the smallest singular value of the
    column-centred composition matrix is structurally ~0 (the sum-to-one null direction); the
    *second-smallest* is the binding one. This returns that singular value divided by sqrt(S), i.e.
    the absolute per-cell RMS variation (in share units, ~[0, 0.5]) of the hardest-to-identify
    non-degenerate contrast. Larger => more cross-cell contextual variation => better identifiability.
    It is an absolute magnitude, NOT a ratio to the largest singular value (that measures conditioning,
    which is non-monotone in the variation that actually drives ecological identifiability).
    """
    N = np.asarray(N, dtype=float)
    totals = N.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 0, 1.0, totals)
    comp = N / totals  # (S, J) row-stochastic composition
    comp = comp - comp.mean(axis=0, keepdims=True)
    S, J = comp.shape
    if S < J or J < 2:
        return 0.0
    sv = np.linalg.svd(comp, compute_uv=False)
    if sv.size < J - 1 or sv[0] <= 0:
        return 0.0
    return float(sv[J - 2] / np.sqrt(S))  # smallest non-degenerate sv, per-cell RMS


def _default_prior_logits(K: int, J: int, emission_prior: np.ndarray | None) -> np.ndarray:
    if emission_prior is not None:
        return np.log(np.clip(np.asarray(emission_prior, dtype=float), 1e-6, 1.0))
    # Soft identity when K == J: admin usually matches self-declared, with mass to spare for confusion.
    base = np.full((K, J), 0.03, dtype=float)
    for d in range(min(K, J)):
        base[d, d] = 0.85
    base = base / base.sum(axis=0, keepdims=True)
    return np.log(np.clip(base, 1e-6, 1.0))


def _fit_map(
    Ys: list[np.ndarray],
    N: np.ndarray,
    Z0: np.ndarray,
    kappa: float,
    *,
    init: np.ndarray | None,
    max_iter: int,
    b_bound: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, bool, int]:
    S, J = N.shape
    K = Z0.shape[0]
    G = len(Ys)
    if init is None:
        total = sum(float(Y.sum()) for Y in Ys) / max(G, 1)
        crude = np.log(max(total, 1.0) / max(float(N.sum()), 1.0))
        init = np.concatenate(
            [Z0.ravel() for _ in range(G)] + [np.full(S, crude), np.zeros(J - 1)]
        )
    bounds = (
        [(-15.0, 15.0)] * (G * K * J)
        + [(-25.0, 5.0)] * S
        + [(-b_bound, b_bound)] * (J - 1)
    )
    res = minimize(
        lambda t: _neg_log_post_and_grad(t, Ys, N, Z0, kappa),
        init,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    Zs, a, b = _unpack(res.x, S, J, K, G)
    return [_softmax_cols(Z) for Z in Zs], a, b, res.x, bool(res.success), int(res.nit)


def fit_ecological_race_deconvolution(
    problem: EcologicalRaceProblem,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    max_iter: int = DEFAULT_MAX_ITER,
    b_bound: float = 2.5,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 20260709,
) -> EcologicalRaceResult:
    """Fit the ecological race-deconvolution and recover per-self-declared-race rate ratios.

    ``prior_strength`` (kappa) is the trust in ``emission_prior``; keep it weak. A large kappa forces
    C toward the prior (identity) and provably re-erases the racial signal, so this never auto-inflates
    it. When ``bootstrap_replicates > 0`` a bounded parametric bootstrap (resample Y ~ Poisson(mu_hat),
    refit) reports a coefficient of variation on each rate ratio; the cost is that many refits and is
    surfaced via ``uncertainty_method`` rather than silently skipped.
    """
    N = np.asarray(problem.N, dtype=float)
    Ys = [np.asarray(Y, dtype=float) for Y in problem.Y_by_stratum]
    J = problem.n_self
    K = problem.n_admin
    if prior_strength < 0:
        raise EcologicalRaceError("prior_strength (kappa) must be nonnegative.")

    Z0 = _default_prior_logits(K, J, problem.emission_prior)
    ident = contextual_identifiability(N)
    identifiable = ident >= IDENTIFIABILITY_FLOOR

    Chats, a, b, xhat, converged, n_iter = _fit_map(
        Ys, N, Z0, prior_strength, init=None, max_iter=max_iter, b_bound=b_bound
    )

    if not converged:
        warnings.warn(
            "ecological race deconvolution did not converge within "
            f"{max_iter} iterations; treat rate ratios as provisional.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not identifiable:
        warnings.warn(
            "ecological race deconvolution is weakly identified "
            f"(contextual_identifiability={ident:.3f} < {IDENTIFIABILITY_FLOOR}): cell compositions "
            "vary too little to separate self-declared races; rate ratios are prior-dominated.",
            RuntimeWarning,
            stacklevel=2,
        )

    cats = problem.self_declared_categories
    rate_ratios = {cats[j]: float(np.exp(b[j])) for j in range(J)}
    log_ratios = {cats[j]: float(b[j]) for j in range(J)}
    latent = np.exp(a[:, None] + b[None, :])  # (S, J)
    emission = {problem.stratum_labels[g]: Chats[g] for g in range(len(Chats))}

    cv: dict[str, float] | None = None
    method = "map_only"
    if bootstrap_replicates > 0:
        cv = _bootstrap_rate_ratio_cv(
            Ys, N, Z0, prior_strength, xhat, cats,
            replicates=bootstrap_replicates, seed=bootstrap_seed,
            max_iter=max_iter, b_bound=b_bound,
        )
        method = f"parametric_bootstrap_{bootstrap_replicates}"

    return EcologicalRaceResult(
        rate_ratios=rate_ratios,
        log_rate_ratios=log_ratios,
        cell_baseline_log_rate=a,
        latent_rate=latent,
        emission_by_stratum=emission,
        contextual_identifiability=ident,
        identifiable=identifiable,
        converged=converged,
        n_iter=n_iter,
        reference_category=cats[0],
        uncertainty_method=method,
        rate_ratio_cv=cv,
        diagnostics={
            "n_cells": problem.n_cells,
            "n_self_declared": J,
            "n_admin": K,
            "n_strata": problem.n_strata,
            "prior_strength": float(prior_strength),
            "identifiability_floor": IDENTIFIABILITY_FLOOR,
        },
    )


def _bootstrap_rate_ratio_cv(
    Ys: list[np.ndarray],
    N: np.ndarray,
    Z0: np.ndarray,
    kappa: float,
    xhat: np.ndarray,
    cats: list[str],
    *,
    replicates: int,
    seed: int,
    max_iter: int,
    b_bound: float,
) -> dict[str, float]:
    """Parametric bootstrap CV on each rate ratio. Refit from the MAP for speed and stability."""
    S, J = N.shape
    K = Z0.shape[0]
    G = len(Ys)
    Chats, a, b = _unpack(xhat, S, J, K, G)
    m = np.exp(a[:, None] + b[None, :]) * N
    mus = [np.clip(m @ _softmax_cols(Z).T, 1e-9, None) for Z in Chats]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {c: [] for c in cats}
    for _ in range(replicates):
        Yb = [rng.poisson(mu).astype(float) for mu in mus]
        _, _, bb, _, _, _ = _fit_map(
            Yb, N, Z0, kappa, init=xhat.copy(), max_iter=max_iter, b_bound=b_bound
        )
        for j in range(J):
            draws[cats[j]].append(float(np.exp(bb[j])))
    cv: dict[str, float] = {}
    for c, vals in draws.items():
        if len(vals) < 2:
            cv[c] = 0.0
            continue
        mean = float(np.mean(vals))
        cv[c] = float(np.std(vals, ddof=1) / mean) if mean > 0 else 0.0
    return cv
