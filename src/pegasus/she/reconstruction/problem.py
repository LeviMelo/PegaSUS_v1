"""Constrained Tensor Reconstruction (CTR) kernel (MSD-II §II.3, MII-CTR-01).

Unification: the population tensor (MSD §2.8), age-bin disaggregation, synthetic
context cubes, and ST-DFM latent factors (§2.10) are *one* object — a constrained
optimization over a non-negative latent tensor with anchors, a linear observation
operator, penalties (2nd-difference on age/time, spatial Laplacian from §II.4),
marginals, and an optional compositional axis.

``CTRProblem`` is that object. The **population tensor is the canonical instance**
(see ``instances.population_ctr_instance``): rather than re-deriving its bespoke
aging/birth/ILR math, that instance carries a ``native_evaluator`` that delegates
to ``she.reconstruction.loss.evaluate_population_loss`` byte-for-byte. Instances whose
structure is genuinely linear (age-bin disaggregation, cube reconstruction) use
the generic weighted-least-squares + quadratic-penalty core here — no bespoke code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ObservationTerm:
    """A weighted linear observation ``weight * || A x - values ||^2``.

    ``A`` maps the flattened latent vector to observed quantities (e.g. an
    aggregation matrix that sums fine age-bins to a broad bin, or a selector for
    anchored cells). ``weight`` may be a scalar or a per-observation vector.
    """

    A: np.ndarray
    values: np.ndarray
    weight: float | np.ndarray = 1.0
    name: str = "observation"


@dataclass(frozen=True)
class QuadraticPenalty:
    """A quadratic smoothness/structure penalty.

    Either ``D`` (a difference/structure operator; penalty ``weight*||D x||^2``,
    e.g. a 2nd-difference matrix along age/time) OR ``Q`` (a PSD quadratic-form
    matrix; penalty ``weight * x^T Q x``, e.g. the spatial Laplacian
    ``graph.view("laplacian")`` from the SpatialWeightGraph, §II.4).
    """

    D: np.ndarray | None = None
    Q: np.ndarray | None = None
    weight: float = 1.0
    name: str = "penalty"

    def form(self) -> np.ndarray:
        """The quadratic-form matrix (``D^T D`` or ``Q``)."""
        if self.Q is not None:
            return self.Q
        if self.D is not None:
            return self.D.T @ self.D
        raise ValueError("QuadraticPenalty requires either D or Q")


@dataclass(frozen=True)
class MarginalConstraint:
    """A soft marginal/closure ``weight * (S x - totals)^2`` (S sums over an axis)."""

    S: np.ndarray
    totals: np.ndarray
    weight: float = 1.0
    name: str = "marginal"


@dataclass(frozen=True)
class CTRProblem:
    latent_shape: tuple[int, ...]
    observations: tuple[ObservationTerm, ...] = ()
    penalties: tuple[QuadraticPenalty, ...] = ()
    marginals: tuple[MarginalConstraint, ...] = ()
    nonnegative: bool = True
    # Instances whose objective is bespoke (the population tensor) delegate here;
    # when set, the generic terms above are ignored and this is the whole loss.
    native_evaluator: Callable[[np.ndarray], tuple[float, np.ndarray, dict[str, float]]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        result = 1
        for dim in self.latent_shape:
            result *= dim
        return result


@dataclass(frozen=True)
class CTREvaluation:
    loss: float
    gradient: np.ndarray
    terms: dict[str, float]


def evaluate_ctr(problem: CTRProblem, x: np.ndarray | list[float]) -> CTREvaluation:
    """Evaluate the CTR objective and gradient at latent ``x`` (flattened)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != problem.n:
        raise ValueError(f"CTR latent vector size {x.size} != problem.n {problem.n}")
    if not np.isfinite(x).all():
        raise ValueError("CTR received a non-finite iterate.")

    if problem.native_evaluator is not None:
        loss, grad, terms = problem.native_evaluator(x)
        return CTREvaluation(float(loss), np.asarray(grad, dtype=np.float64).reshape(-1), dict(terms))

    grad = np.zeros_like(x)
    terms: dict[str, float] = {}

    for obs in problem.observations:
        residual = obs.A @ x - obs.values
        w = obs.weight
        wr = (w * residual) if np.ndim(w) else (w * residual)
        terms[obs.name] = terms.get(obs.name, 0.0) + float(np.sum(w * residual**2))
        grad += 2.0 * (obs.A.T @ wr)

    for pen in problem.penalties:
        if pen.D is not None:
            Dx = pen.D @ x
            terms[pen.name] = terms.get(pen.name, 0.0) + float(pen.weight * np.sum(Dx**2))
            grad += 2.0 * pen.weight * (pen.D.T @ Dx)
        else:
            Qx = pen.Q @ x
            terms[pen.name] = terms.get(pen.name, 0.0) + float(pen.weight * float(x @ Qx))
            grad += 2.0 * pen.weight * Qx

    for marg in problem.marginals:
        residual = marg.S @ x - marg.totals
        terms[marg.name] = terms.get(marg.name, 0.0) + float(marg.weight * np.sum(residual**2))
        grad += 2.0 * marg.weight * (marg.S.T @ residual)

    loss = float(sum(terms.values()))
    return CTREvaluation(loss, grad, terms)


def _quadratic_hessian(problem: CTRProblem) -> np.ndarray | None:
    """Assemble the (constant) Hessian for a generic quadratic CTR, or None.

    For the generic instances every term is a quadratic form, so the objective is
    convex quadratic with Hessian ``H = 2(Σ wᵢ Aᵢᵀ Aᵢ + Σ wⱼ Dⱼᵀ Dⱼ + Σ wₖ Sₖᵀ Sₖ)``.
    Knowing ``H`` lets the projected-gradient solver use the guaranteed-convergent
    Lipschitz step ``1/λ_max(H)`` instead of a hand-tuned step.
    """
    if problem.native_evaluator is not None:
        return None
    n = problem.n
    H = np.zeros((n, n), dtype=np.float64)
    for obs in problem.observations:
        w = obs.weight
        A = obs.A
        H += 2.0 * (A.T @ (A * (w if np.ndim(w) else w)))
    for pen in problem.penalties:
        H += 2.0 * pen.weight * pen.form()
    for marg in problem.marginals:
        H += 2.0 * marg.weight * (marg.S.T @ marg.S)
    return H


def solve_ctr(
    problem: CTRProblem,
    x0: np.ndarray | list[float] | None = None,
    *,
    max_iters: int = 5000,
    tol: float = 1e-10,
    step: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Projected-gradient solve with non-negativity projection.

    The CTR *interface* is locked (MSD-II §II.3); the backend is a replaceable
    Mutable Solver. For generic (quadratic) instances the step defaults to the
    Lipschitz constant ``1/λ_max(H)`` (convergent for convex problems); for native
    instances (population) a backtracking line search is used.
    """
    n = problem.n
    x = np.asarray(x0, dtype=np.float64).reshape(-1) if x0 is not None else np.ones(n, dtype=np.float64)
    if problem.nonnegative:
        x = np.maximum(x, 0.0)

    H = _quadratic_hessian(problem) if step is None else None
    lipschitz_step: float | None = None
    if H is not None and H.size:
        # Direct normal-equations solve for the convex quadratic: grad(x)=Hx-c,
        # so the unconstrained optimum solves Hx=c with c=-grad(0). If it is
        # already feasible (non-negative) we are done in one linear solve.
        c = -evaluate_ctr(problem, np.zeros(n)).gradient
        try:
            x_star = np.linalg.lstsq(H, c, rcond=None)[0]
        except np.linalg.LinAlgError:
            x_star = None
        if x_star is not None and (not problem.nonnegative or bool((x_star >= -1e-9).all())):
            x_star = np.maximum(x_star, 0.0) if problem.nonnegative else x_star
            ev = evaluate_ctr(problem, x_star)
            return x_star, {"iterations": 0, "final_loss": ev.loss, "terms": ev.terms, "converged": True, "solver": "normal_equations"}
        if x_star is not None:
            x = np.maximum(x_star, 0.0)  # warm-start projected gradient from clipped optimum
        lam_max = float(np.linalg.eigvalsh(H)[-1]) if n <= 2000 else float(np.linalg.norm(H, 2))
        lipschitz_step = 1.0 / lam_max if lam_max > 0 else None

    ev = evaluate_ctr(problem, x)
    prev_loss = ev.loss
    iters = 0
    for iters in range(1, max_iters + 1):
        if lipschitz_step is not None:
            candidate = x - lipschitz_step * ev.gradient
            if problem.nonnegative:
                candidate = np.maximum(candidate, 0.0)
            cand_ev = evaluate_ctr(problem, candidate)
        else:
            s = step if step is not None else 1e-2
            for _ in range(40):
                candidate = x - s * ev.gradient
                if problem.nonnegative:
                    candidate = np.maximum(candidate, 0.0)
                cand_ev = evaluate_ctr(problem, candidate)
                if cand_ev.loss <= ev.loss:
                    break
                s *= 0.5
        x, ev = candidate, cand_ev
        if abs(prev_loss - ev.loss) <= tol * (1.0 + abs(prev_loss)):
            break
        prev_loss = ev.loss
    diagnostics = {
        "iterations": iters,
        "final_loss": ev.loss,
        "terms": ev.terms,
        "converged": iters < max_iters,
    }
    return x, diagnostics


__all__ = [
    "CTRProblem",
    "ObservationTerm",
    "QuadraticPenalty",
    "MarginalConstraint",
    "CTREvaluation",
    "evaluate_ctr",
    "solve_ctr",
]
