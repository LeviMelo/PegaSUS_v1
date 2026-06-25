"""Real generalized-linear-model kernel for PIRS (MSD §6.2, §6.4, §6.5, §6.6.1).

This module replaces the previous PIRS modelling theatre, where every family
except Gaussian/Poisson raised ``NotImplementedError`` and "cross-fitted"
residuals were a manifest label with no K-fold computation behind it.

It is intentionally dependency-light (NumPy only) so it can run inside the
compiler without a SciPy/statsmodels dependency. Every family is fitted by a
single iteratively reweighted least squares (IRLS) loop with a family-specific
(link, variance, deviance) triple. Residuals follow the MSD §6.5 residual
registry. Cross-fitted residuals (MSD §6.6.1) are computed by genuinely
holding out blocks and predicting the held-out fold from the complement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# Families the kernel can actually fit. Anything outside this set must raise,
# never be silently downgraded to OLS.
SUPPORTED_FAMILIES: frozenset[str] = frozenset(
    {
        "gaussian_identity",
        "ols",
        "poisson_count_with_log_offset",
        "negative_binomial",
        "gamma",
        "sih_gamma_cost_component",
        "binomial_proportion",
    }
)

# MSD §6.5 residual registry: which residual the HSIC scanner consumes.
RESIDUAL_TYPE_BY_FAMILY: dict[str, str] = {
    "gaussian_identity": "standardized",
    "ols": "standardized",
    "poisson_count_with_log_offset": "deviance",
    "negative_binomial": "deviance",
    "gamma": "deviance",
    "sih_gamma_cost_component": "deviance",
    "binomial_proportion": "deviance",
}

_EPS = 1e-9
_ETA_CLAMP = 30.0


class GLMError(ValueError):
    """Raised when a GLM cannot be fitted for structural reasons."""


@dataclass(frozen=True)
class _Family:
    """Canonical (link, variance, deviance) triple for an exponential family."""

    name: str
    link: str  # "identity" | "log" | "logit"
    linkfun: Callable[[np.ndarray], np.ndarray]
    linkinv: Callable[[np.ndarray], np.ndarray]
    mu_eta: Callable[[np.ndarray], np.ndarray]  # d mu / d eta
    variance: Callable[[np.ndarray], np.ndarray]
    unit_deviance: Callable[[np.ndarray, np.ndarray], np.ndarray]
    uses_log_offset: bool = False


def _clamp_eta(eta: np.ndarray) -> np.ndarray:
    return np.clip(eta, -_ETA_CLAMP, _ETA_CLAMP)


def _gaussian() -> _Family:
    return _Family(
        name="gaussian_identity",
        link="identity",
        linkfun=lambda mu: mu,
        linkinv=lambda eta: eta,
        mu_eta=lambda eta: np.ones_like(eta),
        variance=lambda mu: np.ones_like(mu),
        unit_deviance=lambda y, mu: (y - mu) ** 2,
    )


def _poisson() -> _Family:
    def dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        mu = np.maximum(mu, _EPS)
        term = np.where(y > 0, y * np.log(np.maximum(y, _EPS) / mu), 0.0)
        return 2.0 * (term - (y - mu))

    return _Family(
        name="poisson_count_with_log_offset",
        link="log",
        linkfun=lambda mu: np.log(np.maximum(mu, _EPS)),
        linkinv=lambda eta: np.exp(_clamp_eta(eta)),
        mu_eta=lambda eta: np.exp(_clamp_eta(eta)),
        variance=lambda mu: np.maximum(mu, _EPS),
        unit_deviance=dev,
        uses_log_offset=True,
    )


def _gamma() -> _Family:
    # Log link (MSD §2.10.1 uses log for monetary/positive-continuous fields).
    def dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y = np.maximum(y, _EPS)
        mu = np.maximum(mu, _EPS)
        return 2.0 * (-np.log(y / mu) + (y - mu) / mu)

    return _Family(
        name="gamma",
        link="log",
        linkfun=lambda mu: np.log(np.maximum(mu, _EPS)),
        linkinv=lambda eta: np.exp(_clamp_eta(eta)),
        mu_eta=lambda eta: np.exp(_clamp_eta(eta)),
        variance=lambda mu: np.maximum(mu, _EPS) ** 2,
        unit_deviance=dev,
    )


def _negative_binomial(theta: float) -> _Family:
    theta = max(float(theta), _EPS)

    def dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        mu = np.maximum(mu, _EPS)
        term1 = np.where(y > 0, y * np.log(np.maximum(y, _EPS) / mu), 0.0)
        term2 = (y + theta) * np.log((y + theta) / (mu + theta))
        return 2.0 * (term1 - term2)

    return _Family(
        name="negative_binomial",
        link="log",
        linkfun=lambda mu: np.log(np.maximum(mu, _EPS)),
        linkinv=lambda eta: np.exp(_clamp_eta(eta)),
        mu_eta=lambda eta: np.exp(_clamp_eta(eta)),
        variance=lambda mu: np.maximum(mu, _EPS) + (np.maximum(mu, _EPS) ** 2) / theta,
        unit_deviance=dev,
        uses_log_offset=True,
    )


def _binomial() -> _Family:
    # Quasi-binomial logit on a proportion in (0, 1). Prior weights carry n.
    def dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y = np.clip(y, 0.0, 1.0)
        mu = np.clip(mu, _EPS, 1.0 - _EPS)
        t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        t2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return 2.0 * (t1 + t2)

    def linkinv(eta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-_clamp_eta(eta)))

    return _Family(
        name="binomial_proportion",
        link="logit",
        linkfun=lambda mu: np.log(np.clip(mu, _EPS, 1 - _EPS) / (1 - np.clip(mu, _EPS, 1 - _EPS))),
        linkinv=linkinv,
        mu_eta=lambda eta: linkinv(eta) * (1.0 - linkinv(eta)),
        variance=lambda mu: np.clip(mu, _EPS, 1 - _EPS) * (1.0 - np.clip(mu, _EPS, 1 - _EPS)),
        unit_deviance=dev,
    )


def _resolve_family(family: str, *, y: np.ndarray, nb_theta: float | None) -> _Family:
    if family in {"gaussian_identity", "ols"}:
        return _gaussian()
    if family == "poisson_count_with_log_offset":
        return _poisson()
    if family in {"gamma", "sih_gamma_cost_component"}:
        return _gamma()
    if family == "negative_binomial":
        return _negative_binomial(nb_theta if nb_theta is not None else _estimate_nb_theta(y))
    if family == "binomial_proportion":
        return _binomial()
    raise GLMError(f"unsupported_glm_family:{family}")


def _estimate_nb_theta(y: np.ndarray) -> float:
    """Method-of-moments dispersion seed for the negative binomial."""
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y)) if y.size else 0.0
    var = float(np.var(y, ddof=1)) if y.size > 1 else mean
    if var <= mean + _EPS:
        return 1e6  # effectively Poisson
    return max(mean * mean / (var - mean), _EPS)


@dataclass
class GLMResult:
    family: str
    residual_type: str
    terms: list[str]
    coefficients: np.ndarray
    fitted: np.ndarray
    linear_predictor: np.ndarray
    deviance_residuals: np.ndarray
    pearson_residuals: np.ndarray
    standardized_residuals: np.ndarray
    deviance: float
    dispersion: float
    n_iter: int
    converged: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def primary_residual(self) -> np.ndarray:
        if self.residual_type == "standardized":
            return self.standardized_residuals
        if self.residual_type == "pearson":
            return self.pearson_residuals
        return self.deviance_residuals


def fit_glm(
    *,
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    offset: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    term_names: Sequence[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    nb_theta: float | None = None,
) -> GLMResult:
    """Fit a GLM by IRLS. NumPy-only, no silent family downgrades."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise GLMError("design_matrix_shape_mismatch")
    n, p = X.shape
    if n == 0:
        raise GLMError("empty_design_matrix")
    fam = _resolve_family(family, y=y, nb_theta=nb_theta)
    offset_vec = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    weights0 = np.ones(n) if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()

    # Sensible mu initialisation per family.
    if fam.link == "log":
        mu = np.maximum(y, 0.0) + 0.1
    elif fam.link == "logit":
        mu = np.clip((weights0 * y + 0.5) / (weights0 + 1.0), _EPS, 1 - _EPS)
    else:
        mu = y.copy()
    eta = fam.linkfun(mu) - (offset_vec if fam.uses_log_offset else 0.0)

    ridge = np.eye(p) * 1e-10
    beta = np.zeros(p)
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        mu_eta = np.maximum(np.abs(fam.mu_eta(eta + offset_vec)), _EPS) * np.sign(
            fam.mu_eta(eta + offset_vec) + _EPS
        )
        mu = fam.linkinv(eta + offset_vec)
        var = np.maximum(fam.variance(mu), _EPS)
        w = weights0 * (mu_eta ** 2) / var
        z = eta + (y - mu) / mu_eta
        sw = np.sqrt(np.maximum(w, 0.0))
        xw = X * sw[:, None]
        zw = z * sw
        lhs = xw.T @ xw + ridge
        rhs = xw.T @ zw
        try:
            next_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            next_beta = np.linalg.pinv(lhs) @ rhs
        next_eta = X @ next_beta
        delta = float(np.max(np.abs(next_beta - beta))) if p else 0.0
        beta = next_beta
        eta = next_eta
        if delta < tol:
            converged = True
            break

    mu = fam.linkinv(eta + offset_vec)
    unit_dev = np.maximum(fam.unit_deviance(y, mu), 0.0)
    deviance = float(np.sum(weights0 * unit_dev))
    dev_resid = np.sign(y - mu) * np.sqrt(weights0 * unit_dev)
    var = np.maximum(fam.variance(mu), _EPS)
    pearson = (y - mu) * np.sqrt(weights0) / np.sqrt(var)
    dof = max(n - p, 1)
    dispersion = float(np.sum(pearson ** 2) / dof)
    if family in {"gaussian_identity", "ols"}:
        sd = math.sqrt(dispersion) if dispersion > 0 else 0.0
        standardized = (y - mu) / sd if sd > 0 else (y - mu) * 0.0
    else:
        standardized = dev_resid

    term_list = list(term_names) if term_names is not None else [f"x{i}" for i in range(p)]
    return GLMResult(
        family=family,
        residual_type=RESIDUAL_TYPE_BY_FAMILY.get(family, "deviance"),
        terms=term_list,
        coefficients=beta,
        fitted=mu,
        linear_predictor=eta + offset_vec,
        deviance_residuals=dev_resid,
        pearson_residuals=pearson,
        standardized_residuals=standardized,
        deviance=deviance,
        dispersion=dispersion,
        n_iter=iteration,
        converged=converged,
    )


def make_blocked_folds(
    n: int,
    *,
    n_folds: int,
    block_index: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Partition rows into K folds (MSD §6.6.1).

    When ``block_index`` is supplied (e.g. a spatial or temporal block id per
    row), whole blocks are assigned to folds so spatial/temporal dependence is
    preserved across the train/test split. Otherwise a deterministic contiguous
    partition is used.
    """
    n_folds = max(int(n_folds), 2)
    if n < n_folds:
        n_folds = max(n, 2)
    if block_index is not None:
        blocks = np.asarray(block_index)
        uniq = list(dict.fromkeys(blocks.tolist()))
        fold_of_block = {b: i % n_folds for i, b in enumerate(uniq)}
        assign = np.array([fold_of_block[b] for b in blocks.tolist()])
    else:
        assign = np.floor(np.arange(n) * n_folds / max(n, 1)).astype(int)
        assign = np.clip(assign, 0, n_folds - 1)
    folds = [np.where(assign == k)[0] for k in range(n_folds)]
    return [f for f in folds if f.size > 0]


def crossfit_residuals(
    *,
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    offset: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    n_folds: int = 5,
    block_index: Sequence[int] | None = None,
    nb_theta: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Out-of-fold residuals: e_i = Res(Y_i, M_{-k}(X_i)) for i in fold k.

    Each fold's residuals are produced by a model fitted on the *complement*
    of that fold, never on the fold itself. This is the genuine computation the
    previous ``crossfit.py`` only claimed via a manifest label.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n = y.shape[0]
    offset_vec = None if offset is None else np.asarray(offset, dtype=float).ravel()
    weights = None if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()
    folds = make_blocked_folds(n, n_folds=n_folds, block_index=block_index)
    residual_type = RESIDUAL_TYPE_BY_FAMILY.get(family, "deviance")
    oof = np.full(n, np.nan)
    used_folds = 0
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        if train_idx.size <= X.shape[1]:
            continue  # not enough rows to identify coefficients
        fit = fit_glm(
            y=y[train_idx],
            X=X[train_idx],
            family=family,
            offset=None if offset_vec is None else offset_vec[train_idx],
            prior_weights=None if weights is None else weights[train_idx],
            nb_theta=nb_theta,
        )
        oof[test_idx] = _predict_residuals(
            fit=fit,
            y=y[test_idx],
            X=X[test_idx],
            offset=None if offset_vec is None else offset_vec[test_idx],
            prior_weights=None if weights is None else weights[test_idx],
            family=family,
            residual_type=residual_type,
        )
        used_folds += 1
    coverage = float(np.mean(~np.isnan(oof))) if n else 0.0
    diagnostics = {
        "residual_mode": "cross_fitted",
        "residual_type": residual_type,
        "n_folds_used": used_folds,
        "fold_coverage": coverage,
        "block_preserving": block_index is not None,
        "in_sample": False,
    }
    return oof, diagnostics


def _predict_residuals(
    *,
    fit: GLMResult,
    y: np.ndarray,
    X: np.ndarray,
    offset: np.ndarray | None,
    prior_weights: np.ndarray | None,
    family: str,
    residual_type: str,
) -> np.ndarray:
    fam = _resolve_family(family, y=y, nb_theta=None)
    offset_vec = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float).ravel()
    weights = np.ones(len(y)) if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()
    eta = X @ fit.coefficients
    mu = fam.linkinv(eta + offset_vec)
    if residual_type == "standardized":
        sd = math.sqrt(fit.dispersion) if fit.dispersion > 0 else 0.0
        return (y - mu) / sd if sd > 0 else (y - mu) * 0.0
    if residual_type == "pearson":
        var = np.maximum(fam.variance(mu), _EPS)
        return (y - mu) * np.sqrt(weights) / np.sqrt(var)
    unit_dev = np.maximum(fam.unit_deviance(y, mu), 0.0)
    return np.sign(y - mu) * np.sqrt(weights * unit_dev)
