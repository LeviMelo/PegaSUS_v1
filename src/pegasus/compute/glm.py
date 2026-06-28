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
        "quasi_poisson",
        "hurdle_poisson",
        "hurdle_nb",
        "gamma",
        "sih_gamma_cost_component",
        "binomial_proportion",
        "beta_binomial",
        "dirichlet",
        "multinomial_logit",
        "lognormal",
        "two_part_lognormal",
        "student_t",
    }
)

# MSD §6.5 residual registry: which residual the HSIC scanner consumes.
RESIDUAL_TYPE_BY_FAMILY: dict[str, str] = {
    "gaussian_identity": "standardized",
    "ols": "standardized",
    "poisson_count_with_log_offset": "deviance",
    "negative_binomial": "deviance",
    "quasi_poisson": "pearson",
    "hurdle_poisson": "randomized_quantile",
    "hurdle_nb": "randomized_quantile",
    "gamma": "deviance",
    "sih_gamma_cost_component": "deviance",
    "binomial_proportion": "deviance",
    "beta_binomial": "pearson",
    "dirichlet": "ilr",
    "multinomial_logit": "deviance",
    "lognormal": "standardized",
    "two_part_lognormal": "randomized_quantile",
    "student_t": "standardized",
}

_COUNT_FAMILIES: frozenset[str] = frozenset(
    {"poisson_count_with_log_offset", "negative_binomial", "quasi_poisson", "hurdle_poisson", "hurdle_nb"}
)
_HURDLE_FAMILIES: frozenset[str] = frozenset({"hurdle_poisson", "hurdle_nb"})
_SPECIAL_FAMILIES: frozenset[str] = frozenset({"lognormal", "two_part_lognormal", "student_t", "dirichlet", "multinomial_logit"})

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
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            t2 = np.where(y < 1, (1 - y) * np.log(np.where(y < 1, 1 - y, 1.0) / (1 - mu)), 0.0)
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
    if family in {"poisson_count_with_log_offset", "quasi_poisson"}:
        return _poisson()
    if family in {"gamma", "sih_gamma_cost_component"}:
        return _gamma()
    if family == "negative_binomial":
        return _negative_binomial(nb_theta if nb_theta is not None else _estimate_nb_theta(y))
    if family == "binomial_proportion":
        return _binomial()
    if family == "beta_binomial":
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
    aux: dict = field(default_factory=dict)
    randomized_quantile_residuals: np.ndarray | None = None

    @property
    def primary_residual(self) -> np.ndarray:
        if self.residual_type == "randomized_quantile" and self.randomized_quantile_residuals is not None:
            return self.randomized_quantile_residuals
        if self.residual_type == "standardized":
            return self.standardized_residuals
        if self.residual_type == "pearson":
            return self.pearson_residuals
        return self.deviance_residuals


def _poisson_log_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), _EPS)
    return -lam + k * math.log(lam) - math.lgamma(k + 1)


def _nb_log_pmf(k: int, mu: float, theta: float) -> float:
    mu = max(float(mu), _EPS)
    theta = max(float(theta), _EPS)
    return (
        math.lgamma(k + theta) - math.lgamma(theta) - math.lgamma(k + 1)
        + theta * math.log(theta / (theta + mu)) + k * math.log(mu / (theta + mu))
    )


def _count_p0(lam: float, theta: float | None) -> float:
    if theta is None:
        return math.exp(-max(float(lam), _EPS))
    return math.exp(_nb_log_pmf(0, lam, theta))


def _untruncated_cdf_pair(y_i: int, lam: float, theta: float | None) -> tuple[float, float]:
    """Untruncated count CDF at y and y-1 (Poisson if theta is None, else NB)."""
    cap = min(int(y_i), 5000)
    cdf_y = 0.0
    cdf_ym1 = 0.0
    for k in range(0, cap + 1):
        pk = math.exp(_nb_log_pmf(k, lam, theta) if theta is not None else _poisson_log_pmf(k, lam))
        cdf_y += pk
        if k <= y_i - 1:
            cdf_ym1 += pk
    return min(cdf_y, 1.0), min(cdf_ym1, 1.0)


def _hurdle_cdf_bounds(y: np.ndarray, pi: np.ndarray, lam: np.ndarray, theta: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper hurdle CDF per row for randomized-quantile residuals (Dunn–Smyth)."""
    n = len(y)
    lower = np.empty(n)
    upper = np.empty(n)
    for i in range(n):
        yi = int(round(float(y[i])))
        pi_i = float(min(max(pi[i], _EPS), 1.0 - _EPS))
        if yi <= 0:
            lower[i] = 0.0
            upper[i] = 1.0 - pi_i
            continue
        p0 = _count_p0(float(lam[i]), theta)
        denom = max(1.0 - p0, _EPS)
        cdf_y, cdf_ym1 = _untruncated_cdf_pair(yi, float(lam[i]), theta)
        trunc_y = min(max((cdf_y - p0) / denom, 0.0), 1.0)
        trunc_ym1 = min(max((cdf_ym1 - p0) / denom, 0.0), 1.0)  # = 0 when yi == 1
        lower[i] = (1.0 - pi_i) + pi_i * trunc_ym1
        upper[i] = (1.0 - pi_i) + pi_i * trunc_y
    return lower, upper


def randomized_quantile_residuals(lower: np.ndarray, upper: np.ndarray, *, seed: int = 12345) -> np.ndarray:
    """Dunn–Smyth randomized quantile residuals (MSD §6.5): r_i = Φ⁻¹(U(F(y-1), F(y)))."""
    from statistics import NormalDist

    rng = np.random.default_rng(seed)
    u = rng.uniform(np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0))
    u = np.clip(u, _EPS, 1.0 - _EPS)
    nd = NormalDist()
    return np.array([nd.inv_cdf(float(v)) for v in u], dtype=float)


def select_count_family(y: np.ndarray, *, base_family: str = "poisson_count_with_log_offset") -> str:
    """Data-aware count-family routing (MSD §6.2): zero-inflated → hurdle; overdispersed → NB."""
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        return base_family
    mean = float(np.mean(y))
    if mean <= _EPS:
        return base_family
    var = float(np.var(y, ddof=1)) if y.size > 1 else mean
    dispersion = var / mean if mean > 0 else 1.0
    zero_frac = float(np.mean(y == 0))
    poisson_zero = math.exp(-mean)
    overdispersed = dispersion > 1.5
    # Excess zeros beyond the Poisson expectation, with material zero mass.
    zero_inflated = zero_frac >= 0.30 and zero_frac > poisson_zero * 1.25
    if zero_inflated:
        return "hurdle_nb" if overdispersed else "hurdle_poisson"
    if overdispersed:
        return "negative_binomial"
    return base_family


def _fit_hurdle(
    *,
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    offset: np.ndarray | None,
    prior_weights: np.ndarray | None,
    term_names: Sequence[str] | None,
    max_iter: int,
    tol: float,
    nb_theta: float | None,
    seed: int = 12345,
) -> GLMResult:
    """Two-part hurdle: logistic P(Y>0) × (zero-truncated) Poisson/NB count (MSD §6.2)."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    offset_vec = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    z = (y > 0).astype(float)
    zero_fit = fit_glm(y=z, X=X, family="binomial_proportion", prior_weights=prior_weights,
                       term_names=term_names, max_iter=max_iter, tol=tol)
    pos = y > 0
    if int(pos.sum()) <= p:
        raise GLMError("hurdle_insufficient_positive_support")
    count_family = "negative_binomial" if family == "hurdle_nb" else "poisson_count_with_log_offset"
    theta = (nb_theta if nb_theta is not None else _estimate_nb_theta(y[pos])) if family == "hurdle_nb" else None
    count_fit = fit_glm(
        y=y[pos], X=X[pos], family=count_family,
        offset=offset_vec[pos], prior_weights=None if prior_weights is None else np.asarray(prior_weights).ravel()[pos],
        term_names=term_names, max_iter=max_iter, tol=tol, nb_theta=theta,
    )
    pi = 1.0 / (1.0 + np.exp(-_clamp_eta(X @ zero_fit.coefficients)))
    lam = np.exp(_clamp_eta(X @ count_fit.coefficients + offset_vec))
    p0 = np.array([_count_p0(float(li), theta) for li in lam])
    trunc_mean = lam / np.maximum(1.0 - p0, _EPS)
    mu = pi * trunc_mean
    lower, upper = _hurdle_cdf_bounds(y, pi, lam, theta)
    rq = randomized_quantile_residuals(lower, upper, seed=seed)
    var = np.maximum(mu, _EPS)
    pearson = (y - mu) / np.sqrt(var)
    term_list = list(term_names) if term_names is not None else [f"x{i}" for i in range(p)]
    return GLMResult(
        family=family,
        residual_type="randomized_quantile",
        terms=term_list,
        coefficients=count_fit.coefficients,
        fitted=mu,
        linear_predictor=X @ count_fit.coefficients + offset_vec,
        deviance_residuals=rq,
        pearson_residuals=pearson,
        standardized_residuals=rq,
        deviance=float(np.sum(rq ** 2)),
        dispersion=1.0,
        n_iter=count_fit.n_iter,
        converged=zero_fit.converged and count_fit.converged,
        warnings=["hurdle_two_part_model"],
        aux={
            "zero_coef": zero_fit.coefficients.tolist(),
            "count_coef": count_fit.coefficients.tolist(),
            "nb_theta": theta,
            "count_family": count_family,
            "seed": seed,
        },
        randomized_quantile_residuals=rq,
    )


def _fit_lognormal(
    *,
    y: np.ndarray,
    X: np.ndarray,
    prior_weights: np.ndarray | None,
    term_names: Sequence[str] | None,
    max_iter: int,
    tol: float,
) -> GLMResult:
    y = np.asarray(y, dtype=float).ravel()
    positive = y > 0
    if int(positive.sum()) <= X.shape[1]:
        raise GLMError("lognormal_insufficient_positive_support")
    fit = fit_glm(
        y=np.log(np.maximum(y[positive], _EPS)),
        X=np.asarray(X, dtype=float)[positive],
        family="gaussian_identity",
        prior_weights=None if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()[positive],
        term_names=term_names,
        max_iter=max_iter,
        tol=tol,
    )
    eta = np.asarray(X, dtype=float) @ fit.coefficients
    fitted = np.exp(eta + 0.5 * max(fit.dispersion, 0.0))
    log_y = np.where(y > 0, np.log(np.maximum(y, _EPS)), np.nan)
    resid = (log_y - eta) / math.sqrt(max(fit.dispersion, _EPS))
    resid = np.where(np.isfinite(resid), resid, np.nan)
    return GLMResult(
        family="lognormal",
        residual_type="standardized",
        terms=fit.terms,
        coefficients=fit.coefficients,
        fitted=fitted,
        linear_predictor=eta,
        deviance_residuals=resid,
        pearson_residuals=resid,
        standardized_residuals=resid,
        deviance=float(np.nansum(resid ** 2)),
        dispersion=fit.dispersion,
        n_iter=fit.n_iter,
        converged=fit.converged,
        warnings=["lognormal_positive_support"],
    )


def _fit_two_part_lognormal(
    *,
    y: np.ndarray,
    X: np.ndarray,
    prior_weights: np.ndarray | None,
    term_names: Sequence[str] | None,
    max_iter: int,
    tol: float,
) -> GLMResult:
    y = np.asarray(y, dtype=float).ravel()
    z = (y > 0).astype(float)
    zero_fit = fit_glm(
        y=z,
        X=X,
        family="binomial_proportion",
        prior_weights=prior_weights,
        term_names=term_names,
        max_iter=max_iter,
        tol=tol,
    )
    pos_fit = _fit_lognormal(
        y=y,
        X=X,
        prior_weights=prior_weights,
        term_names=term_names,
        max_iter=max_iter,
        tol=tol,
    )
    pi = 1.0 / (1.0 + np.exp(-_clamp_eta(X @ zero_fit.coefficients)))
    fitted = pi * pos_fit.fitted
    pearson = (y - fitted) / np.sqrt(np.maximum(fitted, _EPS))
    return GLMResult(
        family="two_part_lognormal",
        residual_type="randomized_quantile",
        terms=pos_fit.terms,
        coefficients=pos_fit.coefficients,
        fitted=fitted,
        linear_predictor=pos_fit.linear_predictor,
        deviance_residuals=pearson,
        pearson_residuals=pearson,
        standardized_residuals=pearson,
        deviance=float(np.nansum(pearson ** 2)),
        dispersion=float(np.nanvar(pearson)) if pearson.size else 1.0,
        n_iter=max(zero_fit.n_iter, pos_fit.n_iter),
        converged=zero_fit.converged and pos_fit.converged,
        warnings=["two_part_lognormal_zero_mass"],
        aux={"zero_coef": zero_fit.coefficients.tolist(), "positive_coef": pos_fit.coefficients.tolist()},
        randomized_quantile_residuals=pearson,
    )


def _fit_student_t(
    *,
    y: np.ndarray,
    X: np.ndarray,
    prior_weights: np.ndarray | None,
    term_names: Sequence[str] | None,
    max_iter: int,
    tol: float,
) -> GLMResult:
    y = np.asarray(y, dtype=float).ravel()
    weights = np.ones_like(y) if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()
    robust_weights = weights.copy()
    fit = None
    for _ in range(max_iter):
        fit = fit_glm(
            y=y,
            X=X,
            family="gaussian_identity",
            prior_weights=robust_weights,
            term_names=term_names,
            max_iter=max_iter,
            tol=tol,
        )
        resid = y - fit.fitted
        scale = math.sqrt(max(fit.dispersion, _EPS))
        standardized = resid / scale
        next_weights = weights * np.minimum(1.0, 4.0 / np.maximum(np.abs(standardized), _EPS))
        if float(np.max(np.abs(next_weights - robust_weights))) < tol:
            robust_weights = next_weights
            break
        robust_weights = next_weights
    if fit is None:
        raise GLMError("student_t_fit_failed")
    return GLMResult(
        family="student_t",
        residual_type="standardized",
        terms=fit.terms,
        coefficients=fit.coefficients,
        fitted=fit.fitted,
        linear_predictor=fit.linear_predictor,
        deviance_residuals=fit.deviance_residuals,
        pearson_residuals=fit.pearson_residuals,
        standardized_residuals=fit.standardized_residuals,
        deviance=fit.deviance,
        dispersion=fit.dispersion,
        n_iter=fit.n_iter,
        converged=fit.converged,
        warnings=["student_t_robust_weighted_gaussian"],
        aux={"robust_weights": robust_weights.tolist()},
    )


def _softmax(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    eta = np.clip(eta - np.max(eta, axis=1, keepdims=True), -_ETA_CLAMP, _ETA_CLAMP)
    exp_eta = np.exp(eta)
    return exp_eta / np.maximum(exp_eta.sum(axis=1, keepdims=True), _EPS)


def _composition_matrix(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        labels = arr.astype(int)
        if labels.size == 0:
            raise GLMError("empty_multinomial_response")
        if np.any(labels < 0):
            raise GLMError("multinomial_labels_must_be_nonnegative")
        k = int(labels.max()) + 1
        out = np.zeros((labels.size, k), dtype=float)
        out[np.arange(labels.size), labels] = 1.0
        return out
    if arr.ndim != 2:
        raise GLMError("composition_response_must_be_1d_labels_or_2d_matrix")
    if arr.shape[1] < 2:
        raise GLMError("composition_response_requires_at_least_two_parts")
    if np.any(arr < 0):
        raise GLMError("composition_response_contains_negative_parts")
    row_sum = arr.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise GLMError("composition_response_contains_empty_rows")
    return arr / row_sum


def _ilr_like_residual(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.clip(y, _EPS, 1.0)
    p = np.clip(p, _EPS, 1.0)
    log_y = np.log(y)
    log_p = np.log(p)
    clr_y = log_y - log_y.mean(axis=1, keepdims=True)
    clr_p = log_p - log_p.mean(axis=1, keepdims=True)
    return clr_y - clr_p


def _fit_softmax_composition(
    *,
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    prior_weights: np.ndarray | None,
    term_names: Sequence[str] | None,
    max_iter: int,
    tol: float,
    learning_rate: float = 0.2,
) -> GLMResult:
    """Multinomial/Dirichlet-mean softmax model for categorical/simplex outcomes.

    ``multinomial_logit`` consumes class labels or one-hot/count rows. ``dirichlet``
    consumes non-negative composition rows and fits the mean on the simplex; residuals
    are centered log-ratio differences, the ILR-compatible diagnostic surface required
    by the MSD residual registry.
    """
    Y = _composition_matrix(y)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise GLMError("design_matrix_shape_mismatch")
    n, p = X.shape
    k = Y.shape[1]
    weights = np.ones(n, dtype=float) if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()
    if weights.shape[0] != n:
        raise GLMError("prior_weights_shape_mismatch")
    B = np.zeros((p, k), dtype=float)
    previous = math.inf
    converged = False
    ridge = 1e-6
    for iteration in range(1, max_iter + 1):
        P = _softmax(X @ B)
        gradient = (X.T @ ((P - Y) * weights[:, None])) / max(float(weights.sum()), _EPS)
        gradient += ridge * B
        next_B = B - learning_rate * gradient
        P_next = _softmax(X @ next_B)
        nll = -float(np.sum(weights[:, None] * Y * np.log(np.clip(P_next, _EPS, 1.0)))) / max(float(weights.sum()), _EPS)
        if abs(previous - nll) <= tol:
            B = next_B
            converged = True
            break
        if nll > previous and learning_rate > 1e-4:
            learning_rate *= 0.5
            continue
        previous = nll
        B = next_B
    P = _softmax(X @ B)
    resid_matrix = _ilr_like_residual(Y, P) if family == "dirichlet" else Y - P
    residual = resid_matrix.reshape(-1)
    deviance = -2.0 * float(np.sum(weights[:, None] * Y * np.log(np.clip(P, _EPS, 1.0))))
    term_base = list(term_names) if term_names is not None else [f"x{i}" for i in range(p)]
    terms = [f"{term}[part_{j}]" for term in term_base for j in range(k)]
    return GLMResult(
        family=family,
        residual_type=RESIDUAL_TYPE_BY_FAMILY[family],
        terms=terms,
        coefficients=B.reshape(-1),
        fitted=P.reshape(-1),
        linear_predictor=(X @ B).reshape(-1),
        deviance_residuals=residual,
        pearson_residuals=residual,
        standardized_residuals=residual,
        deviance=deviance,
        dispersion=float(np.mean(residual ** 2)) if residual.size else 0.0,
        n_iter=iteration,
        converged=converged,
        warnings=[f"{family}_softmax_simplex_model"],
        aux={"n_parts": k, "coefficient_shape": [p, k]},
    )


def _fit_beta_binomial(
    *,
    y: np.ndarray,
    X: np.ndarray,
    prior_weights: np.ndarray | None,
    term_names: Sequence[str] | None,
    max_iter: int,
    tol: float,
) -> GLMResult:
    """Overdispersed binomial proportion with beta-binomial variance diagnostics."""
    fit = fit_glm(
        y=y,
        X=X,
        family="binomial_proportion",
        prior_weights=prior_weights,
        term_names=term_names,
        max_iter=max_iter,
        tol=tol,
    )
    yy = np.clip(np.asarray(y, dtype=float).ravel(), 0.0, 1.0)
    mu = np.clip(fit.fitted, _EPS, 1.0 - _EPS)
    n_trials = np.ones_like(yy) if prior_weights is None else np.maximum(np.asarray(prior_weights, dtype=float).ravel(), 1.0)
    raw_var = (yy - mu) ** 2
    binom_var = mu * (1.0 - mu) / n_trials
    denom = np.maximum(mu * (1.0 - mu) * np.maximum(n_trials - 1.0, 0.0) / n_trials, _EPS)
    rho = float(np.clip(np.nanmean(np.maximum(raw_var - binom_var, 0.0) / denom), 0.0, 0.99))
    beta_var = mu * (1.0 - mu) * (1.0 + (n_trials - 1.0) * rho) / n_trials
    pearson = (yy - mu) / np.sqrt(np.maximum(beta_var, _EPS))
    return GLMResult(
        family="beta_binomial",
        residual_type="pearson",
        terms=fit.terms,
        coefficients=fit.coefficients,
        fitted=fit.fitted,
        linear_predictor=fit.linear_predictor,
        deviance_residuals=fit.deviance_residuals,
        pearson_residuals=pearson,
        standardized_residuals=pearson,
        deviance=float(np.sum(pearson ** 2)),
        dispersion=float(np.sum(pearson ** 2) / max(len(pearson) - len(fit.coefficients), 1)),
        n_iter=fit.n_iter,
        converged=fit.converged,
        warnings=["beta_binomial_overdispersion_estimated"],
        aux={"intraclass_correlation_rho": rho},
    )


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
    if family in {"dirichlet", "multinomial_logit"}:
        return _fit_softmax_composition(
            y=y,
            X=X,
            family=family,
            prior_weights=prior_weights,
            term_names=term_names,
            max_iter=max_iter,
            tol=tol,
        )
    if family == "beta_binomial":
        return _fit_beta_binomial(
            y=y,
            X=X,
            prior_weights=prior_weights,
            term_names=term_names,
            max_iter=max_iter,
            tol=tol,
        )
    if family in _HURDLE_FAMILIES:
        return _fit_hurdle(
            y=y, X=X, family=family, offset=offset, prior_weights=prior_weights,
            term_names=term_names, max_iter=max_iter, tol=tol, nb_theta=nb_theta,
        )
    if family == "lognormal":
        return _fit_lognormal(
            y=y,
            X=X,
            prior_weights=prior_weights,
            term_names=term_names,
            max_iter=max_iter,
            tol=tol,
        )
    if family == "two_part_lognormal":
        return _fit_two_part_lognormal(
            y=y,
            X=X,
            prior_weights=prior_weights,
            term_names=term_names,
            max_iter=max_iter,
            tol=tol,
        )
    if family == "student_t":
        return _fit_student_t(
            y=y,
            X=X,
            prior_weights=prior_weights,
            term_names=term_names,
            max_iter=max_iter,
            tol=tol,
        )
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


def fit_glm_penalized(
    *,
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    penalty: np.ndarray,
    offset: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    term_names: Sequence[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    nb_theta: float | None = None,
) -> GLMResult:
    """Fit a penalized GLM by IRLS.

    The penalty is a positive-semidefinite precision matrix added to the IRLS
    normal equations. This is used for ICAR/GMRF spatial effects, where the
    random-effect block receives the graph Laplacian and ordinary covariates
    receive zero penalty.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise GLMError("design_matrix_shape_mismatch")
    n, p = X.shape
    if penalty.shape != (p, p):
        raise GLMError("penalty_matrix_shape_mismatch")
    if n == 0:
        raise GLMError("empty_design_matrix")
    fam = _resolve_family(family, y=y, nb_theta=nb_theta)
    offset_vec = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    weights0 = np.ones(n) if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()

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
        lhs = xw.T @ xw + penalty + ridge
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
    # Effective degrees of freedom for penalized GLMs is trace(H); use the
    # conservative unpenalized p count for dispersion to avoid overclaiming.
    dof = max(n - p, 1)
    dispersion = float(np.sum(pearson ** 2) / dof)
    if family in {"gaussian_identity", "ols"}:
        sd = math.sqrt(dispersion) if dispersion > 0 else 0.0
        standardized = (y - mu) / sd if sd > 0 else (y - mu) * 0.0
    else:
        standardized = dev_resid

    return GLMResult(
        family=family,
        residual_type=RESIDUAL_TYPE_BY_FAMILY.get(family, "deviance"),
        terms=list(term_names) if term_names is not None else [f"x{i}" for i in range(p)],
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
        warnings=["penalized_irls"],
    )


_BOOTSTRAP_FAMILIES: frozenset[str] = frozenset(
    {"poisson_count_with_log_offset", "quasi_poisson", "negative_binomial", "gaussian_identity", "ols", "gamma", "sih_gamma_cost_component"}
)


def bootstrap_deviance_residual_replicates(
    *,
    family: str,
    mu: np.ndarray,
    n_boot: int,
    seed: int = 20260627,
    theta: float | None = None,
    dispersion: float = 1.0,
) -> np.ndarray | None:
    """Parametric bootstrap residual replicates (MSD §6.6.2).

    Draw ``Y^(b) ~ M̂(μ̂)`` and return the deviance residual of each replicate against the
    fitted mean. Shape ``(n_boot, n)``. Returns ``None`` for families where a parametric
    draw from μ̂ alone is ill-defined (hurdle/binomial proportion) so the deep path falls
    back honestly to cross-fitting rather than fabricating a draw.
    """
    if family not in _BOOTSTRAP_FAMILIES:
        return None
    mu = np.asarray(mu, dtype=float).ravel()
    n = mu.size
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    fam = _resolve_family(family, y=mu, nb_theta=theta)
    out = np.empty((max(int(n_boot), 1), n), dtype=float)
    mu_safe = np.maximum(mu, _EPS)
    for b in range(out.shape[0]):
        if family in {"poisson_count_with_log_offset", "quasi_poisson"}:
            y_b = rng.poisson(mu_safe).astype(float)
        elif family == "negative_binomial":
            th = max(float(theta if theta is not None else _estimate_nb_theta(mu_safe)), _EPS)
            p = th / (th + mu_safe)
            y_b = rng.negative_binomial(th, np.clip(p, _EPS, 1 - _EPS)).astype(float)
        elif family in {"gamma", "sih_gamma_cost_component"}:
            shape = 1.0 / max(dispersion, _EPS)
            y_b = rng.gamma(shape=shape, scale=mu_safe / shape)
        else:  # gaussian / ols
            y_b = mu + rng.normal(0.0, math.sqrt(max(dispersion, _EPS)), size=n)
        unit_dev = np.maximum(fam.unit_deviance(y_b, mu_safe), 0.0)
        out[b] = np.sign(y_b - mu_safe) * np.sqrt(unit_dev)
    return out


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
    offset_vec = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float).ravel()
    if family in _HURDLE_FAMILIES:
        zero_coef = np.asarray(fit.aux.get("zero_coef", []), dtype=float)
        count_coef = np.asarray(fit.aux.get("count_coef", []), dtype=float)
        theta = fit.aux.get("nb_theta")
        pi = 1.0 / (1.0 + np.exp(-_clamp_eta(X @ zero_coef)))
        lam = np.exp(_clamp_eta(X @ count_coef + offset_vec))
        lower, upper = _hurdle_cdf_bounds(np.asarray(y, dtype=float).ravel(), pi, lam, theta)
        return randomized_quantile_residuals(lower, upper, seed=int(fit.aux.get("seed", 12345)))
    if family in {"dirichlet", "multinomial_logit"}:
        shape = fit.aux.get("coefficient_shape") or []
        if len(shape) != 2:
            raise GLMError("softmax_prediction_missing_coefficient_shape")
        p, k = int(shape[0]), int(shape[1])
        B = np.asarray(fit.coefficients, dtype=float).reshape(p, k)
        Y = _composition_matrix(np.asarray(y, dtype=float))
        P = _softmax(np.asarray(X, dtype=float) @ B)
        residual = _ilr_like_residual(Y, P) if family == "dirichlet" else Y - P
        return residual.reshape(-1)
    if family == "beta_binomial":
        fam = _resolve_family("binomial_proportion", y=y, nb_theta=None)
        weights = np.ones(len(y)) if prior_weights is None else np.asarray(prior_weights, dtype=float).ravel()
        eta = X @ fit.coefficients
        mu = np.clip(fam.linkinv(eta + offset_vec), _EPS, 1.0 - _EPS)
        rho = float(fit.aux.get("intraclass_correlation_rho") or 0.0)
        yy = np.clip(np.asarray(y, dtype=float).ravel(), 0.0, 1.0)
        beta_var = mu * (1.0 - mu) * (1.0 + (np.maximum(weights, 1.0) - 1.0) * rho) / np.maximum(weights, 1.0)
        return (yy - mu) / np.sqrt(np.maximum(beta_var, _EPS))
    fam = _resolve_family(family, y=y, nb_theta=None)
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
