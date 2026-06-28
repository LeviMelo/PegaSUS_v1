from __future__ import annotations

import numpy as np

from pegasus.compute.glm import fit_glm


def _design(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.ones(n), x])


def test_beta_binomial_family_fits_overdispersed_proportions() -> None:
    y = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    fit = fit_glm(y=y, X=_design(len(y)), family="beta_binomial", prior_weights=np.repeat(20.0, len(y)))
    assert fit.family == "beta_binomial"
    assert fit.residual_type == "pearson"
    assert fit.fitted.shape == y.shape


def test_lognormal_and_two_part_lognormal_fit_skewed_outcomes() -> None:
    y = np.array([1.0, 2.0, 2.5, 4.0, 8.0, 12.0])
    lognormal = fit_glm(y=y, X=_design(len(y)), family="lognormal")
    assert lognormal.family == "lognormal"
    assert np.all(lognormal.fitted > 0)

    y_zero = np.array([0.0, 1.0, 0.0, 3.0, 5.0, 9.0])
    two_part = fit_glm(y=y_zero, X=_design(len(y_zero)), family="two_part_lognormal")
    assert two_part.family == "two_part_lognormal"
    assert two_part.randomized_quantile_residuals is not None


def test_student_t_family_uses_robust_weights() -> None:
    y = np.array([1.0, 1.2, 1.1, 1.3, 25.0, 1.4])
    fit = fit_glm(y=y, X=_design(len(y)), family="student_t")
    assert fit.family == "student_t"
    assert "student_t_robust_weighted_gaussian" in fit.warnings
    assert "robust_weights" in fit.aux

