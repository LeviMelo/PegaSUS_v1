"""HSIC residual-scanner contracts and tiny deterministic kernels for Slice 9A."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

HSICMode = Literal["exact", "nystrom", "rff", "disabled", "cuda_unavailable_abort"]


@dataclass(frozen=True)
class HSICInput:
    outcome_residual_field_id: str
    covariate_field_id: str
    support_intersection: dict[str, Any]
    mode: str
    kernel: str
    bandwidth_policy: str
    landmark_policy: str | None
    n_landmarks: int | None
    null_strategy: str
    permutations: int
    seed: int
    residual_mode: str
    budget: str
    cuda_required: bool = False

    def as_manifest(self) -> dict[str, Any]:
        return {
            "outcome_residual_field_id": self.outcome_residual_field_id,
            "covariate_field_id": self.covariate_field_id,
            "support_intersection": self.support_intersection,
            "mode": self.mode,
            "kernel": self.kernel,
            "bandwidth_policy": self.bandwidth_policy,
            "landmark_policy": self.landmark_policy,
            "n_landmarks": self.n_landmarks,
            "null_strategy": self.null_strategy,
            "permutations": self.permutations,
            "seed": self.seed,
            "residual_mode": self.residual_mode,
            "budget": self.budget,
            "cuda_required": self.cuda_required,
        }


@dataclass(frozen=True)
class HSICOutput:
    hypothesis_id: str
    outcome_field_id: str
    covariate_field_id: str
    residual_field_id: str
    hsic_mode: str
    statistic: float | None
    p_value: float | None
    q_value: float | None
    null_strategy: str
    fdr_method: str
    n_eff: float
    approximation_diagnostics: dict[str, Any]
    warnings: list[str]
    residual_mode: str
    support_intersection: dict[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "outcome_field_id": self.outcome_field_id,
            "covariate_field_id": self.covariate_field_id,
            "residual_field_id": self.residual_field_id,
            "hsic_mode": self.hsic_mode,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "null_strategy": self.null_strategy,
            "fdr_method": self.fdr_method,
            "n_eff": self.n_eff,
            "approximation_diagnostics": self.approximation_diagnostics,
            "warnings": list(self.warnings),
            "residual_mode": self.residual_mode,
            "support_intersection": self.support_intersection,
        }


def select_hsic_mode(*, n_eff: int | float, budget: str, user_disabled: bool = False, cuda_required: bool = False, cuda_available: bool = False) -> str:
    if user_disabled:
        return "disabled"
    if cuda_required and not cuda_available:
        return "cuda_unavailable_abort"
    if n_eff < 100:
        return "disabled"
    if n_eff > 5000 and budget in {"standard", "deep"}:
        return "nystrom"
    if n_eff > 5000 and budget == "fast":
        return "rff"
    return "exact"


def residual_mode_for_hsic(*, budget: str) -> str:
    if budget == "fast":
        return "in_sample"
    if budget == "deep":
        return "cross_fitted_parametric_bootstrap"
    return "cross_fitted"


def validate_residual_mode_for_hsic(*, budget: str, residual_mode: str) -> None:
    if budget in {"standard", "deep"} and residual_mode == "in_sample":
        raise ValueError("standard/deep HSIC must not consume in-sample residuals")


def _center(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    return [float(v) - mean for v in values]


def linear_hsic_statistic(x: list[float], residuals: list[float]) -> float:
    if len(x) != len(residuals):
        raise ValueError("HSIC inputs must have identical support length")
    n = len(x)
    if n < 2:
        return 0.0
    xc = _center([float(v) for v in x])
    ec = _center([float(v) for v in residuals])
    cov = sum(a * b for a, b in zip(xc, ec)) / (n - 1)
    vx = sum(a * a for a in xc) / (n - 1)
    ve = sum(b * b for b in ec) / (n - 1)
    if vx <= 0 or ve <= 0:
        return 0.0
    corr = cov / math.sqrt(vx * ve)
    return float(max(0.0, corr * corr))


def permutation_p_value(*, statistic: float, permutations: int, n_eff: int | float) -> float:
    # Deterministic fixture-safe conservative approximation; numerical permutation kernels are Slice 9B+.
    if statistic <= 0:
        return 1.0
    effective_permutations = max(10, int(permutations))
    scaled = statistic * max(1.0, float(n_eff) ** 0.5)
    rank = max(1, int(effective_permutations / (1.0 + scaled)))
    return min(1.0, max(1.0 / effective_permutations, rank / effective_permutations))


def run_hsic_scan(
    *,
    outcome_residual_field_id: str,
    covariate_field_id: str,
    residuals: list[float],
    covariate: list[float],
    support_intersection: dict[str, Any],
    budget: str,
    null_strategy: str,
    fdr_method: str,
    permutations: int,
    seed: int = 20260611,
    user_disabled: bool = False,
    cuda_required: bool = False,
    cuda_available: bool = False,
) -> HSICOutput:
    n_eff = float(min(len(residuals), len(covariate)))
    residual_mode = residual_mode_for_hsic(budget=budget)
    validate_residual_mode_for_hsic(budget=budget, residual_mode="in_sample" if residual_mode == "in_sample" else residual_mode)
    mode = select_hsic_mode(n_eff=n_eff, budget=budget, user_disabled=user_disabled, cuda_required=cuda_required, cuda_available=cuda_available)
    warnings: list[str] = []
    stat: float | None = None
    p_value: float | None = None
    diagnostics: dict[str, Any] = {
        "kernel": "linear_centered_fixture",
        "bandwidth_policy": "not_required_for_linear_fixture",
        "seed": seed,
        "n_eff": n_eff,
        "budget": budget,
    }
    if mode == "disabled":
        warnings.append("hsic_disabled_insufficient_support" if n_eff < 100 else "hsic_disabled_by_user")
    elif mode == "cuda_unavailable_abort":
        warnings.append("cuda_required_unavailable")
        diagnostics["abort_reason"] = "cuda_required_but_unavailable"
    else:
        stat = linear_hsic_statistic(covariate[: int(n_eff)], residuals[: int(n_eff)])
        p_value = permutation_p_value(statistic=stat, permutations=permutations, n_eff=n_eff)
        if mode in {"nystrom", "rff"}:
            warnings.append("hsic_approximation_diagnostics_emitted")
        if residual_mode.startswith("cross_fitted"):
            warnings.append("hsic_consumes_cross_fitted_residuals")
        diagnostics.update({"approximation_mode": mode, "statistic_family": "linear_hsic_fixture"})
    return HSICOutput(
        hypothesis_id=f"hsic__{outcome_residual_field_id}__{covariate_field_id}",
        outcome_field_id=outcome_residual_field_id,
        covariate_field_id=covariate_field_id,
        residual_field_id=outcome_residual_field_id,
        hsic_mode=mode,
        statistic=stat,
        p_value=p_value,
        q_value=None,
        null_strategy=null_strategy,
        fdr_method=fdr_method,
        n_eff=n_eff,
        approximation_diagnostics=diagnostics,
        warnings=warnings,
        residual_mode=residual_mode,
        support_intersection=support_intersection,
    )
