"""Exact and approximate HSIC kernels with empirical structured nulls."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import median
from typing import Any, Literal

from pegasus.pirs.nulls import generate_null_indices
from pegasus.compute.devices import resolve_torch_device
from pegasus.compute.kernels import tensor_nbytes
from pegasus.compute.torch_backend import torch_runtime
from pegasus.compute.random import torch_generator

HSICMode = Literal["exact", "nystrom", "rff", "disabled", "cuda_unavailable_abort"]


def hsic_formula_contract() -> dict[str, Any]:
    return {
        "inputs": {
            "X": {"shape": "[n, d_x]", "dtype": "float64", "missing": "mutually_observed_only"},
            "E": {"shape": "[n, d_e]", "dtype": "float64", "meaning": "outcome residuals"},
            "support": "ordered support metadata required by the selected null strategy",
        },
        "outputs": {
            "statistic": "biased centered-kernel HSIC or feature cross-covariance norm",
            "p_value": "(1 + null exceedances) / (1 + executed permutations)",
            "diagnostics": "bandwidth, approximation rank/features, seed, device, null moments",
        },
        "epsilon_stabilization": "bandwidth and Nyström eigenvalues clamp at 1e-9 and 1e-8",
        "warnings": ["hsic_disabled_insufficient_support", "hsic_approximation_diagnostics_emitted"],
        "failure_modes": [
            "constant_support",
            "missing_null_support_metadata",
            "cuda_required_unavailable",
            "standard_or_deep_in_sample_residuals",
        ],
    }


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
        return dict(vars(self))


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
        return dict(vars(self))


def select_hsic_mode(
    *,
    n_eff: int | float,
    budget: str,
    user_disabled: bool = False,
    cuda_required: bool = False,
    cuda_available: bool = False,
) -> str:
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


def _bandwidth(values: list[float], *, seed: int) -> float:
    rng = random.Random(seed)
    n = len(values)
    pairs: list[float] = []
    maximum = min(2048, n * (n - 1) // 2)
    if maximum == n * (n - 1) // 2:
        pairs = [abs(values[i] - values[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        for _ in range(maximum):
            i, j = rng.sample(range(n), 2)
            pairs.append(abs(values[i] - values[j]))
    positive = [value for value in pairs if value > 0]
    return max(median(positive) if positive else 1.0, 1e-9)


def _kernel_matrix(torch: Any, values: Any, *, kernel: str, bandwidth: float) -> Any:
    distances = torch.abs(values[:, None] - values[None, :])
    if kernel == "linear":
        return values[:, None] * values[None, :]
    if kernel == "rbf":
        return torch.exp(-(distances**2) / (2.0 * bandwidth**2))
    if kernel == "matern":
        scaled = math.sqrt(3.0) * distances / bandwidth
        return (1.0 + scaled) * torch.exp(-scaled)
    raise ValueError("HSIC kernel must be one of: rbf, linear, matern.")


def _center_kernel(kernel: Any) -> Any:
    return kernel - kernel.mean(dim=0, keepdim=True) - kernel.mean(dim=1, keepdim=True) + kernel.mean()


def _exact_components(torch: Any, x: Any, y: Any, *, kernel: str, bandwidth_x: float, bandwidth_y: float):
    return (
        _center_kernel(_kernel_matrix(torch, x, kernel=kernel, bandwidth=bandwidth_x)),
        _center_kernel(_kernel_matrix(torch, y, kernel=kernel, bandwidth=bandwidth_y)),
    )


def _kernel_hsic(k_centered: Any, l_centered: Any) -> float:
    n = k_centered.shape[0]
    return float(torch_sum(k_centered * l_centered) / max((n - 1) ** 2, 1))


def torch_sum(value: Any) -> Any:
    return value.sum()


def linear_hsic_statistic(x: list[float], residuals: list[float]) -> float:
    if len(x) != len(residuals):
        raise ValueError("HSIC inputs must have identical support length")
    if len(x) < 2:
        return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(residuals) / len(residuals)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, residuals, strict=True)) / (len(x) - 1)
    variance_x = sum((a - mean_x) ** 2 for a in x) / (len(x) - 1)
    variance_y = sum((b - mean_y) ** 2 for b in residuals) / (len(x) - 1)
    if variance_x <= 0 or variance_y <= 0:
        return 0.0
    return max(0.0, covariance * covariance / (variance_x * variance_y))


def _np_rff_features(values: "Any", *, bandwidth: float, features: int, seed: int) -> "Any":
    """Random Fourier features approximating an RBF kernel (NumPy)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    omega = rng.standard_normal(features) / bandwidth
    phase = rng.uniform(0.0, 2.0 * math.pi, features)
    return math.sqrt(2.0 / features) * np.cos(np.outer(v, omega) + phase[None, :])


def _np_nystrom_features(values: "Any", *, bandwidth: float, landmarks: int, seed: int) -> "Any":
    """Nyström RBF feature map from a landmark subset (NumPy)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    m = min(landmarks, v.shape[0])
    idx = rng.choice(v.shape[0], size=m, replace=False)
    sel = v[idx]
    cross = np.exp(-((v[:, None] - sel[None, :]) ** 2) / (2.0 * bandwidth ** 2))
    basis = np.exp(-((sel[:, None] - sel[None, :]) ** 2) / (2.0 * bandwidth ** 2))
    eigenvalues, eigenvectors = np.linalg.eigh(basis)
    inv_root = eigenvectors @ np.diag(1.0 / np.sqrt(np.clip(eigenvalues, 1e-8, None))) @ eigenvectors.T
    return cross @ inv_root


def _np_feature_hsic(x_features: "Any", y_features: "Any") -> float:
    import numpy as np

    n = x_features.shape[0]
    xc = x_features - x_features.mean(axis=0, keepdims=True)
    yc = y_features - y_features.mean(axis=0, keepdims=True)
    cross = xc.T @ yc / max(n - 1, 1)
    return float((cross * cross).sum())


def numpy_kernel_hsic_permutation_test(
    *,
    covariate: list[float],
    residuals: list[float],
    permutations: int,
    seed: int,
    kernel: str = "rbf",
    permutation_indices: list[list[int]] | None = None,
    budget: str = "fast",
    max_exact: int = 5000,
) -> tuple[float, list[float], dict[str, Any]]:
    """Real non-linear centered-kernel HSIC + permutation null, NumPy-only.

    Mode selection (MSD §6.7): exact centered-kernel HSIC for ``n <= max_exact``;
    for larger ``n`` it switches to feature-map HSIC to avoid an n×n matrix —
    Random Fourier Features for fast budget, Nyström for standard/deep — keeping
    the estimator genuinely non-linear at national scale rather than degrading to
    a linear stand-in.

    ``permutation_indices`` supplies structured permutations (e.g. within-block
    cyclic time shifts, MSD §6.8/§6.9); otherwise an i.i.d. null is used.
    """
    import numpy as np

    n = len(covariate)
    if n < 2 or len(residuals) != n:
        return 0.0, [], {"kernel": kernel, "n": n, "degenerate": True}

    bandwidth_x = _bandwidth(covariate, seed=seed)
    bandwidth_y = _bandwidth(residuals, seed=seed + 1)
    null_kind = "structured_provided" if permutation_indices is not None else "iid_permutation"

    def _iter_perms() -> "Any":
        if permutation_indices is not None:
            for perm in permutation_indices:
                perm_arr = np.asarray(perm, dtype=int)
                if perm_arr.shape[0] == n:
                    yield perm_arr
        else:
            rng = np.random.default_rng(seed)
            for _ in range(max(int(permutations), 1)):
                yield rng.permutation(n)

    if n <= max_exact:
        hsic_mode = "exact"

        def _kernel_matrix(values: list[float], bandwidth: float) -> "np.ndarray":
            v = np.asarray(values, dtype=float)
            dist = np.abs(v[:, None] - v[None, :])
            if kernel == "linear":
                return v[:, None] * v[None, :]
            if kernel == "matern":
                scaled = math.sqrt(3.0) * dist / bandwidth
                return (1.0 + scaled) * np.exp(-scaled)
            return np.exp(-(dist ** 2) / (2.0 * bandwidth ** 2))

        def _center(matrix: "np.ndarray") -> "np.ndarray":
            m = matrix.shape[0]
            h = np.eye(m) - np.full((m, m), 1.0 / m)
            return h @ matrix @ h

        k_centered = _center(_kernel_matrix(covariate, bandwidth_x))
        l_centered = _center(_kernel_matrix(residuals, bandwidth_y))
        denom = max((n - 1) ** 2, 1)
        statistic = float(np.sum(k_centered * l_centered) / denom)
        null_statistics = [
            float(np.sum(k_centered * l_centered[np.ix_(perm, perm)]) / denom) for perm in _iter_perms()
        ]
        approximation = None
    else:
        n_features = int(min(max(128, int(math.sqrt(n) * 4)), 1024))
        if budget == "fast":
            hsic_mode = "rff"
            x_feat = _np_rff_features(covariate, bandwidth=bandwidth_x, features=n_features, seed=seed)
            y_feat = _np_rff_features(residuals, bandwidth=bandwidth_y, features=n_features, seed=seed + 1)
        else:
            hsic_mode = "nystrom"
            landmarks = int(min(max(64, int(math.sqrt(n))), 1024))
            x_feat = _np_nystrom_features(covariate, bandwidth=bandwidth_x, landmarks=landmarks, seed=seed)
            y_feat = _np_nystrom_features(residuals, bandwidth=bandwidth_y, landmarks=landmarks, seed=seed + 1)
        statistic = _np_feature_hsic(x_feat, y_feat)
        null_statistics = [_np_feature_hsic(x_feat, y_feat[perm]) for perm in _iter_perms()]
        approximation = hsic_mode

    diagnostics = {
        "kernel": kernel,
        "bandwidth_x": bandwidth_x,
        "bandwidth_y": bandwidth_y,
        "estimator": "biased_centered_kernel_hsic_numpy",
        "hsic_mode": hsic_mode,
        "approximation": approximation,
        "null_kind": null_kind,
        "permutations_executed": len(null_statistics),
        "n": n,
    }
    return statistic, null_statistics, diagnostics


def permutation_p_value(
    *,
    statistic: float,
    permutations: int,
    n_eff: int | float,
    null_statistics: list[float] | None = None,
) -> float:
    if null_statistics is None:
        raise ValueError("Empirical HSIC p-values require null_statistics from actual permutations.")
    exceedances = sum(value >= statistic - 1e-15 for value in null_statistics)
    return (exceedances + 1.0) / (len(null_statistics) + 1.0)


def _rff_features(torch: Any, values: Any, *, bandwidth: float, features: int, seed: int) -> Any:
    generator = torch_generator(torch, seed=seed, device=values.device)
    omega = torch.randn(features, dtype=values.dtype, device=values.device, generator=generator) / bandwidth
    phase = 2.0 * math.pi * torch.rand(features, dtype=values.dtype, device=values.device, generator=generator)
    return math.sqrt(2.0 / features) * torch.cos(values[:, None] * omega[None, :] + phase[None, :])


def _feature_hsic(x_features: Any, y_features: Any) -> float:
    x_centered = x_features - x_features.mean(dim=0, keepdim=True)
    y_centered = y_features - y_features.mean(dim=0, keepdim=True)
    cross = x_centered.T @ y_centered / max(x_features.shape[0] - 1, 1)
    return float((cross * cross).sum())


def _nystrom_features(torch: Any, values: Any, *, bandwidth: float, landmarks: int, seed: int) -> Any:
    generator = torch_generator(torch, seed=seed, device=values.device)
    indices = torch.randperm(values.shape[0], device=values.device, generator=generator)[:landmarks]
    selected = values[indices]
    cross = torch.exp(-((values[:, None] - selected[None, :]) ** 2) / (2.0 * bandwidth**2))
    basis = torch.exp(-((selected[:, None] - selected[None, :]) ** 2) / (2.0 * bandwidth**2))
    eigenvalues, eigenvectors = torch.linalg.eigh(basis)
    inverse_root = eigenvectors @ torch.diag(torch.clamp(eigenvalues, min=1e-8).rsqrt()) @ eigenvectors.T
    return cross @ inverse_root


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
    kernel: str = "rbf",
    mode_override: str | None = None,
) -> HSICOutput:
    if len(residuals) != len(covariate):
        raise ValueError("HSIC inputs must have identical support length")
    paired: list[tuple[float, float]] = []
    for residual, value in zip(residuals, covariate, strict=True):
        try:
            residual_value = float(residual)
            covariate_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(residual_value) and math.isfinite(covariate_value):
            paired.append((residual_value, covariate_value))
    y_values = [pair[0] for pair in paired]
    x_values = [pair[1] for pair in paired]
    n_eff = len(paired)
    residual_mode = residual_mode_for_hsic(budget=budget)
    validate_residual_mode_for_hsic(budget=budget, residual_mode=residual_mode)
    mode = select_hsic_mode(
        n_eff=n_eff,
        budget=budget,
        user_disabled=user_disabled,
        cuda_required=cuda_required,
        cuda_available=cuda_available,
    )
    if mode_override is not None and mode not in {"disabled", "cuda_unavailable_abort"}:
        if mode_override not in {"exact", "nystrom", "rff"}:
            raise ValueError("HSIC mode_override must be exact, nystrom, or rff.")
        mode = mode_override
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {
        "kernel": kernel, "seed": seed, "n_eff": n_eff, "input_rows": len(residuals),
        "dropped_nonfinite_rows": len(residuals) - n_eff, "budget": budget,
        "null_regime_id": null_strategy, "permutations_requested": int(permutations), "fdr_family": fdr_method,
    }
    statistic: float | None = None
    p_value: float | None = None
    constant_residual = n_eff > 0 and len(set(y_values)) <= 1
    constant_covariate = n_eff > 0 and len(set(x_values)) <= 1
    spatial_block_count = len(set(str(value) for value in support_intersection.get("spatial_blocks", [])))
    panel_shape = support_intersection.get("panel_shape", [0, 0])
    temporal_block_count = int(support_intersection.get("temporal_block_count") or (panel_shape[-1] if panel_shape else 0) or 0)
    diagnostics.update({"spatial_block_count": spatial_block_count, "temporal_block_count": temporal_block_count})
    if n_eff == 0:
        mode = "disabled"
        warnings.append("hsic_disabled_no_mutually_observed_finite_rows")
    elif constant_residual:
        mode = "disabled"
        warnings.append("hsic_descriptive_only_constant_residual")
    elif constant_covariate:
        mode = "disabled"
        warnings.append("hsic_descriptive_only_constant_covariate")
    elif null_strategy not in {"unrestricted_permutation", "permutation_linear_centered"} and (
        (spatial_block_count and spatial_block_count < 5) or (temporal_block_count and temporal_block_count < 5)
    ):
        mode = "disabled"
        warnings.append("hsic_descriptive_only_insufficient_null_blocks")
    if mode == "disabled":
        if not warnings:
            warnings.append("hsic_disabled_insufficient_support" if n_eff < 100 else "hsic_disabled_by_user")
        diagnostics["descriptive_only_reason"] = warnings[-1]
        diagnostics["permutations_executed"] = 0
    elif mode == "cuda_unavailable_abort":
        warnings.append("cuda_required_unavailable")
    else:
        task_id = f"pirs_hsic_{mode}"
        representation_width = n_eff if mode == "exact" else min(512, max(128, int(math.sqrt(n_eff) * 4)))
        plan = resolve_torch_device(
            task_id,
            cuda_required=cuda_required,
            prefer_cuda=True,
            seed=seed,
            estimated_bytes=tensor_nbytes((n_eff, representation_width), copies=4),
            cuda_available_override=cuda_available,
        )
        torch, device, dtype = torch_runtime(plan)
        x = torch.tensor(x_values, dtype=dtype, device=device)
        y = torch.tensor(y_values, dtype=dtype, device=device)
        bandwidth_x = _bandwidth(x_values, seed=seed)
        bandwidth_y = _bandwidth(y_values, seed=seed + 1)
        if mode == "exact":
            x_repr, y_repr = _exact_components(
                torch, x, y, kernel=kernel, bandwidth_x=bandwidth_x, bandwidth_y=bandwidth_y
            )
            statistic = _kernel_hsic(x_repr, y_repr)
            representation_kind = "kernel"
            diagnostics["matrix_shape"] = [n_eff, n_eff]
        elif mode == "rff":
            features = min(max(128, int(math.sqrt(n_eff) * 4)), 512)
            x_repr = _rff_features(torch, x, bandwidth=bandwidth_x, features=features, seed=seed)
            y_repr = _rff_features(torch, y, bandwidth=bandwidth_y, features=features, seed=seed + 1)
            statistic = _feature_hsic(x_repr, y_repr)
            representation_kind = "features"
            diagnostics.update({"n_features": features, "approximation": "rff"})
            warnings.append("hsic_approximation_diagnostics_emitted")
        else:
            landmarks = min(max(32, int(math.sqrt(n_eff))), 512, n_eff)
            x_repr = _nystrom_features(torch, x, bandwidth=bandwidth_x, landmarks=landmarks, seed=seed)
            y_repr = _nystrom_features(torch, y, bandwidth=bandwidth_y, landmarks=landmarks, seed=seed + 1)
            statistic = _feature_hsic(x_repr, y_repr)
            representation_kind = "features"
            diagnostics.update({"n_landmarks": landmarks, "approximation": "nystrom"})
            warnings.append("hsic_approximation_diagnostics_emitted")
        rng = random.Random(seed)
        null_statistics = []
        for _ in range(max(int(permutations), 1)):
            indices = generate_null_indices(strategy=null_strategy, n=n_eff, support=support_intersection, rng=rng)
            tensor_indices = torch.tensor(indices, dtype=torch.long, device=device)
            if representation_kind == "kernel":
                null_statistics.append(_kernel_hsic(x_repr, y_repr[tensor_indices][:, tensor_indices]))
            else:
                null_statistics.append(_feature_hsic(x_repr, y_repr[tensor_indices]))
        p_value = permutation_p_value(
            statistic=statistic,
            permutations=permutations,
            n_eff=n_eff,
            null_statistics=null_statistics,
        )
        diagnostics.update(
            {
                "bandwidth_x": bandwidth_x,
                "bandwidth_y": bandwidth_y,
                "permutations_executed": len(null_statistics),
                "null_mean": sum(null_statistics) / len(null_statistics),
                "device": str(device),
                "compute_plan": plan.as_manifest(),
            }
        )
        if residual_mode.startswith("cross_fitted"):
            warnings.append("hsic_consumes_cross_fitted_residuals")
    return HSICOutput(
        hypothesis_id=f"hsic__{outcome_residual_field_id}__{covariate_field_id}",
        outcome_field_id=outcome_residual_field_id,
        covariate_field_id=covariate_field_id,
        residual_field_id=outcome_residual_field_id,
        hsic_mode=mode,
        statistic=statistic,
        p_value=p_value,
        q_value=None,
        null_strategy=null_strategy,
        fdr_method=fdr_method,
        n_eff=float(n_eff),
        approximation_diagnostics=diagnostics,
        warnings=warnings,
        residual_mode=residual_mode,
        support_intersection=support_intersection,
    )
