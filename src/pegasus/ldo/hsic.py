"""Exact and approximate HSIC kernels with empirical structured nulls.

This module owns the ONE HSIC kernel implementation: bandwidth (``_bandwidth``), the RFF/Nyström
feature maps, the doubly-centered kernel, and the statistic + permutation null. The public API is
``build_hsic_representation`` (one variable's reusable representation) + ``hsic_pair_stat_and_null``
(statistic + null for a pair), with ``hsic_pair_stat_and_null_gpu`` a float32 feature-map GPU
variant. Both the reference scanner (``run_hsic_scan`` / ``numpy_kernel_hsic_permutation_test``)
and the LIVE LDO residual scan (:mod:`pegasus.ldo.residual_scan`) route through these — the
residual scan adds only its own layer (panel-aware structured nulls §6.8, per-variable caching,
§II.7 coarsening) on top, without re-deriving any kernel math.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import median
from typing import Any, Literal

from pegasus.ldo.nulls import generate_null_indices
from pegasus.compute.devices import resolve_torch_device
from pegasus.compute.kernels import tensor_nbytes
from pegasus.compute.torch_backend import torch_runtime
from pegasus.compute.random import torch_generator

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
        # Full pairwise enumeration (order-independent median): vectorize the O(n^2)
        # upper-triangle abs-differences in numpy instead of a Python double comprehension.
        import numpy as _np

        arr = _np.asarray(values, dtype=_np.float64)
        diff = _np.abs(arr[:, None] - arr[None, :])
        iu = _np.triu_indices(n, k=1)
        pairs = diff[iu].tolist()
    else:
        # Sampled branch: the median depends on the exact rng.sample sequence (reproducibility
        # of HSIC statistics), so the draw order is preserved verbatim.
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


# --- Single public HSIC representation + statistic API (the ONE kernel implementation) -------
# Both the reference scanner (numpy_kernel_hsic_permutation_test) and the LIVE LDO residual scan
# (pegasus.ldo.residual_scan) route through these. A "representation" holds a variable's HSIC
# feature once so a pair is a cheap product-sum; the x-side uses `seed`, the y-side `seed+1`
# (the two bandwidth/feature regimes the pairwise call consumes). Modes: exact (doubly-centered
# n×n kernel), rff / nystrom (mean-centered n×D feature map, O(N·D)).


def hsic_features_dim(*, n: int, budget: str) -> int:
    if budget == "fast":
        return int(min(max(128, int(math.sqrt(n) * 4)), 1024))
    return int(min(max(64, int(math.sqrt(n))), 1024))


def hsic_mode_for_n(*, n: int, budget: str, max_exact: int = 5000) -> str:
    if n <= max_exact:
        return "exact"
    return "rff" if budget == "fast" else "nystrom"


def _centered_kernel_np(v: "Any", *, bandwidth: float, kernel: str) -> "Any":
    import numpy as np

    v = np.asarray(v, dtype=float)
    dist = np.abs(v[:, None] - v[None, :])
    if kernel == "linear":
        k = v[:, None] * v[None, :]
    elif kernel == "matern":
        scaled = math.sqrt(3.0) * dist / bandwidth
        k = (1.0 + scaled) * np.exp(-scaled)
    else:
        k = np.exp(-(dist ** 2) / (2.0 * bandwidth ** 2))
    return k - k.mean(axis=0, keepdims=True) - k.mean(axis=1, keepdims=True) + k.mean()


def build_hsic_representation(
    values: "Any", *, mode: str, kernel: str, seed: int, budget: str, n: int
) -> dict:
    """One variable's HSIC representation (x-side=seed, y-side=seed+1)."""
    import numpy as np

    row = np.asarray(values, dtype=float)
    bw_x = _bandwidth(row.tolist(), seed=seed)
    bw_y = _bandwidth(row.tolist(), seed=seed + 1)
    if mode == "exact":
        return {
            "mode": "exact",
            "kx": _centered_kernel_np(row, bandwidth=bw_x, kernel=kernel),
            "ky": _centered_kernel_np(row, bandwidth=bw_y, kernel=kernel),
        }
    if mode == "rff":
        fx = _np_rff_features(row, bandwidth=bw_x, features=hsic_features_dim(n=n, budget="fast"), seed=seed)
        fy = _np_rff_features(row, bandwidth=bw_y, features=hsic_features_dim(n=n, budget="fast"), seed=seed + 1)
    else:
        landmarks = hsic_features_dim(n=n, budget="standard")
        fx = _np_nystrom_features(row, bandwidth=bw_x, landmarks=landmarks, seed=seed)
        fy = _np_nystrom_features(row, bandwidth=bw_y, landmarks=landmarks, seed=seed + 1)
    return {
        "mode": mode,
        "fx": fx - fx.mean(axis=0, keepdims=True),
        "fy": fy - fy.mean(axis=0, keepdims=True),
    }


def hsic_stat_from_reprs(ri: dict, rj: dict, *, n: int) -> float:
    if ri["mode"] == "exact":
        return float((ri["kx"] * rj["ky"]).sum() / max((n - 1) ** 2, 1))
    cross = ri["fx"].T @ rj["fy"] / max(n - 1, 1)
    return float((cross * cross).sum())


def hsic_pair_stat_and_null(
    ri: dict, rj: dict, *, n: int, perms: "Any | None"
) -> tuple[float, "Any"]:
    """CPU/numpy reference: HSIC statistic + permutation null for pair (i as x, j as y)."""
    import numpy as np

    if ri["mode"] == "exact":
        kx, ky = ri["kx"], rj["ky"]
        denom = max((n - 1) ** 2, 1)
        stat = float((kx * ky).sum() / denom)
        if perms is None:
            return stat, np.empty(0)
        null = np.array([float((kx * ky[np.ix_(pm, pm)]).sum() / denom) for pm in perms])
        return stat, null
    fx, fy = ri["fx"], rj["fy"]
    denom = max(n - 1, 1)
    stat = float(((fx.T @ fy / denom) ** 2).sum())
    if perms is None:
        return stat, np.empty(0)
    null = np.empty(len(perms), dtype=np.float64)
    for k, pm in enumerate(perms):
        null[k] = float(((fx.T @ fy[pm] / denom) ** 2).sum())
    return stat, null


def hsic_pair_stat_and_null_gpu(
    fx: "Any", fy: "Any", *, n: int, perms: "Any", seed: int
) -> tuple[float, "Any"]:
    """Float32 feature-map HSIC + permutation null on the GPU (O(N·D), no dense N×N kernel).

    ``fx``/``fy`` are the same mean-centered feature maps the CPU path uses; ``perms`` the same
    permutation index arrays, so the statistic matches within float32 tolerance and the p-value
    (exceedance count over the shared perms) is identical."""
    import numpy as np

    plan = resolve_torch_device(
        "pirs_hsic_gpu_features",
        prefer_cuda=True,
        seed=seed,
        estimated_bytes=tensor_nbytes((n, fx.shape[1]), dtype="float32", copies=4),
    )
    torch, device, _ = torch_runtime(plan)
    xf = torch.as_tensor(np.ascontiguousarray(fx), dtype=torch.float32, device=device)
    yf = torch.as_tensor(np.ascontiguousarray(fy), dtype=torch.float32, device=device)
    denom = float(max(n - 1, 1))
    stat = float(((xf.t() @ yf / denom) ** 2).sum().item())
    perm_t = torch.as_tensor(np.asarray(perms, dtype=np.int64), device=device)  # (P, n)
    null = torch.empty(perm_t.shape[0], dtype=torch.float32, device=device)
    for k in range(perm_t.shape[0]):
        cross = xf.t() @ yf.index_select(0, perm_t[k]) / denom
        null[k] = (cross * cross).sum()
    return stat, null.cpu().numpy().astype(np.float64)


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

    hsic_mode = hsic_mode_for_n(n=n, budget=budget, max_exact=max_exact)
    # Build the two variables' representations through the shared public builder (x-side=seed,
    # y-side=seed+1), then reuse them for the statistic and every permutation.
    x_repr = build_hsic_representation(covariate, mode=hsic_mode, kernel=kernel, seed=seed, budget=budget, n=n)
    y_repr = build_hsic_representation(residuals, mode=hsic_mode, kernel=kernel, seed=seed, budget=budget, n=n)

    if permutation_indices is not None:
        perms = [np.asarray(p, dtype=int) for p in permutation_indices if np.asarray(p).shape[0] == n]
        null_kind = "structured_provided"
    else:
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(n) for _ in range(max(int(permutations), 1))]
        null_kind = "iid_permutation"

    statistic, null_arr = hsic_pair_stat_and_null(x_repr, y_repr, n=n, perms=perms)
    null_statistics = null_arr.tolist()
    diagnostics = {
        "kernel": kernel,
        "bandwidth_x": _bandwidth(covariate, seed=seed),
        "bandwidth_y": _bandwidth(residuals, seed=seed + 1),
        "estimator": "biased_centered_kernel_hsic_numpy",
        "hsic_mode": hsic_mode,
        "approximation": None if hsic_mode == "exact" else hsic_mode,
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
        import numpy as _np

        # Build the value tensors through a single numpy buffer rather than element-wise
        # from a Python list: for national n this avoids O(n) Python float boxing on each
        # of the two host->device copies.
        x = torch.as_tensor(_np.asarray(x_values, dtype=_np.float64), dtype=dtype, device=device)
        y = torch.as_tensor(_np.asarray(y_values, dtype=_np.float64), dtype=dtype, device=device)
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
            # Marshal the CPU-generated permutation through a numpy int64 buffer (one contiguous
            # H2D copy) instead of a Python list (element-wise boxing) per permutation.
            tensor_indices = torch.as_tensor(_np.asarray(indices, dtype=_np.int64), device=device)
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
