from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any

from pegasus.compute.devices import resolve_torch_device
from pegasus.compute.kernels import tensor_nbytes
from pegasus.compute.random import seed_everything
from pegasus.compute.torch_backend import torch_runtime
from pegasus.core.exceptions import ComputeBackendError
from pegasus.she.stdfm.blocked import blocked_solver_pending
from pegasus.she.stdfm.objective import transform_observations, validate_stdfm_problem
from pegasus.she.stdfm.schema import (
    STDFMFitResult,
    STDFMInputSchema,
    STDFMOutputSchema,
    STDFMProblem,
    STDFMSolverTelemetry,
)


@dataclass
class _Fit:
    factors: Any
    loadings: Any
    transition: Any
    reconstruction: Any
    initial_objective: float
    final_objective: float
    relative_change: float
    iterations: int
    converged: bool
    terms: dict[str, float]


def _inverse_transform(values: list[float], problem: STDFMProblem) -> tuple[float, ...]:
    _, _, fields = problem.shape
    output: list[float] = [0.0] * len(values)
    for i in range(0, len(values), fields):
        group_vals = values[i:i+fields]
        
        clr_exp = [0.0] * fields
        for f_idx in range(fields):
            if problem.link_function_by_field[f_idx] == "clr":
                clr_exp[f_idx] = math.exp(max(min(group_vals[f_idx], 700.0), -700.0))
        
        clr_sum = sum(clr_exp)
        
        for f_idx in range(fields):
            link = problem.link_function_by_field[f_idx]
            val = group_vals[f_idx]
            if link == "identity":
                output[i + f_idx] = val
            elif link == "log":
                output[i + f_idx] = max(math.exp(min(val, 700.0)) - 1e-9, 0.0)
            elif link == "clr":
                output[i + f_idx] = clr_exp[f_idx] / clr_sum if clr_sum > 0 else 0.0
            else:
                val_sig = 1.0 / (1.0 + math.exp(-max(min(val, 700.0), -700.0)))
                denominator = problem.denominator_by_cell[i + f_idx] if problem.denominator_by_cell else None
                
                if denominator is not None and denominator > 1:
                    recovered = (val_sig * denominator - 0.5) / (denominator - 1.0)
                else:
                    recovered = val_sig * (1.0 + 2.0 * 1e-9) - 1e-9
                    
                output[i + f_idx] = max(0.0, min(1.0, recovered))
    return tuple(output)


def _identified_loadings(torch: Any, raw: Any, n_factors: int) -> Any:
    loadings = torch.tril(raw, diagonal=-1)
    diagonal_count = min(raw.shape[0], n_factors)
    diagonal = torch.nn.functional.softplus(torch.diagonal(raw)[:diagonal_count]) + 1e-6
    indices = torch.arange(diagonal_count, device=raw.device)
    loadings[indices, indices] = diagonal
    return loadings


def _fit_start(
    torch: Any,
    problem: STDFMProblem,
    transformed: tuple[float, ...],
    *,
    device: Any,
    dtype: Any,
    seed: int,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
) -> _Fit:
    seed_everything(seed, torch_module=torch)
    space, time, fields = problem.shape
    factors_count = problem.n_factors
    y = torch.tensor(transformed, dtype=dtype, device=device).reshape(space, time, fields)
    observed = torch.tensor(problem.observed_mask, dtype=torch.bool, device=device).reshape(space, time, fields)
    validation = (
        torch.tensor(problem.validation_mask, dtype=torch.bool, device=device).reshape(space, time, fields)
        if problem.validation_mask is not None
        else torch.zeros_like(observed)
    )
    train_mask = observed & ~validation
    if not bool(train_mask.any()):
        raise ValueError("ST-DFM training mask is empty after holdout exclusion.")
    laplacian = (
        torch.tensor(problem.spatial_laplacian, dtype=dtype, device=device).reshape(space, space)
        if problem.spatial_laplacian is not None
        else None
    )
    covariates = (
        torch.tensor(problem.covariates, dtype=dtype, device=device).reshape(space, time, problem.n_covariates)
        if problem.covariates is not None and problem.n_covariates
        else None
    )

    factors = torch.nn.Parameter(torch.randn(space, time, factors_count, dtype=dtype, device=device) * 0.1)
    raw_loadings = torch.nn.Parameter(torch.randn(fields, factors_count, dtype=dtype, device=device) * 0.1)
    transition = torch.nn.Parameter(torch.eye(factors_count, dtype=dtype, device=device) * 0.5)
    covariate_weights = (
        torch.nn.Parameter(torch.zeros(problem.n_covariates, factors_count, dtype=dtype, device=device))
        if covariates is not None
        else None
    )
    parameters = [factors, raw_loadings, transition]
    if covariate_weights is not None:
        parameters.append(covariate_weights)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    initial_objective = math.nan
    previous = math.inf
    relative_change = math.inf
    stable_steps = 0
    terms: dict[str, float] = {}

    for iteration in range(1, max_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loadings = _identified_loadings(torch, raw_loadings, factors_count)
        reconstruction = torch.einsum("stk,qk->stq", factors, loadings)
        data_loss = torch.mean((reconstruction[train_mask] - y[train_mask]) ** 2)
        if time >= 3:
            temporal_loss = torch.mean((factors[:, 2:] - 2.0 * factors[:, 1:-1] + factors[:, :-2]) ** 2)
        else:
            temporal_loss = torch.zeros((), dtype=dtype, device=device)
        if laplacian is not None:
            spatial_loss = torch.einsum("stk,su,utk->", factors, laplacian, factors) / (time * factors_count)
        else:
            spatial_loss = torch.zeros((), dtype=dtype, device=device)
        predicted = torch.einsum("stk,lk->stl", factors[:, :-1], transition)
        if covariates is not None and covariate_weights is not None:
            predicted = predicted + torch.einsum("stc,ck->stk", covariates[:, 1:], covariate_weights)
        transition_loss = torch.mean((factors[:, 1:] - predicted) ** 2)
        covariance = torch.einsum("stk,stl->kl", factors, factors) / (space * time)
        identity_loss = torch.mean((covariance - torch.eye(factors_count, dtype=dtype, device=device)) ** 2)
        total = (
            data_loss
            + problem.gamma_temporal * temporal_loss
            + problem.gamma_spatial * spatial_loss
            + problem.gamma_transition * transition_loss
            + 0.01 * identity_loss
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("ST-DFM objective became non-finite.")
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=100.0)
        optimizer.step()
        objective = float(total.detach().cpu())
        if iteration == 1:
            initial_objective = objective
        relative_change = abs(previous - objective) / max(abs(previous), 1.0) if math.isfinite(previous) else math.inf
        stable_steps = stable_steps + 1 if relative_change <= tolerance else 0
        previous = objective
        if stable_steps >= 10:
            break

    converged = stable_steps >= 10
    with torch.no_grad():
        loadings = _identified_loadings(torch, raw_loadings, factors_count)
        reconstruction = torch.einsum("stk,qk->stq", factors, loadings)
        terms = {
            "data": float(data_loss.detach().cpu()),
            "temporal": float(temporal_loss.detach().cpu()),
            "spatial": float(spatial_loss.detach().cpu()),
            "transition": float(transition_loss.detach().cpu()),
            "identifiability": float(identity_loss.detach().cpu()),
        }
    return _Fit(
        factors=factors.detach(),
        loadings=loadings.detach(),
        transition=transition.detach(),
        reconstruction=reconstruction.detach(),
        initial_objective=initial_objective,
        final_objective=previous,
        relative_change=relative_change,
        iterations=iteration,
        converged=converged,
        terms=terms,
    )


def _align_and_correlate(torch: Any, reference: Any, candidate: Any) -> float:
    ref = reference.reshape(-1, reference.shape[-1])
    other = candidate.reshape(-1, candidate.shape[-1])
    cross = other.T @ ref
    u, _, vh = torch.linalg.svd(cross)
    aligned = other @ (u @ vh)
    ref_flat = ref.flatten()
    aligned_flat = aligned.flatten()
    ref_centered = ref_flat - ref_flat.mean()
    aligned_centered = aligned_flat - aligned_flat.mean()
    denominator = torch.linalg.vector_norm(ref_centered) * torch.linalg.vector_norm(aligned_centered)
    if float(denominator) <= 1e-12:
        return 1.0 if bool(torch.allclose(ref_flat, aligned_flat, atol=1e-6)) else 0.0
    return float(torch.clamp((ref_centered @ aligned_centered) / denominator, -1.0, 1.0).cpu())


def solve_stdfm(
    input_schema: STDFMInputSchema,
    *,
    problem: STDFMProblem | None = None,
    allow_uncertified: bool = False,
    require_cuda: bool = False,
    prefer_cuda: bool = False,
    max_iterations: int = 2_000,
    learning_rate: float = 0.05,
    tolerance: float = 5e-5,
) -> STDFMOutputSchema | STDFMFitResult:
    if problem is None:
        return blocked_solver_pending(
            field_id=input_schema.field_id,
            reason="ST-DFM numerical observations and masks were not supplied.",
        )
    validate_stdfm_problem(problem)
    if input_schema.observation_shape != problem.shape[:2]:
        raise ValueError("ST-DFM input schema support does not match numerical problem support.")
    try:
        plan = resolve_torch_device(
            "stdfm_solver",
            cuda_required=require_cuda,
            prefer_cuda=prefer_cuda,
            seed=problem.seed,
            estimated_bytes=tensor_nbytes(problem.shape, copies=12),
        )
        torch, device, dtype = torch_runtime(plan)
    except ComputeBackendError as exc:
        return blocked_solver_pending(field_id=input_schema.field_id, reason=str(exc))
    transformed, transform_warnings = transform_observations(problem)
    fits = [
        _fit_start(
            torch,
            problem,
            transformed,
            device=device,
            dtype=dtype,
            seed=problem.seed + start,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            tolerance=tolerance,
        )
        for start in range(problem.multi_starts)
    ]
    best = min(fits, key=lambda fit: fit.final_objective)
    correlations = [
        _align_and_correlate(torch, fits[i].factors, fits[j].factors)
        for i in range(len(fits))
        for j in range(i + 1, len(fits))
    ]
    stability = median(correlations) if correlations else 1.0
    warnings = set(input_schema.warnings) | set(transform_warnings)
    if stability < problem.stability_threshold:
        warnings.add("stdfm_factor_instability")
    if not best.converged:
        warnings.add("stdfm_optimizer_nonconvergence")

    reconstructed_transformed = best.reconstruction.detach().cpu().reshape(-1).tolist()
    reconstructed = _inverse_transform(reconstructed_transformed, problem)
    observed = list(problem.observations)
    train_residuals: list[list[float]] = [[] for _ in problem.field_ids]
    for index, is_observed in enumerate(problem.observed_mask):
        held_out = problem.validation_mask[index] if problem.validation_mask else False
        if is_observed and not held_out:
            train_residuals[index % problem.shape[2]].append(observed[index] - reconstructed[index])
    sigma = [
        math.sqrt(sum(value * value for value in residuals) / max(len(residuals), 1)) if residuals else 0.0
        for residuals in train_residuals
    ]
    uncertainty = tuple(sigma[index % problem.shape[2]] for index in range(problem.n_cells))

    has_holdout = bool(problem.validation_mask and any(problem.validation_mask))
    status = "uncertified"
    reason = "fit_completed_without_holdout_certification"
    certification: dict[str, Any] = {
        "status": "uncertified",
        "mape_holdout": None,
        "rmse_holdout": None,
        "residual_variance_ratio": None,
        "boundary_violations": 0,
        "factor_stability": stability,
    }
    if has_holdout:
        validation_indices = [
            i for i, held_out in enumerate(problem.validation_mask or ()) if held_out and problem.observed_mask[i]
        ]
        if not validation_indices:
            raise ValueError("ST-DFM validation mask contains no observed holdout cells.")
        absolute_errors = [abs(reconstructed[i] - observed[i]) for i in validation_indices]
        mape = sum(
            error / max(abs(observed[i]), 1e-9)
            for error, i in zip(absolute_errors, validation_indices, strict=True)
        ) / len(validation_indices)
        rmse = math.sqrt(sum(error * error for error in absolute_errors) / len(absolute_errors))
        observed_mean = sum(observed[i] for i in validation_indices) / len(validation_indices)
        observed_variance = sum((observed[i] - observed_mean) ** 2 for i in validation_indices) / max(
            len(validation_indices), 1
        )
        residual_variance = sum(error * error for error in absolute_errors) / len(absolute_errors)
        variance_ratio = residual_variance / max(observed_variance, 1e-9)
        verified = (
            best.converged
            and stability >= problem.stability_threshold
            and mape <= 0.15
            and variance_ratio <= 0.25
        )
        status = "verified" if verified else "uncertified"
        reason = (
            f"holdout_mape={mape:.6g};residual_variance_ratio={variance_ratio:.6g};"
            f"factor_stability={stability:.6g}"
        )
        certification.update(
            {
                "status": status,
                "mape_holdout": mape,
                "rmse_holdout": rmse,
                "residual_variance_ratio": variance_ratio,
            }
        )
    if status == "uncertified" and not allow_uncertified:
        warnings.add("stdfm_certification_required")

    telemetry = STDFMSolverTelemetry(
        converged=best.converged,
        iterations=best.iterations,
        initial_objective=best.initial_objective,
        final_objective=best.final_objective,
        relative_objective_change=best.relative_change,
        factor_stability=stability,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        starts_completed=len(fits),
        objective_terms=best.terms,
    )
    output = STDFMOutputSchema(
        field_id=input_schema.field_id,
        status=status,
        solver_backend=f"pytorch_{device.type}_adam_stdfm",
        certification_id=f"cert::{input_schema.field_id}" if has_holdout else None,
        uncertainty=sum(sigma) / len(sigma),
        warnings=tuple(sorted(warnings)),
        reason=reason,
        telemetry=telemetry,
    )
    return STDFMFitResult(
        output=output,
        latent_factors=tuple(best.factors.detach().cpu().reshape(-1).tolist()),
        loadings=tuple(best.loadings.detach().cpu().reshape(-1).tolist()),
        transition_matrix=tuple(best.transition.detach().cpu().reshape(-1).tolist()),
        reconstructed=reconstructed,
        uncertainty=uncertainty,
        telemetry=telemetry,
        certification=certification,
    )
