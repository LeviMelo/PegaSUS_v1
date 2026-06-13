from __future__ import annotations

import math
from typing import Any

from pegasus.she.stdfm.schema import STDFMProblem


ALLOWED_LINKS = {"identity", "log", "proportion_logit"}


def validate_stdfm_problem(problem: STDFMProblem) -> None:
    if len(problem.shape) != 3 or any(dimension <= 0 for dimension in problem.shape):
        raise ValueError("ST-DFM shape must contain positive (space, time, field) dimensions.")
    space, time, fields = problem.shape
    if len(problem.field_ids) != fields or len(problem.link_function_by_field) != fields:
        raise ValueError("ST-DFM field metadata does not match the field dimension.")
    if len(problem.observations) != problem.n_cells or len(problem.observed_mask) != problem.n_cells:
        raise ValueError("ST-DFM observations and masks must match n_cells.")
    if time < 3:
        raise ValueError("ST-DFM requires at least three temporal points.")
    if not 1 <= problem.n_factors <= fields:
        raise ValueError("ST-DFM n_factors must be between one and n_fields.")
    if any(link not in ALLOWED_LINKS for link in problem.link_function_by_field):
        raise ValueError(f"Unsupported ST-DFM link; allowed links are {sorted(ALLOWED_LINKS)}.")
    if problem.spatial_laplacian is not None and len(problem.spatial_laplacian) != space * space:
        raise ValueError("spatial_laplacian must have shape (space, space).")
    if problem.n_covariates < 0:
        raise ValueError("n_covariates must be nonnegative.")
    if problem.covariates is not None and len(problem.covariates) != space * time * problem.n_covariates:
        raise ValueError("covariates must have shape (space, time, n_covariates).")
    if problem.denominator_by_cell is not None and len(problem.denominator_by_cell) != problem.n_cells:
        raise ValueError("denominator_by_cell must match n_cells.")
    if problem.validation_mask is not None and len(problem.validation_mask) != problem.n_cells:
        raise ValueError("validation_mask must match n_cells.")
    if not any(problem.observed_mask):
        raise ValueError("ST-DFM requires at least one observed cell.")
    if problem.multi_starts < 1 or not 0 <= problem.stability_threshold <= 1:
        raise ValueError("ST-DFM multi-start controls are invalid.")
    penalties = (problem.gamma_temporal, problem.gamma_spatial, problem.gamma_transition)
    if any(value < 0 or not math.isfinite(value) for value in penalties):
        raise ValueError("ST-DFM penalties must be finite and nonnegative.")


def transform_observations(problem: STDFMProblem) -> tuple[tuple[float, ...], tuple[str, ...]]:
    validate_stdfm_problem(problem)
    _, _, fields = problem.shape
    transformed: list[float] = []
    warnings: set[str] = set()
    epsilon = 1e-9
    for index, value in enumerate(problem.observations):
        field = index % fields
        link = problem.link_function_by_field[field]
        if not problem.observed_mask[index]:
            transformed.append(0.0)
        elif link == "identity":
            transformed.append(float(value))
        elif link == "log":
            if value < 0:
                raise ValueError("Log-link ST-DFM observations must be nonnegative.")
            transformed.append(math.log(value + epsilon))
        else:
            denominator = problem.denominator_by_cell[index] if problem.denominator_by_cell else None
            if denominator is not None and denominator > 1:
                bounded = (value * (denominator - 1.0) + 0.5) / denominator
            else:
                bounded = (value + epsilon) / (1.0 + 2.0 * epsilon)
                warnings.add("proportion_denominator_unknown")
            bounded = min(max(bounded, epsilon), 1.0 - epsilon)
            transformed.append(math.log(bounded / (1.0 - bounded)))
    return tuple(transformed), tuple(sorted(warnings))


def stdfm_objective_pseudocode_contract() -> dict[str, Any]:
    """Typed formula-to-code contract for the gated ST-DFM objective.

    The production PyTorch implementation consumes this exact contract.
    """
    return {
        "inputs": {
            "Y": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64", "missing": "mask"},
            "M": {"shape": "[n_space, n_time, n_fields]", "dtype": "bool", "meaning": "observed mask"},
            "Z": {"shape": "[n_space, n_time, n_covariates]", "dtype": "float64"},
            "support": "municipality/year support aligned before invocation",
        },
        "outputs": {
            "latent_factor": {"shape": "[n_space, n_time, k]", "dtype": "float64"},
            "reconstruction": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64"},
            "uncertainty": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64"},
        },
        "epsilon_stabilization": "variance and bounded-link denominators clamp at eps=1e-9",
        "warnings": [
            "stdfm_certification_required",
            "blocked_solver_pending",
            "proportion_denominator_unknown",
            "stdfm_factor_instability",
        ],
        "failure_modes": [
            "insufficient_temporal_points",
            "concept_incompatibility",
            "uncertified_verified_promotion",
            "optimizer_nonconvergence",
            "spatial_laplacian_shape_mismatch",
            "cuda_required_unavailable",
        ],
    }
