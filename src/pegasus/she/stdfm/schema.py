from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

STDFMState = Literal[
    "blocked_invalid_support",
    "fitted",
    "failed",
    "failed_certification",
    "fragile",
    "uncertified",
    "verified",
]


@dataclass(frozen=True)
class STDFMProblem:
    """Numerical ST-DFM input on (space, time, field) support."""

    field_ids: tuple[str, ...]
    shape: tuple[int, int, int]
    observations: tuple[float, ...]
    observed_mask: tuple[bool, ...]
    link_function_by_field: tuple[str, ...]
    n_factors: int = 1
    spatial_laplacian: tuple[float, ...] | None = None
    covariates: tuple[float, ...] | None = None
    n_covariates: int = 0
    denominator_by_cell: tuple[float | None, ...] | None = None
    validation_mask: tuple[bool, ...] | None = None
    gamma_temporal: float = 0.1
    gamma_spatial: float = 0.1
    gamma_transition: float = 0.1
    multi_starts: int = 3
    stability_threshold: float = 0.85
    seed: int = 1729

    @property
    def n_cells(self) -> int:
        space, time, fields = self.shape
        return space * time * fields


@dataclass(frozen=True)
class STDFMSolverTelemetry:
    converged: bool
    iterations: int
    initial_objective: float
    final_objective: float
    relative_objective_change: float
    factor_stability: float
    device: str
    dtype: str
    starts_completed: int
    objective_terms: dict[str, float] = field(default_factory=dict)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "relative_objective_change": self.relative_objective_change,
            "factor_stability": self.factor_stability,
            "device": self.device,
            "dtype": self.dtype,
            "starts_completed": self.starts_completed,
            "objective_terms": dict(self.objective_terms),
        }


@dataclass(frozen=True)
class STDFMFitResult:
    output: "STDFMOutputSchema"
    latent_factors: tuple[float, ...]
    loadings: tuple[float, ...]
    transition_matrix: tuple[float, ...]
    reconstructed: tuple[float, ...]
    uncertainty: tuple[float, ...]
    telemetry: STDFMSolverTelemetry
    certification: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class STDFMInputSchema:
    field_id: str
    concept_id: str
    support: dict[str, Any]
    observation_shape: tuple[int, int]
    transform: str
    dynamics: str
    projection_matrix_id: str | None
    stitch_metadata: dict[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "concept_id": self.concept_id,
            "support": self.support,
            "observation_shape": list(self.observation_shape),
            "transform": self.transform,
            "dynamics": self.dynamics,
            "projection_matrix_id": self.projection_matrix_id,
            "stitch_metadata": self.stitch_metadata,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class STDFMOutputSchema:
    field_id: str
    status: STDFMState
    solver_backend: str
    certification_id: str | None
    uncertainty: float | None
    warnings: tuple[str, ...]
    reason: str
    latent_factor_path: str | None = None
    loading_matrix_path: str | None = None
    reconstructed_fields_path: str | None = None
    certification_table_path: str | None = None
    uncertainty_path: str | None = None
    objective_trace_path: str | None = None
    telemetry: STDFMSolverTelemetry | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "status": self.status,
            "solver_backend": self.solver_backend,
            "certification_id": self.certification_id,
            "uncertainty": self.uncertainty,
            "warnings": list(self.warnings),
            "reason": self.reason,
            "latent_factor_path": self.latent_factor_path,
            "loading_matrix_path": self.loading_matrix_path,
            "reconstructed_fields_path": self.reconstructed_fields_path,
            "certification_table_path": self.certification_table_path,
            "uncertainty_path": self.uncertainty_path,
            "objective_trace_path": self.objective_trace_path,
            "telemetry": self.telemetry.as_manifest() if self.telemetry else None,
        }


def build_stdfm_input_schema(
    *,
    field_id: str,
    concept_id: str,
    support: dict[str, Any],
    periods: list[str],
    localities: list[str],
    transform: str,
    dynamics: str,
    projection_matrix_id: str | None,
    stitch_metadata: dict[str, Any],
    warnings: list[str] | tuple[str, ...] | None = None,
) -> STDFMInputSchema:
    return STDFMInputSchema(
        field_id=field_id,
        concept_id=concept_id,
        support=support,
        observation_shape=(len(localities), len(periods)),
        transform=transform,
        dynamics=dynamics,
        projection_matrix_id=projection_matrix_id,
        stitch_metadata=stitch_metadata,
        warnings=tuple(warnings or ()),
    )
