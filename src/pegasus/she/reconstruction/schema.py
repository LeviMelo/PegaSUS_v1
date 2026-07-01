
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pegasus.core.schemas import DenominatorContract


PopulationTensorMode = Literal["independent_denominator", "sim_informed_denominator"]


@dataclass(frozen=True)
class PopulationObjectiveWeights:
    anchor: float = 10.0
    aging: float = 1.0
    birth: float = 1.0
    death: float = 0.0
    migration: float = 0.1
    # Soft anchor tying each locality-year's total net migration flow to an
    # observed residual (MSD §2.8.7 "open national residual" case); distinct from
    # ``migration`` above, which is only the second-difference smoothness prior.
    migration_total: float = 0.0
    race: float = 0.0
    age_smooth: float = 0.05
    # Soft anchor pulling each cell toward its census-composition-implied value
    # (closure total * census joint share) so intercensal years inherit the census
    # (age,sex,race) STRUCTURE instead of collapsing to a uniform split (MSD §2.8.10).
    composition: float = 0.0


@dataclass(frozen=True)
class PopulationTensorProblem:
    """Typed population objective over (locality, time, age, sex, race)."""

    shape: tuple[int, int, int, int, int]
    anchors: tuple[float | None, ...]
    mode: PopulationTensorMode = "independent_denominator"
    hard_anchor_mask: tuple[bool, ...] | None = None
    births: tuple[float | None, ...] | None = None
    death_rates: tuple[float | None, ...] | None = None
    sim_deaths: tuple[float | None, ...] | None = None
    race_composition_prior: tuple[float | None, ...] | None = None
    # Full-joint (age,sex,race) composition prior: per-cell target population
    # (closure_total * census joint share), None where no census composition or
    # closure exists. Drives intercensal demographic structure (MSD §2.8.10).
    composition_prior: tuple[float | None, ...] | None = None
    closure_totals: tuple[float | None, ...] | None = None
    migration_totals: tuple[float | None, ...] | None = None
    # Observed net-migration total per (locality, time), shape (S, T). ``None`` in a
    # cell = no residual observation for that locality-year (left to smoothness).
    migration_locality_totals: tuple[float | None, ...] | None = None
    migration_bounds: tuple[float, ...] | None = None
    initial_population: tuple[float, ...] | None = None
    initial_migration: tuple[float, ...] | None = None
    weights: PopulationObjectiveWeights = field(default_factory=PopulationObjectiveWeights)

    @property
    def n_cells(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass(frozen=True)
class PopulationSolverTelemetry:
    converged: bool
    iterations: int
    initial_objective: float
    final_objective: float
    projected_gradient_norm: float
    relative_objective_change: float
    step_size: float
    objective_terms: dict[str, float] = field(default_factory=dict)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "projected_gradient_norm": self.projected_gradient_norm,
            "relative_objective_change": self.relative_objective_change,
            "step_size": self.step_size,
            "objective_terms": dict(self.objective_terms),
        }


def official_sidra_anchor_contract(
    *,
    source: str = "SIDRA",
    warnings: list[str] | None = None,
) -> DenominatorContract:
    return DenominatorContract(
        mode="official_sidra_anchor",
        source=source,
        provenance=["official"],
        state="fragile",
        dashboard_safe="warning",
        allowed_for_rates=True,
        warnings=warnings or ["unvalidated_sidra_anchor"],
    )


def blocked_missing_population_contract(
    *,
    reason: str,
) -> DenominatorContract:
    return DenominatorContract(
        mode="blocked_missing",
        source="none",
        provenance=[],
        state="illegal_excluded",
        dashboard_safe=False,
        allowed_for_rates=False,
        warnings=[reason],
    )


@dataclass(frozen=True)
class PopulationTensorRequest:
    mode: PopulationTensorMode
    solver_id: str
    solver_backend: str
    locality_ids: tuple[str, ...]
    periods: tuple[str, ...]
    strata: tuple[str, ...] = ("total",)
    require_sparse: bool = True

    @property
    def n_cells(self) -> int:
        return len(self.locality_ids) * len(self.periods) * len(self.strata)

    def support(self) -> dict[str, Any]:
        return {
            "locality_ids": list(self.locality_ids),
            "periods": list(self.periods),
            "strata": list(self.strata),
            "n_cells": self.n_cells,
        }


@dataclass(frozen=True)
class PopulationTensorDiagnostics:
    n_cells: int
    solver_backend: str
    sparse_jacobian: bool
    reconstruction_uncertainty: float
    denominator_feedback_warning: bool
    dense_abort_threshold: int
    converged: bool = True
    iterations: int = 0
    initial_objective: float = 0.0
    final_objective: float = 0.0
    projected_gradient_norm: float = 0.0
    relative_objective_change: float = 0.0
    objective_terms: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "solver_backend": self.solver_backend,
            "sparse_jacobian": self.sparse_jacobian,
            "reconstruction_uncertainty": self.reconstruction_uncertainty,
            "denominator_feedback_warning": self.denominator_feedback_warning,
            "dense_abort_threshold": self.dense_abort_threshold,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "projected_gradient_norm": self.projected_gradient_norm,
            "relative_objective_change": self.relative_objective_change,
            "objective_terms": dict(self.objective_terms),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PopulationTensorResult:
    tensor_id: str
    mode: PopulationTensorMode
    solver_id: str
    solver_backend: str
    sparse_jacobian: bool
    value: float
    unit: str
    locality_id: str
    period: str
    source_anchor_field_id: str
    source_table_id: str
    source_variable_id: str
    source_request_hash: str
    source_metadata_hash: str
    reconstruction_uncertainty: float
    denominator_feedback_warning: bool
    state: str
    warnings: tuple[str, ...]
    diagnostics: PopulationTensorDiagnostics
    tensor_shape: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1)
    tensor_values: tuple[float, ...] = field(default_factory=tuple)
    migration_values: tuple[float, ...] = field(default_factory=tuple)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "mode": self.mode,
            "PopulationTensorMode": self.mode,
            "solver_backend": self.solver_backend,
            "SolverBackend": self.solver_backend,
            "solver_id": self.solver_id,
            "SolverID": self.solver_id,
            "sparse_jacobian": self.sparse_jacobian,
            "SparseJacobian": self.sparse_jacobian,
            "value": self.value,
            "unit": self.unit,
            "locality_id": self.locality_id,
            "period": self.period,
            "source_anchor_field_id": self.source_anchor_field_id,
            "source_table_id": self.source_table_id,
            "source_variable_id": self.source_variable_id,
            "source_request_hash": self.source_request_hash,
            "source_metadata_hash": self.source_metadata_hash,
            "reconstruction_uncertainty": self.reconstruction_uncertainty,
            "denominator_feedback_warning": self.denominator_feedback_warning,
            "DenominatorFeedbackWarning": self.denominator_feedback_warning,
            "state": self.state,
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics.as_manifest(),
            "tensor_shape": list(self.tensor_shape),
            "tensor_values": list(self.tensor_values),
            "migration_values": list(self.migration_values),
        }
