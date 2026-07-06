
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pegasus.core.schemas import DenominatorContract


PopulationTensorMode = Literal["independent_denominator", "sim_informed_denominator"]


def _to_f64_array(value: Any) -> np.ndarray | None:
    """Normalize an ``O(n_cells)`` value field to a float64 numpy array (``None`` elements → NaN,
    which every consumer treats as 'absent'). A whole-field ``None`` stays ``None`` (term skipped).

    This is the §V.1 memory contract: the problem holds numpy arrays, not Python float tuples
    (~32 B/element → tens of GB at national scale), and the solver/loss read them directly instead
    of rebuilding a numpy array from the tuple every iteration."""
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)  # None → NaN in a float cast


def _to_bool_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=bool)


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
    closure_totals: tuple[float | None, ...] | None = None
    migration_totals: tuple[float | None, ...] | None = None
    # Observed net-migration total per (locality, time), shape (S, T). ``None`` in a
    # cell = no residual observation for that locality-year (left to smoothness).
    migration_locality_totals: tuple[float | None, ...] | None = None
    migration_bounds: tuple[float, ...] | None = None
    initial_population: tuple[float, ...] | None = None
    initial_migration: tuple[float, ...] | None = None
    weights: PopulationObjectiveWeights = field(default_factory=PopulationObjectiveWeights)

    def __post_init__(self) -> None:
        # Normalize every O(n_cells) array field to a numpy array once, at construction, so the
        # problem is stored compactly (float64/bool arrays, not Python tuples) and downstream reads
        # are direct. None elements become NaN (a sentinel every consumer already treats as absent);
        # a whole-field None stays None. Callers may pass tuples/lists/arrays interchangeably.
        object.__setattr__(self, "anchors", _to_f64_array(self.anchors))
        object.__setattr__(self, "hard_anchor_mask", _to_bool_array(self.hard_anchor_mask))
        for _name in (
            "births", "death_rates", "sim_deaths", "race_composition_prior",
            "closure_totals", "migration_totals", "migration_locality_totals",
            "migration_bounds", "initial_population", "initial_migration",
        ):
            object.__setattr__(self, _name, _to_f64_array(getattr(self, _name)))

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
            # tensor_values and migration_values are O(muni x year x sex x race x age) arrays.
            # They are persisted to the population-tensor parquet (sidra.population_cube.build) and
            # are never read back from this manifest, so the manifest references them by length only
            # rather than inlining. Inlining them produced a ~450MB ReproducibilityManifest at
            # national scale (22M+ float literals) that dominated run memory and disk.
            "tensor_values_count": len(self.tensor_values),
            "migration_values_count": len(self.migration_values),
            "large_arrays_persisted_to": "population_tensor parquet (values not inlined in manifest)",
        }
