from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pegasus.core.schemas import DenominatorContract


PopulationTensorMode = Literal["independent_denominator", "sim_informed_denominator"]


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
        warnings=warnings or ["fixture_or_unvalidated_sidra_anchor"],
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
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "solver_backend": self.solver_backend,
            "sparse_jacobian": self.sparse_jacobian,
            "reconstruction_uncertainty": self.reconstruction_uncertainty,
            "denominator_feedback_warning": self.denominator_feedback_warning,
            "dense_abort_threshold": self.dense_abort_threshold,
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

    def as_manifest(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "PopulationTensorMode": self.mode,
            "SolverBackend": self.solver_backend,
            "SolverID": self.solver_id,
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
            "DenominatorFeedbackWarning": self.denominator_feedback_warning,
            "state": self.state,
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics.as_manifest(),
        }
