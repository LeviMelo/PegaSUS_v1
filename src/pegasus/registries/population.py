from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PopulationTensorMode = Literal[
    "independent_denominator",
    "sim_informed_denominator",
]


class PopulationRegistryError(ValueError):
    """Raised when a population tensor solver request violates registry constraints."""


class PopulationTensorScaleError(PopulationRegistryError):
    """Raised when a dense population tensor request exceeds the allowed scale."""


@dataclass(frozen=True)
class PopulationSolverSpec:
    solver_id: str
    mode: str
    backend: str
    sparse_jacobian: bool
    max_cells: int
    status: str
    warning_code: str | None = None

    def as_manifest(self) -> dict[str, object]:
        return {
            "solver_id": self.solver_id,
            "mode": self.mode,
            "backend": self.backend,
            "sparse_jacobian": self.sparse_jacobian,
            "max_cells": self.max_cells,
            "status": self.status,
            "warning_code": self.warning_code,
        }


SOLVER_REGISTRY_VERSION = "population_solver_registry_v2"
DENSE_NATIONAL_CELL_THRESHOLD = 10_000_000

POPULATION_SOLVERS: dict[str, PopulationSolverSpec] = {
    "projected_gradient_small_v1": PopulationSolverSpec(
        solver_id="projected_gradient_small_v1",
        mode="independent_denominator",
        backend="projected_gradient_small_sparse_analytic",
        sparse_jacobian=True,
        max_cells=DENSE_NATIONAL_CELL_THRESHOLD,
        status="active",
    ),
    "projected_gradient_small_sim_informed_v1": PopulationSolverSpec(
        solver_id="projected_gradient_small_sim_informed_v1",
        mode="sim_informed_denominator",
        backend="projected_gradient_small_sparse_analytic",
        sparse_jacobian=True,
        max_cells=DENSE_NATIONAL_CELL_THRESHOLD,
        status="active_warning",
        warning_code="sim_informed_population_feedback_risk",
    ),
    "independent_sidra_anchor_v1": PopulationSolverSpec(
        solver_id="independent_sidra_anchor_v1",
        mode="independent_denominator",
        backend="algebraic_sidra_anchor_identity",
        sparse_jacobian=True,
        max_cells=DENSE_NATIONAL_CELL_THRESHOLD,
        status="legacy_identity",
    ),
    "sim_informed_sparse_admm_scaffold_v1": PopulationSolverSpec(
        solver_id="sim_informed_sparse_admm_scaffold_v1",
        mode="sim_informed_denominator",
        backend="sparse_admm_scaffold_blocked_feedback",
        sparse_jacobian=True,
        max_cells=DENSE_NATIONAL_CELL_THRESHOLD,
        status="legacy_warning_scaffold",
        warning_code="sim_informed_population_feedback_risk",
    ),
}


def registry_manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "registry_version": SOLVER_REGISTRY_VERSION,
        "dense_national_cell_threshold": DENSE_NATIONAL_CELL_THRESHOLD,
        "solvers": {key: value.as_manifest() for key, value in POPULATION_SOLVERS.items()},
    }


def get_population_solver(solver_id: str) -> PopulationSolverSpec:
    try:
        return POPULATION_SOLVERS[solver_id]
    except KeyError as exc:
        raise PopulationRegistryError(f"Unknown population tensor solver: {solver_id}") from exc


def select_population_solver(*, mode: str, solver_id: str | None = None) -> PopulationSolverSpec:
    if solver_id is not None:
        spec = get_population_solver(solver_id)
        if spec.mode != mode:
            raise PopulationRegistryError(f"Solver {solver_id} has mode {spec.mode}, not requested mode {mode}.")
        return spec

    for spec in POPULATION_SOLVERS.values():
        if spec.mode == mode and spec.status.startswith("active"):
            return spec
    raise PopulationRegistryError(f"No population tensor solver registered for mode: {mode}")


def estimate_population_tensor_cells(*, localities: int, periods: int, strata: int = 1) -> int:
    if localities < 0 or periods < 0 or strata < 0:
        raise PopulationRegistryError("Population tensor dimensions must be nonnegative.")
    return int(localities) * int(periods) * int(strata)


def assert_dense_population_tensor_allowed(
    *,
    localities: int,
    periods: int,
    strata: int = 1,
    threshold: int = DENSE_NATIONAL_CELL_THRESHOLD,
) -> int:
    cells = estimate_population_tensor_cells(localities=localities, periods=periods, strata=strata)
    if cells > threshold:
        raise PopulationTensorScaleError(
            "dense national population tensor request exceeds scale threshold: "
            f"cells={cells} threshold={threshold}. Use sparse/block solver planning instead."
        )
    return cells
