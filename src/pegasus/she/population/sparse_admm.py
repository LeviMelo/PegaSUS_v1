"""Sparse population solver facade with scale and memory proofs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegasus.compute.memory import MemoryPreflight, preflight_memory
from pegasus.she.population.block_coordinate import solve_population_block_coordinate
from pegasus.she.population.projected_gradient import PopulationOptimizationResult
from pegasus.she.population.schema import PopulationTensorProblem
from pegasus.she.population.state_space import PopulationStateSpace, build_population_state_space


@dataclass(frozen=True)
class SparsePopulationPlan:
    solver_id: str
    state_space: PopulationStateSpace
    memory: MemoryPreflight
    sparse_jacobian: bool = True

    def as_manifest(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "sparse_jacobian": self.sparse_jacobian,
            "state_space": self.state_space.as_manifest(),
            "memory": self.memory.as_manifest(),
        }


def plan_sparse_population_solver(
    problem: PopulationTensorProblem,
    *,
    solver_id: str = "sparse_block_coordinate_v1",
    available_bytes: int | None = None,
    max_memory_fraction: float = 0.8,
) -> SparsePopulationPlan:
    state_space = build_population_state_space(problem)
    # Two float64 iterates, two gradients, and conservative edge/index overhead.
    estimated_bytes = problem.n_cells * 8 * 6 + (len(state_space.aging_edges) + len(state_space.birth_edges)) * 24
    memory = preflight_memory(
        estimated_bytes,
        available_bytes=available_bytes,
        max_fraction=max_memory_fraction,
        memory_kind="ram",
    )
    return SparsePopulationPlan(solver_id=solver_id, state_space=state_space, memory=memory)


def solve_sparse_population(
    problem: PopulationTensorProblem,
    *,
    solver_id: str = "sparse_block_coordinate_v1",
    max_iterations: int = 2_000,
    tolerance: float = 1e-5,
    available_bytes: int | None = None,
) -> tuple[PopulationOptimizationResult, SparsePopulationPlan]:
    plan = plan_sparse_population_solver(problem, solver_id=solver_id, available_bytes=available_bytes)
    result = solve_population_block_coordinate(problem, max_iterations=max_iterations, tolerance=tolerance)
    return result, plan


@dataclass(frozen=True)
class SparseADMMScaffold:
    solver_id: str
    status: str = "deprecated_replaced_by_sparse_block_coordinate"

    def as_manifest(self) -> dict[str, Any]:
        return {"solver_id": self.solver_id, "status": self.status, "sparse_jacobian": True}


def build_sim_informed_sparse_admm_scaffold(*, solver_id: str) -> SparseADMMScaffold:
    """Compatibility metadata for historical manifests; never selected for execution."""
    return SparseADMMScaffold(solver_id=solver_id)
