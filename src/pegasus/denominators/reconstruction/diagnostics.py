from __future__ import annotations

from pegasus.registries.population import DENSE_NATIONAL_CELL_THRESHOLD, PopulationSolverSpec
from pegasus.denominators.reconstruction.schema import PopulationSolverTelemetry, PopulationTensorDiagnostics, PopulationTensorRequest


def population_tensor_diagnostics(
    *,
    request: PopulationTensorRequest,
    solver: PopulationSolverSpec,
    reconstruction_uncertainty: float,
    denominator_feedback_warning: bool,
    telemetry: PopulationSolverTelemetry,
    warnings: list[str] | tuple[str, ...] | None = None,
) -> PopulationTensorDiagnostics:
    return PopulationTensorDiagnostics(
        n_cells=request.n_cells,
        solver_backend=solver.backend,
        sparse_jacobian=solver.sparse_jacobian,
        reconstruction_uncertainty=float(reconstruction_uncertainty),
        denominator_feedback_warning=bool(denominator_feedback_warning),
        dense_abort_threshold=DENSE_NATIONAL_CELL_THRESHOLD,
        converged=telemetry.converged,
        iterations=telemetry.iterations,
        initial_objective=telemetry.initial_objective,
        final_objective=telemetry.final_objective,
        projected_gradient_norm=telemetry.projected_gradient_norm,
        relative_objective_change=telemetry.relative_objective_change,
        objective_terms=telemetry.objective_terms,
        warnings=tuple(warnings or ()),
    )
