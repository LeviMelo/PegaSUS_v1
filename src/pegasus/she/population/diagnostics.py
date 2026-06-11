from __future__ import annotations

from pegasus.registries.population import DENSE_NATIONAL_CELL_THRESHOLD, PopulationSolverSpec
from pegasus.she.population.schema import PopulationTensorDiagnostics, PopulationTensorRequest


def population_tensor_diagnostics(
    *,
    request: PopulationTensorRequest,
    solver: PopulationSolverSpec,
    reconstruction_uncertainty: float,
    denominator_feedback_warning: bool,
    warnings: list[str] | tuple[str, ...] | None = None,
) -> PopulationTensorDiagnostics:
    return PopulationTensorDiagnostics(
        n_cells=request.n_cells,
        solver_backend=solver.backend,
        sparse_jacobian=solver.sparse_jacobian,
        reconstruction_uncertainty=float(reconstruction_uncertainty),
        denominator_feedback_warning=bool(denominator_feedback_warning),
        dense_abort_threshold=DENSE_NATIONAL_CELL_THRESHOLD,
        warnings=tuple(warnings or ()),
    )
