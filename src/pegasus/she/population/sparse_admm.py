from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SparseADMMScaffold:
    solver_id: str
    status: str
    reason: str
    sparse_jacobian: bool
    warnings: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "status": self.status,
            "reason": self.reason,
            "sparse_jacobian": self.sparse_jacobian,
            "warnings": list(self.warnings),
        }


def build_sim_informed_sparse_admm_scaffold(*, solver_id: str) -> SparseADMMScaffold:
    return SparseADMMScaffold(
        solver_id=solver_id,
        status="blocked_feedback_scaffold",
        reason=(
            "SIM-informed population tensor mode is architecturally visible but remains warning-only "
            "until calibrated feedback loss and uncertainty propagation are implemented."
        ),
        sparse_jacobian=True,
        warnings=("sim_informed_population_feedback_risk",),
    )
