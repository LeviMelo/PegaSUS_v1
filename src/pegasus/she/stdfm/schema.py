from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

STDFMState = Literal["blocked_solver_pending", "certification_pending", "verified"]


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

    def as_manifest(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "status": self.status,
            "solver_backend": self.solver_backend,
            "certification_id": self.certification_id,
            "uncertainty": self.uncertainty,
            "warnings": list(self.warnings),
            "reason": self.reason,
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
