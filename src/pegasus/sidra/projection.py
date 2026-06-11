from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegasus.sidra.schemas import ProjectionResult


class SIDRAProjectionError(ValueError):
    """Raised when a SIDRA classification projection matrix is malformed."""


@dataclass(frozen=True)
class ClassificationProjectionMatrix:
    matrix_id: str
    source_axis: str
    target_axis: str
    weights: dict[str, dict[str, float]]
    fractional: bool

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source, targets in self.weights.items():
            for target, weight in targets.items():
                rows.append(
                    {
                        "projection_matrix_id": self.matrix_id,
                        "source_axis": self.source_axis,
                        "target_axis": self.target_axis,
                        "source_category": source,
                        "target_category": target,
                        "weight": float(weight),
                        "fractional": bool(weight not in {0.0, 1.0}),
                    }
                )
        return rows

    def warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.fractional:
            warnings.append("sidra_fractional_classification_projection")
        return warnings


def project_classification_to_axis(
    *,
    measure_kind: str,
    has_denominator: bool,
    projection_matrix_id: str | None,
) -> ProjectionResult:
    if projection_matrix_id is None:
        return ProjectionResult(
            status="blocked",
            projection_matrix_id=None,
            warnings=["projection_matrix_missing"],
            reason="No registered SIDRA classification projection matrix was supplied.",
        )

    if measure_kind in {"rate", "proportion", "percentage"} and not has_denominator:
        return ProjectionResult(
            status="blocked",
            projection_matrix_id=projection_matrix_id,
            warnings=["ratio_projection_requires_denominator_recovery"],
            reason="Direct projection of rates/proportions is illegal without numerator/denominator recovery.",
        )

    return ProjectionResult(
        status="projected",
        projection_matrix_id=projection_matrix_id,
        warnings=[],
    )


def load_projection_matrix(payload: dict[str, Any]) -> ClassificationProjectionMatrix:
    matrix_id = str(payload.get("matrix_id") or payload.get("projection_matrix_id") or "projection_unset")
    source_axis = str(payload.get("source_axis") or "source")
    target_axis = str(payload.get("target_axis") or "target")
    raw_entries = payload.get("entries", [])
    weights: dict[str, dict[str, float]] = {}
    for entry in raw_entries:
        source = str(entry["source"])
        target = str(entry["target"])
        weight = float(entry["weight"])
        if weight < 0:
            raise SIDRAProjectionError("Projection matrix weights must be non-negative.")
        weights.setdefault(source, {})[target] = weights.setdefault(source, {}).get(target, 0.0) + weight
    for source, targets in weights.items():
        total = sum(targets.values())
        if abs(total - 1.0) > 1e-9:
            raise SIDRAProjectionError(f"Projection matrix row {source!r} sums to {total}, not 1.")
    fractional = any(weight not in {0.0, 1.0} for targets in weights.values() for weight in targets.values())
    return ClassificationProjectionMatrix(
        matrix_id=matrix_id,
        source_axis=source_axis,
        target_axis=target_axis,
        weights=weights,
        fractional=fractional,
    )


def projection_metadata(
    *,
    measure_kind: str,
    has_denominator: bool,
    matrix: ClassificationProjectionMatrix | None,
) -> dict[str, Any]:
    result = project_classification_to_axis(
        measure_kind=measure_kind,
        has_denominator=has_denominator,
        projection_matrix_id=matrix.matrix_id if matrix else None,
    )
    warnings = list(result.warnings)
    if matrix is not None:
        warnings.extend(matrix.warnings())
    return {
        "status": result.status,
        "projection_matrix_id": result.projection_matrix_id,
        "source_axis": matrix.source_axis if matrix else None,
        "target_axis": matrix.target_axis if matrix else None,
        "fractional": matrix.fractional if matrix else False,
        "warnings": list(dict.fromkeys(warnings)),
        "reason": result.reason,
    }
