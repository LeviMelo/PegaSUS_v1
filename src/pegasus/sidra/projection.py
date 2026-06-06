from __future__ import annotations

from pegasus.sidra.schemas import ProjectionResult


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
