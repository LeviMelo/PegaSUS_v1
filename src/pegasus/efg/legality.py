from __future__ import annotations

from datetime import datetime, timezone

from pegasus.core.schemas import DeltaResult, FailedBranch, FieldNode
from pegasus.efg.declaration import OperatorSpec, evaluate_declaration_compatibility


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quality_ok(field: FieldNode) -> bool:
    return field.state.value not in {"illegal_excluded", "quarantined_nochildren"}


def evaluate_delta(
    *,
    parents: list[FieldNode],
    operator: OperatorSpec,
    intent=None,
    registries=None,
    alignment=None,
) -> DeltaResult:
    numerator = parents[0] if parents else None
    denominator = parents[1] if len(parents) > 1 else None

    delta_support = 1
    delta_axes = 1
    delta_carrier = 1
    delta_unit = 1
    delta_aggregation = 1
    delta_provenance = 1
    delta_quality = 1
    delta_declaration = 1
    warnings: list[str] = []
    failed_terms: list[str] = []

    if operator.name == "RN":
        if numerator is None or denominator is None:
            delta_support = 0
            failed_terms.append("support")
        else:
            if not (numerator.carrier == "Deaths" and denominator.carrier == "Population"):
                delta_carrier = 0
                failed_terms.append("carrier")
            if not (numerator.unit == "counts" and denominator.unit == "person_years"):
                delta_unit = 0
                failed_terms.append("unit")
            if numerator.aggregation != "additive" or denominator.aggregation != "additive":
                delta_aggregation = 0
                failed_terms.append("aggregation")

    for parent in parents:
        if not _quality_ok(parent):
            delta_quality = 0
            if "quality" not in failed_terms:
                failed_terms.append("quality")

    if numerator is not None:
        declaration = evaluate_declaration_compatibility(
            numerator=numerator,
            denominator=denominator,
            operator=operator,
            registries=registries,
        )
        if not declaration.ok:
            delta_declaration = 0
            failed_terms.extend(x for x in declaration.failed_terms if x not in failed_terms)
            warnings.extend(declaration.warnings)

    legal = all(
        [
            delta_support,
            delta_axes,
            delta_carrier,
            delta_unit,
            delta_aggregation,
            delta_provenance,
            delta_quality,
            delta_declaration,
        ]
    )

    failed_branch_id = None
    if not legal:
        failed_branch_id = "failed_" + "_".join(failed_terms) if failed_terms else "failed_unknown"

    return DeltaResult(
        legal=legal,
        delta_support=delta_support,
        delta_axes=delta_axes,
        delta_carrier=delta_carrier,
        delta_unit=delta_unit,
        delta_aggregation=delta_aggregation,
        delta_provenance=delta_provenance,
        delta_quality=delta_quality,
        delta_declaration=delta_declaration,
        failed_terms=failed_terms,
        warnings=warnings,
        failed_branch_id=failed_branch_id,
    )


def make_failed_branch(
    *,
    parents: list[FieldNode],
    operator: OperatorSpec,
    delta: DeltaResult,
    reason: str,
) -> FailedBranch:
    failure_stage = "declaration" if "declaration" in delta.failed_terms else (
        delta.failed_terms[0] if delta.failed_terms else "output_validation"
    )
    return FailedBranch(
        failed_branch_id=delta.failed_branch_id or "failed_unknown",
        attempted_operator=operator.name,
        parent_field_ids=[p.id for p in parents],
        failure_stage=failure_stage,
        failed_terms=delta.failed_terms,
        reason=reason,
        warnings=delta.warnings,
        created_at=_now(),
    )
