from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from pegasus.core.schemas import FieldNode


class OperatorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    output_kind: str | None = None
    params: dict = {}


class DeclarationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    failed_terms: list[str]
    warnings: list[str]
    reason: str | None = None


def _race_axis(field: FieldNode | None) -> str | None:
    if field is None:
        return None
    axes = field.axes or {}
    return (
        axes.get("race_axis_type")
        or axes.get("race_axis")
        or axes.get("declaration_process")
    )


def _bridge_applied(field: FieldNode) -> bool:
    provenance = set(field.provenance or [])
    operators = {field.operator} if field.operator else set()
    warnings = set(field.warnings or [])
    return bool(
        provenance
        & {
            "bayesian_race_axis_bridge",
            "bridge_derived",
            "Bridge_R",
        }
    ) or bool(
        operators
        & {
            "Bridge_R",
            "Bridge_R_fixedC_dynamicW",
            "Bridge_R_posteriorC",
        }
    ) or "race_bridge_applied" in warnings


def evaluate_declaration_compatibility(
    *,
    numerator: FieldNode,
    denominator: FieldNode | None,
    operator: OperatorSpec,
    registries=None,
) -> DeclarationResult:
    if operator.name != "RN" or denominator is None:
        return DeclarationResult(ok=True, failed_terms=[], warnings=[])

    numerator_race = _race_axis(numerator)
    denominator_race = _race_axis(denominator)

    if numerator_race is None or denominator_race is None:
        return DeclarationResult(ok=True, failed_terms=[], warnings=[])

    if numerator_race == denominator_race:
        return DeclarationResult(ok=True, failed_terms=[], warnings=[])

    if _bridge_applied(numerator):
        return DeclarationResult(
            ok=True,
            failed_terms=[],
            warnings=["race_axis_aligned_by_bridge"],
        )

    return DeclarationResult(
        ok=False,
        failed_terms=["declaration"],
        warnings=["race_axis_declaration_incommensurable"],
        reason=(
            f"Race-axis mismatch: numerator={numerator_race}; "
            f"denominator={denominator_race}; Bridge_R not applied."
        ),
    )
