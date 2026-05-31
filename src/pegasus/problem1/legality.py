from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pegasus.problem1.contracts import DeltaResult, FailedBranch
from pegasus.registries.index import RegistryIndex


@dataclass(frozen=True)
class RateLegalityRequest:
    """Minimal legality request for a numerator/denominator rate or proportion.

    declaration_axis_required controls whether declaration-process compatibility
    should be evaluated. For crude all-axis rates, this should remain False. For
    race-specific rates, it must be True, unless explicit compatible race axes
    are supplied.
    """

    numerator_carrier: str
    denominator_carrier: str
    role: str

    numerator_unit: str
    denominator_unit: str

    numerator_source_system: str | None = None
    denominator_source_system: str | None = None

    numerator_race_axis: str | None = None
    denominator_race_axis: str | None = None

    declaration_axis_required: bool = False
    bridge_applied: Literal["Bridge_R"] | None = None

    support_compatible: bool = True
    axes_compatible: bool = True
    aggregation_compatible: bool = True
    provenance_compatible: bool = True
    quality_compatible: bool = True


def evaluate_rate_legality(
    request: RateLegalityRequest,
    registry_index: RegistryIndex,
) -> DeltaResult:
    """Evaluate a first implementation legality predicate.

    Implements the current executable subset of:

        Δ = Δ_support Δ_axes Δ_carrier Δ_unit Δ_aggregation
            Δ_provenance Δ_quality Δ_declaration

    The declaration term is deliberately gated. Source-system race-axis
    noncommensurability must not block all-race crude rates. It only blocks
    rates whose estimand actually uses the race/color declaration axis.
    """
    failed_terms: list[str] = []
    warnings: list[str] = []

    delta_support = int(request.support_compatible)
    if not delta_support:
        failed_terms.append("delta_support")

    delta_axes = int(request.axes_compatible)
    if not delta_axes:
        failed_terms.append("delta_axes")

    carrier_rule = registry_index.carrier_rule(
        request.numerator_carrier,
        request.denominator_carrier,
        request.role,
    )
    delta_carrier = int(carrier_rule is not None and carrier_rule.legal)
    if not delta_carrier:
        failed_terms.append("delta_carrier")
        warnings.append(
            "carrier_incompatible:"
            f"{request.numerator_carrier}/{request.denominator_carrier}->{request.role}"
        )

    unit_rule = registry_index.unit_rule(
        request.numerator_unit,
        request.denominator_unit,
        request.role,
    )
    delta_unit = int(unit_rule is not None)
    if not delta_unit:
        failed_terms.append("delta_unit")
        warnings.append(
            "unit_incompatible:"
            f"{request.numerator_unit}/{request.denominator_unit}->{request.role}"
        )

    delta_aggregation = int(request.aggregation_compatible)
    if not delta_aggregation:
        failed_terms.append("delta_aggregation")

    delta_provenance = int(request.provenance_compatible)
    if not delta_provenance:
        failed_terms.append("delta_provenance")

    delta_quality = int(request.quality_compatible)
    if not delta_quality:
        failed_terms.append("delta_quality")

    delta_declaration, declaration_warnings = _evaluate_declaration_compatibility(
        request,
        registry_index,
    )
    warnings.extend(declaration_warnings)

    if not delta_declaration:
        failed_terms.append("delta_declaration")

    legal = all(
        term == 1
        for term in (
            delta_support,
            delta_axes,
            delta_carrier,
            delta_unit,
            delta_aggregation,
            delta_provenance,
            delta_quality,
            delta_declaration,
        )
    )

    failed_branch_id = None
    if not legal:
        failed_branch = FailedBranch(
            attempted_operator="RN",
            parent_field_ids=[],
            reason_code="illegal_rate_transformation",
            recoverable=True,
            suggested_route="Apply Bridge_R or change denominator/numerator compatibility.",
        )
        failed_branch_id = failed_branch.failed_branch_id

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


def _evaluate_declaration_compatibility(
    request: RateLegalityRequest,
    registry_index: RegistryIndex,
) -> tuple[int, list[str]]:
    warnings: list[str] = []

    explicit_race_axes = (
        request.numerator_race_axis is not None
        or request.denominator_race_axis is not None
    )

    if not request.declaration_axis_required and not explicit_race_axes:
        return 1, warnings

    numerator_axis = request.numerator_race_axis
    denominator_axis = request.denominator_race_axis

    if numerator_axis is None and request.numerator_source_system is not None:
        entry = registry_index.race_axis_for_source(request.numerator_source_system)
        if entry is not None:
            numerator_axis = entry.race_axis

    if denominator_axis is None and request.denominator_source_system is not None:
        entry = registry_index.race_axis_for_source(request.denominator_source_system)
        if entry is not None:
            denominator_axis = entry.race_axis

    if numerator_axis is None or denominator_axis is None:
        warnings.append(
            "race_axis_required_but_unresolved:"
            f"numerator={numerator_axis}; denominator={denominator_axis}"
        )
        return 0, warnings

    if numerator_axis == denominator_axis:
        return 1, warnings

    if request.bridge_applied == "Bridge_R":
        warnings.append(
            "race_axis_bridge_applied:"
            f"{numerator_axis}->self_declared via Bridge_R"
        )
        return 1, warnings

    warnings.append(
        "race_axis_noncommensurable:"
        f"numerator={numerator_axis}; denominator={denominator_axis}; "
        "Bridge_R required"
    )
    return 0, warnings