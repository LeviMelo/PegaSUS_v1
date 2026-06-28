from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pegasus.core.schemas import FieldNode


class OperatorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    output_kind: str | None = None
    params: dict = Field(default_factory=dict)


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


def _race_metadata_required(field: FieldNode | None) -> bool:
    """Detect race-stratified operands whose declaration axis must be explicit."""
    if field is None:
        return False
    roles = {str(value).casefold() for value in (field.role or [])}
    sources = {str(value).casefold() for value in (field.source or [])}
    axes = {str(key).casefold(): str(value).casefold() for key, value in (field.axes or {}).items()}
    support = {str(key).casefold(): str(value).casefold() for key, value in (field.support or {}).items()}
    text = " ".join([
        " ".join(roles),
        " ".join(sources),
        " ".join(f"{key}={value}" for key, value in axes.items()),
        " ".join(f"{key}={value}" for key, value in support.items()),
    ])
    tokens = ("race", "raca", "raça", "cor_raca", "cor/raça", "race_axis")
    return any(token in text for token in tokens)


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
            "Bridge_R_localPi_posteriorC",
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
    numerator_requires_race_axis = _race_metadata_required(numerator)
    denominator_requires_race_axis = _race_metadata_required(denominator)

    if numerator_race is None and denominator_race is None:
        if numerator_requires_race_axis or denominator_requires_race_axis:
            return DeclarationResult(
                ok=False,
                failed_terms=["declaration"],
                warnings=["race_axis_declaration_unverifiable"],
                reason=(
                    "Race-stratified RN declaration lacks explicit race_axis_type "
                    f"metadata: numerator_required={numerator_requires_race_axis}; "
                    f"denominator_required={denominator_requires_race_axis}."
                ),
            )
        return DeclarationResult(ok=True, failed_terms=[], warnings=[])

    if numerator_race is None or denominator_race is None:
        # Exactly one operand declares a race axis and the other does not. This is
        # the dangerous incommensurability case: a race-stratified operand cannot
        # be combined with one whose race semantics are unknown. Fail closed
        # (EFG-DECL-02, MSD §4.3). The metadata-missing warning is always emitted;
        # the unverifiable warning is added when the declared operand additionally
        # required an explicit axis.
        warnings = ["race_axis_metadata_missing_fail_closed"]
        if numerator_requires_race_axis or denominator_requires_race_axis:
            warnings.append("race_axis_declaration_unverifiable")
        return DeclarationResult(
            ok=False,
            failed_terms=["declaration"],
            warnings=warnings,
            reason=(
                f"Race-axis metadata missing for RN declaration: "
                f"numerator={numerator_race}; denominator={denominator_race}."
            ),
        )

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
