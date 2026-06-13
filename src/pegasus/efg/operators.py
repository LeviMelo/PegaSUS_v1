"""Typed EFG operators and metadata-only operator application.

Macro-Slice 19A deliberately keeps tensor execution outside this module.  The
operator registry defines graph semantics and creates content-addressed field
nodes whose materialization state makes that boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pegasus.core.schemas import AlignmentResult, DeltaResult, FieldNode, OperatorResult
from pegasus.efg.declaration import OperatorSpec
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node


class EFGOperator(str, Enum):
    RAW_FIELD = "raw_field"
    OBSERVER_FIELD = "observer_field"
    DIAGNOSTIC_TOPOLOGY = "diagnostic_topology"
    COUNT_MEASURE = "count_measure"
    DENOMINATOR_LINK = "denominator_link"
    RN = "RN"
    PROJECT = "pi_*"
    CLASSIFICATION_PROJECT = "Pi_Clsf_to_Axis"
    BOUNDED_PROJECT = "pi_bound_*"
    BRIDGE = "Bridge"


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    arity: int | None
    output_kind: str
    output_aggregation: str
    requires_alignment: bool
    description: str

    def spec(self, *, role: str, params: dict[str, Any] | None = None) -> OperatorSpec:
        return OperatorSpec(
            name=self.name,
            role=role,
            output_kind=self.output_kind,
            params=params or {},
        )


OPERATOR_REGISTRY: dict[str, OperatorDefinition] = {
    EFGOperator.RAW_FIELD.value: OperatorDefinition(
        EFGOperator.RAW_FIELD.value, 1, "observer_proxy", "non_aggregable", False,
        "Admit one registry-backed SHE field without claiming a derived measure.",
    ),
    EFGOperator.OBSERVER_FIELD.value: OperatorDefinition(
        EFGOperator.OBSERVER_FIELD.value, 1, "observer_proxy", "non_aggregable", False,
        "Admit a metadata/measurement-process observer field.",
    ),
    EFGOperator.DIAGNOSTIC_TOPOLOGY.value: OperatorDefinition(
        EFGOperator.DIAGNOSTIC_TOPOLOGY.value, 1, "observer_proxy", "non_aggregable", False,
        "Preserve diagnostic role and topology; never treat ICD codes as additive values.",
    ),
    EFGOperator.COUNT_MEASURE.value: OperatorDefinition(
        EFGOperator.COUNT_MEASURE.value, None, "extensive_measure", "additive", False,
        "Create a planned source-event count over the common source artifact support.",
    ),
    EFGOperator.DENOMINATOR_LINK.value: OperatorDefinition(
        EFGOperator.DENOMINATOR_LINK.value, 2, "observer_proxy", "non_aggregable", True,
        "Record a legal numerator/denominator relation before RN materialization.",
    ),
    EFGOperator.RN.value: OperatorDefinition(
        EFGOperator.RN.value, 2, "intensive_density", "weighted_mean", True,
        "Finite-cell Radon-Nikodym ratio after alignment and Delta legality.",
    ),
    EFGOperator.PROJECT.value: OperatorDefinition(
        EFGOperator.PROJECT.value, 1, "extensive_measure", "additive", False,
        "Marginalize additive measures over explicitly declared axes.",
    ),
    EFGOperator.CLASSIFICATION_PROJECT.value: OperatorDefinition(
        EFGOperator.CLASSIFICATION_PROJECT.value, 1, "context_gradient", "additive", False,
        "Project source classifications into a registered canonical axis.",
    ),
    EFGOperator.BOUNDED_PROJECT.value: OperatorDefinition(
        EFGOperator.BOUNDED_PROJECT.value, 1, "extensive_measure", "additive", False,
        "Mandatory early marginalization of high-cardinality additive fields.",
    ),
    EFGOperator.BRIDGE.value: OperatorDefinition(
        EFGOperator.BRIDGE.value, None, "bridge_module", "non_aggregable", True,
        "Typed cross-system bridge; unavailable bridges fail explicitly.",
    ),
}


@dataclass(frozen=True)
class RatioRule:
    numerator_carrier: str
    denominator_carrier: str
    numerator_unit: str
    denominator_units: tuple[str, ...]
    role: str
    output_unit: str


RATIO_RULES: tuple[RatioRule, ...] = (
    RatioRule("Deaths", "Population", "counts", ("person_years", "persons"),
              "mortality_rate", "rate"),
    RatioRule("HospitalAdmissions", "Population", "counts", ("person_years", "persons"),
              "hospitalization_rate", "rate"),
    RatioRule("LiveBirths", "Population", "counts", ("person_years", "persons"),
              "birth_rate", "rate"),
    RatioRule("HospitalDeaths", "HospitalAdmissions", "counts", ("counts",),
              "inpatient_mortality", "proportion"),
    RatioRule("InfantDeaths", "LiveBirths", "counts", ("counts",),
              "infant_mortality", "proportion"),
    RatioRule("LowBirthWeightBirths", "LiveBirths", "counts", ("counts",),
              "birth_outcome_share", "proportion"),
)


def get_operator(name: str) -> OperatorDefinition:
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unsupported EFG operator: {name}") from exc


def ratio_rule(numerator: FieldNode, denominator: FieldNode, role: str | None = None) -> RatioRule | None:
    for rule in RATIO_RULES:
        if (
            rule.numerator_carrier == numerator.carrier
            and rule.denominator_carrier == denominator.carrier
            and rule.numerator_unit == numerator.unit
            and denominator.unit in rule.denominator_units
            and (role is None or role == rule.role)
        ):
            return rule
    return None


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _registry_versions(parents: list[FieldNode]) -> dict[str, str]:
    versions: dict[str, str] = {"efg_operator_registry": "slice19a"}
    for parent in parents:
        versions.update(parent.lineage.registry_versions)
    return versions


def _source_hashes(parents: list[FieldNode]) -> list[str]:
    return _unique(
        value
        for parent in parents
        for value in parent.lineage.source_manifest_hashes
    )


def _result_for_field(field: FieldNode) -> OperatorResult:
    return OperatorResult(
        status="success",
        output_field_id=field.id,
        materialization_state=field.materialization_state,
        provenance=field.provenance,
        warnings=field.warnings,
        failed_branch_id=None,
    )


def apply_operator(
    *,
    operator: OperatorSpec,
    parents: list[FieldNode],
    delta: DeltaResult,
    alignment: AlignmentResult | None = None,
    registries: Any = None,
) -> tuple[OperatorResult, FieldNode | None]:
    """Apply one legal operator at the graph metadata boundary."""

    del registries
    if not delta.legal:
        return (
            OperatorResult(
                status="failed",
                output_field_id=None,
                materialization_state="failed",
                provenance=[],
                warnings=list(delta.warnings),
                failed_branch_id=delta.failed_branch_id,
            ),
            None,
        )

    definition = get_operator(operator.name)
    if definition.arity is not None and len(parents) != definition.arity:
        return (
            OperatorResult(
                status="failed",
                output_field_id=None,
                materialization_state="failed",
                provenance=[],
                warnings=["operator_arity_mismatch"],
                failed_branch_id="failed_operator_arity",
            ),
            None,
        )
    if not parents:
        raise ValueError(f"operator {operator.name} requires at least one parent")

    if operator.name == EFGOperator.COUNT_MEASURE.value:
        parent = parents[0]
        artifact = Path(str(parent.support.get("artifact_path", "source"))).stem
        support = dict(parent.support)
        support.update({
            "support_kind": "source_artifact_event_count",
            "source_columns": sorted({str(p.support.get("column")) for p in parents}),
        })
        axes: dict[str, Any] = {}
        for item in parents:
            axes.update(item.axes)
        carrier = str(operator.params.get("carrier", parent.carrier))
        name = str(operator.params.get("name", f"{parent.source[0]}.{artifact}.count"))
        unit = "counts"
        aggregation = "additive"
        kind = "extensive_measure"
        role = ["source_event_count", operator.role]
    elif operator.name in {EFGOperator.PROJECT.value, EFGOperator.BOUNDED_PROJECT.value}:
        parent = parents[0]
        drop_axes = {str(axis) for axis in operator.params.get("drop_axes", [])}
        support = dict(parent.support)
        axes = {key: value for key, value in parent.axes.items() if key not in drop_axes}
        carrier = parent.carrier
        name = str(operator.params.get("name", f"{parent.name}.projected"))
        unit = parent.unit
        aggregation = parent.aggregation
        kind = parent.kind
        role = _unique([*parent.role, "projected"])
    elif operator.name == EFGOperator.RN.value:
        numerator, denominator = parents
        rule = ratio_rule(numerator, denominator, operator.role)
        if rule is None:
            raise ValueError("legal RN operator has no matching ratio rule")
        support = dict((alignment.support_after_alignment if alignment else None) or numerator.support)
        axes = {
            key: value
            for key, value in numerator.axes.items()
            if key in denominator.axes and denominator.axes[key] == value
        }
        carrier = f"{numerator.carrier}/{denominator.carrier}"
        name = str(operator.params.get("name", f"{numerator.name}.{rule.role}"))
        unit = rule.output_unit
        aggregation = "weighted_mean"
        kind = "intensive_density"
        role = [rule.role, "derived_ratio"]
    else:
        return (
            OperatorResult(
                status="blocked",
                output_field_id=None,
                materialization_state="blocked",
                provenance=[],
                warnings=[f"operator_execution_not_implemented:{operator.name}"],
                failed_branch_id="failed_operator_unsupported",
            ),
            None,
        )

    lineage = make_lineage(
        parent_ids=[parent.id for parent in parents],
        operator_type=operator.name,
        operator_params=dict(operator.params),
        registry_versions=_registry_versions(parents),
        source_manifest_hashes=_source_hashes(parents),
        code_version="slice19a",
    )
    warnings = _unique([
        *(warning for parent in parents for warning in parent.warnings),
        *delta.warnings,
        *((alignment.warnings if alignment else [])),
    ])
    provenance = _unique([
        *(tag for parent in parents for tag in parent.provenance),
        "efg_derived",
    ])
    field = make_field_node(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        carrier=carrier,
        unit=unit,
        support=support,
        axes=axes,
        aggregation=aggregation,  # type: ignore[arg-type]
        role=_unique(role),
        source=_unique(source for parent in parents for source in parent.source),
        operator=operator.name,
        provenance=provenance,
        state="fragile" if warnings else "verified",
        warnings=warnings,
        lineage=lineage,
        materialization_state="planned",
        dashboard_safe="warning" if warnings else False,
    )
    return _result_for_field(field), field
