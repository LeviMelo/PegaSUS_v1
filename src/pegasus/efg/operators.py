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
    PSI_FUNCTIONAL = "psi_functional"
    DENOMINATOR_LINK = "denominator_link"
    RN = "RN"
    DIVERGENCE = "divergence_log_ratio"
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
    EFGOperator.PSI_FUNCTIONAL.value: OperatorDefinition(
        EFGOperator.PSI_FUNCTIONAL.value, None, "marked_functional", "statistical_functional", False,
        "Statistical functional (mean/median) of a per-record mark over a support cell (§3.10.4-6).",
    ),
    EFGOperator.DENOMINATOR_LINK.value: OperatorDefinition(
        EFGOperator.DENOMINATOR_LINK.value, 2, "observer_proxy", "non_aggregable", True,
        "Record a legal numerator/denominator relation before RN materialization.",
    ),
    EFGOperator.RN.value: OperatorDefinition(
        EFGOperator.RN.value, 2, "intensive_density", "weighted_mean", True,
        "Finite-cell Radon-Nikodym ratio after alignment and Delta legality.",
    ),
    EFGOperator.DIVERGENCE.value: OperatorDefinition(
        EFGOperator.DIVERGENCE.value, 2, "bridge_divergence", "non_aggregable", True,
        "Cross-source divergence: log-ratio of two count measures on shared support (§2.11).",
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


def _registry_root_str(registries: Any) -> str:
    if isinstance(registries, (str, Path)):
        return str(registries)
    if isinstance(registries, dict) and registries.get("registry_root"):
        return str(registries["registry_root"])
    root = getattr(registries, "root", None)
    if root is not None:
        return str(root)
    return "config/registries"


def get_operator(name: str) -> OperatorDefinition:
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unsupported EFG operator: {name}") from exc


def ratio_rule(
    numerator: FieldNode,
    denominator: FieldNode,
    role: str | None = None,
    *,
    registry_root: str | Path = "config/registries",
) -> RatioRule | None:
    """Resolve a legal RN ratio from the clinical event registry (MSD §2.6).

    Carrier/role pairings come from ``clinical_event_definitions.yaml``; the engine
    holds no hardcoded ratio table.  Units are validated against the registry-declared
    numerator unit and the general person-time/proportion denominator policy.
    """
    from pegasus.registries.events import clinical_ratio_specs

    for spec in clinical_ratio_specs(root=registry_root):
        if (
            spec.numerator_carrier == numerator.carrier
            and spec.denominator_carrier == denominator.carrier
            and spec.numerator_unit == numerator.unit
            and denominator.unit in spec.denominator_units
            and (role is None or role == spec.role)
        ):
            return RatioRule(
                numerator_carrier=spec.numerator_carrier,
                denominator_carrier=spec.denominator_carrier,
                numerator_unit=spec.numerator_unit,
                denominator_units=spec.denominator_units,
                role=spec.role,
                output_unit=spec.output_unit,
            )
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


def _support_axis_columns(parents: list[FieldNode]) -> dict[str, str]:
    """Carry registry-declared support-axis columns into derived operators.

    Source fields already know their raw column in ``support.column`` and their
    semantic axis in ``axes``/``role``. The physical executor should not guess
    source-specific names; it consumes these canonical declarations.
    """

    output: dict[str, str] = {}
    for parent in parents:
        column = parent.support.get("column")
        if not column:
            continue
        roles = set(parent.role or [])
        axes = dict(parent.axes or {})
        if "time" in axes or "time_axis_candidate" in roles:
            output.setdefault("time_column", str(column))
        if "geography" in axes or "geography_axis" in roles:
            output.setdefault("geography_column", str(column))
    return output


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

    registry_root = _registry_root_str(registries)
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
            **_support_axis_columns(parents),
        })
        axes: dict[str, Any] = {}
        for item in parents:
            axes.update(item.axes)
        if support.get("geography_column"):
            axes["geography"] = str(support["geography_column"])
        if support.get("time_column"):
            axes["time"] = str(support["time_column"])
        # σ_C restriction (MSD §3.11): declare the ICD chapter/block stratification so
        # the physical executor groups counts within diagnostic strata rather than over
        # the whole event population. The restriction is what makes the diagnostic
        # observer legally countable.
        stratify_icd = operator.params.get("stratify_icd")
        role = ["source_event_count", operator.role]
        # σ restriction (MSD §2.6/§3.10.4): a declarative predicate over the parent's
        # source records yields a derived event carrier (e.g. InfantDeaths from Deaths).
        # The role carries "restricted_count" so the downstream RN ratio is legal.
        restrict_predicate = operator.params.get("restrict_predicate")
        if restrict_predicate:
            support.update({
                "support_kind": "source_artifact_event_count_restricted",
                "restrict_predicate": str(restrict_predicate),
                "restrict_conditions": list(operator.params.get("restrict_conditions") or []),
                "restrict_of_carrier": operator.params.get("restrict_of_carrier"),
                "restrict_event_id": operator.params.get("restrict_event_id"),
            })
            role.append("restricted_count")
        if stratify_icd:
            icd_axis = str(operator.params.get("icd_axis") or f"icd_{stratify_icd}")
            support.update({
                "support_kind": "source_artifact_event_count_icd_restricted",
                "stratify_icd": str(stratify_icd),
                "icd_column": str(operator.params.get("icd_column")),
                "icd_axis": icd_axis,
                "icd_source_field": operator.params.get("icd_source_field"),
            })
            axes[icd_axis] = "diagnostic_restriction"
            role.append("restricted_count")
        # General demographic stratification (sex/age/race) by a canonical-mapped column.
        stratify_column = operator.params.get("stratify_column")
        if stratify_column:
            strat_axis = str(operator.params.get("stratify_axis") or stratify_column)
            support.update({
                "support_kind": "source_artifact_event_count_demographic_stratified",
                "stratify_column": str(stratify_column),
                "stratify_axis": strat_axis,
                "stratify_source": operator.params.get("stratify_source"),
            })
            axes[strat_axis] = "demographic_stratifier"
            role.append("demographic_stratified_count")
        # Carry the spatial-aggregation level (MSD §3.7) into the support so the
        # executor coarsens the geography cell consistently for every count.
        if operator.params.get("geography_aggregation"):
            support["geography_aggregation"] = str(operator.params["geography_aggregation"])
        carrier = str(operator.params.get("carrier", parent.carrier))
        name = str(operator.params.get("name", f"{parent.source[0]}.{artifact}.count"))
        unit = "counts"
        aggregation = "additive"
        kind = "extensive_measure"
    elif operator.name == EFGOperator.DIVERGENCE.value:
        left, right = parents
        support = dict((alignment.support_after_alignment if alignment else None) or left.support)
        support.update({"support_kind": "cross_source_divergence", "divergence_left_carrier": left.carrier,
                        "divergence_right_carrier": right.carrier, "epsilon": 1e-9})
        # Carry a declared year lag (MSD §2.11) into the support so the executor pairs
        # left(t-k) with right(t) when materializing the divergence tensor.
        if operator.params.get("temporal_lag"):
            support["temporal_lag"] = int(operator.params["temporal_lag"])
        axes = {key: value for key, value in left.axes.items() if key in right.axes and right.axes[key] == value}
        # Preserve shared stratifier axes (cause-specific / demographic divergence).
        for axis_name in ("icd_chapter", "icd_block", "curated_cause_group", "sex", "age_group", "race"):
            if axis_name in left.axes and axis_name in right.axes:
                axes[axis_name] = left.axes[axis_name]
        carrier = f"{left.carrier}_vs_{right.carrier}"
        name = str(operator.params.get("name", f"{left.name}.divergence"))
        unit = "log_ratio"
        aggregation = "non_aggregable"
        kind = "bridge_divergence"
        role = [operator.role or "cross_source_divergence", "cross_source_relationship", "covariate"]
    elif operator.name == EFGOperator.PSI_FUNCTIONAL.value:
        parent = parents[0]
        artifact = Path(str(parent.support.get("artifact_path", "source"))).stem
        functional = str(operator.params.get("functional", "mean"))
        mark_column = str(operator.params.get("mark_column", ""))
        support = dict(parent.support)
        support.update({
            "support_kind": "source_artifact_statistical_functional",
            "functional": functional,
            "mark_column": mark_column,
            **_support_axis_columns(parents),
        })
        axes = {}
        for item in parents:
            axes.update(item.axes)
        carrier = str(operator.params.get("carrier", parent.carrier))
        name = str(operator.params.get("name", f"{parent.source[0]}.{artifact}.{functional}.{mark_column}"))
        unit = str(operator.params.get("unit", "value"))
        aggregation = "statistical_functional"
        kind = "marked_functional"
        role = ["statistical_functional", "covariate", operator.role]
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
        rule = ratio_rule(numerator, denominator, operator.role, registry_root=registry_root)
        if rule is None:
            raise ValueError("legal RN operator has no matching ratio rule")
        support = dict((alignment.support_after_alignment if alignment else None) or numerator.support)
        axes = {
            key: value
            for key, value in numerator.axes.items()
            if key in denominator.axes and denominator.axes[key] == value
        }
        # Preserve the numerator's diagnostic-restriction stratifier axes: the population
        # denominator is unstratified, so the intersection above would drop them, yet the
        # physical cause-specific rate tensor carries them and they define the estimand.
        for axis_name in ("icd_chapter", "icd_block", "curated_cause_group", "sex", "age_group", "race"):
            if axis_name in numerator.axes:
                axes[axis_name] = numerator.axes[axis_name]
        if numerator.support.get("stratify_icd"):
            support.update({
                "stratify_icd": numerator.support.get("stratify_icd"),
                "icd_axis": numerator.support.get("icd_axis"),
            })
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
