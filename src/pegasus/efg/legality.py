"""Registry-aware Delta legality predicate for EFG graph expansion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import AlignmentResult, DeltaResult, FailedBranch, FieldNode
from pegasus.efg.declaration import OperatorSpec, evaluate_declaration_compatibility
from pegasus.efg.operators import EFGOperator, get_operator, ratio_rule
from pegasus.registries.aggregation import load_aggregation_registry
from pegasus.registries.carrier import load_carrier_registry
from pegasus.registries.unit import load_unit_registry


def _registry_root(registries: Any) -> Path:
    if isinstance(registries, (str, Path)):
        return Path(registries)
    if isinstance(registries, dict) and registries.get("registry_root"):
        return Path(registries["registry_root"])
    return Path("config/registries")


def _append(failed: list[str], term: str) -> None:
    if term not in failed:
        failed.append(term)


def _known_unit(unit: str, registry_root: Path) -> bool:
    if unit in {"ICD10", "rate", "proportion", "reais_hospital_services",
                "reais_professional_services", "reais_icu_services",
                "reais_total_billing"}:
        return True
    return unit in load_unit_registry(registry_root)


def _known_carrier(carrier: str, registry_root: Path) -> bool:
    # Carrier knowledge is fully registry-driven: all event/derived carriers are
    # declared in ontology/carrier.yaml (see health/clinical_event_definitions.yaml for the
    # restricted ones). No hardcoded carrier set in the engine.
    return carrier in load_carrier_registry(registry_root)


def _quality_allowed(field: FieldNode, operator: OperatorSpec) -> bool:
    if field.materialization_state.value in {"blocked", "failed"}:
        return False
    if field.state.value in {"illegal_excluded", "quarantined_nochildren"}:
        return False
    if field.state.value == "quarantined_descriptive":
        return operator.name in {
            EFGOperator.OBSERVER_FIELD.value,
            EFGOperator.DIAGNOSTIC_TOPOLOGY.value,
        }
    return True


def _provenance_allowed(field: FieldNode, operator: OperatorSpec) -> bool:
    provenance = set(field.provenance)
    if "illegal_excluded" in provenance or "zero_variance_excluded" in provenance:
        return False
    if "observer_rerouted" in provenance and operator.role in {"outcome", "mortality_rate"}:
        return False
    if operator.params.get("require_materialized_external") and (
        "fixture" in provenance or "synthetic" in provenance
    ):
        return False
    return True


def _specialized_semantics_ok(field: FieldNode) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    roles = set(field.role)
    if field.carrier in {"Beds", "GenericBeds"}:
        return False, ["generic_cnes_capacity_forbidden"]
    if "cnes_capacity_component" in roles:
        capacity_axis = field.axes.get("capacity_index") or field.axes.get("capacity_family")
        if not capacity_axis:
            return False, ["cnes_capacity_index_missing"]
    if "sih_cost_component" in roles:
        component = field.axes.get("cost_component")
        if component not in {"VAL_SH", "VAL_SP", "VAL_UTI", "VAL_TOT"}:
            return False, ["sih_cost_component_missing"]
    if field.unit == "ICD10" or "diagnostic_topology" in roles:
        warnings.append("diagnostic_topology_non_additive")
    return True, warnings


def _high_dimensional_axes_ok(field: FieldNode) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    support = field.support or {}
    axes = field.axes or {}
    source = {str(value).upper() for value in field.source or []}
    is_sidra = "SIDRA" in source or str(support.get("source_system") or "").upper() == "SIDRA"
    high_dimensional = bool(
        support.get("high_dimensional")
        or axes.get("high_dimensional")
        or support.get("estimated_cells_raw", 0) and int(support.get("estimated_cells_raw") or 0) > 49900
    )
    if not is_sidra or not high_dimensional:
        return True, warnings
    bound = support.get("high_dimensional_bound")
    if not isinstance(bound, dict):
        return False, ["high_dimensional_sidra_missing_bounded_pushforward"]
    status = str(bound.get("status") or "")
    if status not in {"bounded", "not_required"}:
        return False, [f"high_dimensional_sidra_unbounded:{status or 'missing_status'}"]
    if status == "bounded":
        warnings.append("high_dimensional_bounded_pushforward")
    return True, warnings




def _registry_root_for_evidence(registries: Any) -> str:
    if isinstance(registries, (str, Path)):
        return str(registries)
    if isinstance(registries, dict) and registries.get("registry_root"):
        return str(registries["registry_root"])
    root = getattr(registries, "root", None)
    if root is not None:
        return str(root)
    return "config/registries"



def _append_registry_evidence(
    parents: list[FieldNode],
    operator: OperatorSpec,
    warnings: list[str],
    registries: Any = None,
) -> list[str]:
    """Attach compact registry evidence without changing DeltaResult schema."""
    del operator
    merged = list(warnings or [])
    root = _registry_root_for_evidence(registries)
    evidence: list[str] = []
    try:
        from pegasus.registries.diagnostic_topology import diagnostic_evidence
        from pegasus.registries.cnes_capacity import capacity_evidence
        from pegasus.registries.sih_cost import cost_evidence
    except Exception:
        return merged
    for parent in parents or []:
        diagnostic = diagnostic_evidence(parent, registry_root=root)
        if diagnostic:
            evidence.append(f"diagnostic_topology={diagnostic.get('entry_id')}")
        capacity = capacity_evidence(parent, registry_root=root)
        if capacity:
            evidence.append(f"cnes_capacity={capacity.get('entry_id')}")
        cost = cost_evidence(parent, registry_root=root)
        if cost:
            evidence.append(f"sih_cost={cost.get('entry_id')}")
    if evidence:
        if "registry_evidence_attached" not in merged:
            merged.append("registry_evidence_attached")
        for item in evidence:
            token = f"registry_evidence:{item}"
            if token not in merged:
                merged.append(token)
    return merged

def evaluate_delta(
    *,
    parents: list[FieldNode],
    operator: OperatorSpec,
    intent: Any = None,
    registries: Any = None,
    alignment: AlignmentResult | None = None,
) -> DeltaResult:
    """Evaluate support, axes, carrier, unit, aggregation, provenance,
    quality, and declaration compatibility.
    """

    del intent
    registry_root = _registry_root(registries)
    get_operator(operator.name)
    numerator = parents[0] if parents else None
    denominator = parents[1] if len(parents) > 1 else None
    deltas = {
        "support": 1,
        "axes": 1,
        "carrier": 1,
        "unit": 1,
        "aggregation": 1,
        "provenance": 1,
        "quality": 1,
        "declaration": 1,
    }
    failed: list[str] = []
    warnings: list[str] = []

    if not parents:
        deltas["support"] = 0
        _append(failed, "support")
    if alignment is not None:
        warnings.extend(alignment.warnings)
        if not alignment.ok:
            deltas["support"] = 0
            deltas["axes"] = 0
            _append(failed, "support")
            _append(failed, "axes")

    for parent in parents:
        if not _known_carrier(parent.carrier, registry_root):
            deltas["carrier"] = 0
            _append(failed, "carrier")
            warnings.append(f"unknown_carrier:{parent.carrier}")
        if not _known_unit(parent.unit, registry_root):
            deltas["unit"] = 0
            _append(failed, "unit")
            warnings.append(f"unknown_unit:{parent.unit}")
        if parent.aggregation not in load_aggregation_registry(registry_root):
            deltas["aggregation"] = 0
            _append(failed, "aggregation")
        if not _quality_allowed(parent, operator):
            deltas["quality"] = 0
            _append(failed, "quality")
        if not _provenance_allowed(parent, operator):
            deltas["provenance"] = 0
            _append(failed, "provenance")
        specialized_ok, specialized_warnings = _specialized_semantics_ok(parent)
        warnings.extend(specialized_warnings)
        if not specialized_ok:
            deltas["carrier"] = 0
            _append(failed, "carrier")
        axes_ok, axes_warnings = _high_dimensional_axes_ok(parent)
        warnings.extend(axes_warnings)
        if not axes_ok:
            deltas["axes"] = 0
            _append(failed, "axes")

    if operator.name == EFGOperator.COUNT_MEASURE.value and parents:
        from pegasus.registries.events import primary_event_carriers

        carriers = {parent.carrier for parent in parents}
        artifacts = {parent.support.get("artifact_path") for parent in parents}
        if len(carriers) != 1 or len(artifacts) != 1:
            deltas["support"] = 0
            _append(failed, "support")
        # COUNT_MEASURE is only legal for source-countable primary event carriers,
        # as declared by the clinical event registry (no hardcoded carrier set).
        if next(iter(carriers)) not in primary_event_carriers(root=registry_root):
            deltas["carrier"] = 0
            _append(failed, "carrier")

    if operator.name == EFGOperator.PSI_FUNCTIONAL.value and parents:
        from pegasus.registries.events import primary_event_carriers

        carriers = {parent.carrier for parent in parents}
        artifacts = {parent.support.get("artifact_path") for parent in parents}
        if len(carriers) != 1 or len(artifacts) != 1:
            deltas["support"] = 0
            _append(failed, "support")
        if next(iter(carriers)) not in primary_event_carriers(root=registry_root):
            deltas["carrier"] = 0
            _append(failed, "carrier")

    if operator.name in {EFGOperator.PROJECT.value, EFGOperator.BOUNDED_PROJECT.value}:
        if len(parents) != 1 or parents[0].aggregation != "additive":
            deltas["aggregation"] = 0
            _append(failed, "aggregation")

    if operator.name == EFGOperator.RN.value:
        if numerator is None or denominator is None:
            deltas["support"] = 0
            _append(failed, "support")
        else:
            if (
                "source_field" in numerator.role
                and not set(numerator.role).intersection({
                    "source_event_count", "restricted_count", "numerator_measure",
                    "inpatient_death", "birth_outcome_indicator",
                })
            ):
                deltas["aggregation"] = 0
                _append(failed, "aggregation")
                warnings.append("raw_source_field_requires_measure_operator")
            if ratio_rule(numerator, denominator, operator.role, registry_root=registry_root) is None:
                deltas["carrier"] = 0
                deltas["unit"] = 0
                _append(failed, "carrier")
                _append(failed, "unit")
                warnings.append("unregistered_ratio_semantics")
            if numerator.aggregation != "additive" or denominator.aggregation != "additive":
                deltas["aggregation"] = 0
                _append(failed, "aggregation")
            if "diagnostic_topology" in numerator.role or numerator.unit == "ICD10":
                deltas["aggregation"] = 0
                _append(failed, "aggregation")
                warnings.append("diagnostic_observer_requires_restriction_before_ratio")

    if numerator is not None:
        declaration = evaluate_declaration_compatibility(
            numerator=numerator,
            denominator=denominator,
            operator=operator,
            registries=registries,
        )
        warnings.extend(declaration.warnings)
        if not declaration.ok:
            deltas["declaration"] = 0
            _append(failed, "declaration")

    legal = all(deltas.values())
    failed_id = None
    if not legal:
        failed_id = "failed_" + content_hash({
            "operator": operator.model_dump(mode="json"),
            "parents": [parent.id for parent in parents],
            "failed": failed,
        })[:24]
    result_warnings = _append_registry_evidence(
        parents,
        operator,
        warnings,
        registries=registries,
    )
    return DeltaResult(
        legal=legal,
        delta_support=deltas["support"],
        delta_axes=deltas["axes"],
        delta_carrier=deltas["carrier"],
        delta_unit=deltas["unit"],
        delta_aggregation=deltas["aggregation"],
        delta_provenance=deltas["provenance"],
        delta_quality=deltas["quality"],
        delta_declaration=deltas["declaration"],
        failed_terms=failed,
        warnings=list(dict.fromkeys(result_warnings)),
        failed_branch_id=failed_id,
    )


def make_failed_branch(
    *, parents: list[FieldNode], operator: OperatorSpec, delta: DeltaResult, reason: str,
) -> FailedBranch:
    """Compatibility adapter for earlier callers using the core schema."""

    from datetime import datetime, timezone

    stage = "declaration" if "declaration" in delta.failed_terms else (
        delta.failed_terms[0] if delta.failed_terms else "output_validation"
    )
    return FailedBranch(
        failed_branch_id=delta.failed_branch_id or "failed_unknown",
        attempted_operator=operator.name,
        parent_field_ids=[parent.id for parent in parents],
        failure_stage=stage,  # type: ignore[arg-type]
        failed_terms=delta.failed_terms,
        reason=reason,
        warnings=delta.warnings,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
