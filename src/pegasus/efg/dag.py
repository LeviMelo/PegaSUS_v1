"""Autonomous, source-agnostic Epidemiological Field Graph compiler core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.core.schemas import FieldNode, UserIntent
from pegasus.efg.align import align_fields
from pegasus.efg.bridges import bridge_summary, plan_bridge_candidates
from pegasus.efg.core_seed import build_core_seed_set, core_seed_summary
from pegasus.efg.declaration import OperatorSpec
from pegasus.efg.equivalence import PrecompressionReport, precompress_fields
from pegasus.efg.failed_branch import (
    FailedBranchRecord,
    failed_exclusion,
    make_failed_branch_record,
)
from pegasus.efg.legality import evaluate_delta
from pegasus.efg.lineage import lineage_hash, make_lineage
from pegasus.efg.materialize import materialize_substrate_bundle
from pegasus.efg.node import make_field_node
from pegasus.efg.operators import EFGOperator, apply_operator
from pegasus.she.substrate import SubstrateBundle


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EFGEdge:
    edge_id: str
    parent_field_id: str
    child_field_id: str
    operator: str
    operator_params: dict[str, Any]
    registry_versions: dict[str, str]
    created_at: str

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["operator_params_json"] = dict(self.operator_params)
        payload["registry_versions_json"] = dict(self.registry_versions)
        return payload


@dataclass(frozen=True)
class EFGResult:
    schema_version: str
    efg_id: str
    substrate_id: str
    fields: tuple[FieldNode, ...]
    edges: tuple[EFGEdge, ...]
    failed_branches: tuple[FailedBranchRecord, ...]
    warnings: tuple[str, ...]
    variable_dictionary: tuple[dict[str, Any], ...]
    precompression: PrecompressionReport
    source_hashes: tuple[str, ...]
    registry_hashes: dict[str, str]
    legality_summary: dict[str, int]
    operator_mode: str
    core_seed_summary: dict[str, Any] = field(default_factory=dict)
    bridge_plan_summary: dict[str, Any] = field(default_factory=dict)
    domain_summaries: dict[str, Any] = field(default_factory=dict)

    @property
    def field_count(self) -> int:
        return len(self.fields)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_manifest_schema": "28Y.1",
            "efg_id": self.efg_id,
            "substrate_id": self.substrate_id,
            "field_count": self.field_count,
            "edge_count": self.edge_count,
            "failed_branch_count": len(self.failed_branches),
            "fields": [
                {
                    "field": field.model_dump(mode="json"),
                    "lineage_hash": lineage_hash(field.lineage),
                    "materialization_reason": "autonomous_efg_core",
                }
                for field in self.fields
            ],
            "edges": [edge.as_manifest() for edge in self.edges],
            "failed_branches": [branch.as_manifest() for branch in self.failed_branches],
            "warnings": list(self.warnings),
            "variable_dictionary": list(self.variable_dictionary),
            "precompression": self.precompression.as_manifest(),
            "source_hashes": list(self.source_hashes),
            "registry_hashes": dict(self.registry_hashes),
            "legality_summary": dict(self.legality_summary),
            "operator_mode": self.operator_mode,
            "core_seed_summary": dict(self.core_seed_summary),
            "bridge_plan_summary": dict(self.bridge_plan_summary),
            "domain_summaries": dict(self.domain_summaries),
        }


def _empty_core_seed_summary(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    return {
        "registry_root": str(registry_root),
        "seed_count": 0,
        "blocked_count": 0,
        "role_counts": {},
        "seeds": [],
        "blocked": [],
    }


def _empty_bridge_plan_summary() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "blocked_count": 0,
        "bridge_type_counts": {},
        "candidates": [],
        "blocked": [],
    }


def _edge(
    parent: FieldNode,
    child: FieldNode,
    operator: OperatorSpec,
    *,
    edge_operator: str | None = None,
    edge_params: dict[str, Any] | None = None,
) -> EFGEdge:
    operator_name = edge_operator or operator.name
    params = {**operator.params, **(edge_params or {})}
    payload = {
        "parent": parent.id,
        "child": child.id,
        "operator": operator_name,
        "params": params,
    }
    versions = dict(child.lineage.registry_versions)
    return EFGEdge(
        edge_id=f"edge_{content_hash(payload)[:24]}",
        parent_field_id=parent.id,
        child_field_id=child.id,
        operator=operator_name,
        operator_params=params,
        registry_versions=versions,
        created_at=_now(),
    )


def _registry_hashes(substrate: SubstrateBundle, root: Path) -> dict[str, str]:
    hashes = dict(substrate.registry_hashes)
    for name in (
        "source_fields.yaml", "carrier.yaml", "unit.yaml", "aggregation.yaml",
        "provenance.yaml", "quality.yaml", "quality_permissions.yaml",
        "cnes_capacity_registry.yaml", "sih_cost_registry.yaml", "diagnostic_topology.yaml",
    ):
        path = root / name
        if path.exists():
            hashes[name] = sha256_file(path)
    hashes["efg_operator_registry"] = content_hash({
        "version": "slice19a",
        "operators": [operator.value for operator in EFGOperator],
    })
    return hashes


def _source_hashes(substrate: SubstrateBundle) -> tuple[str, ...]:
    return tuple(sorted({
        value
        for artifact in substrate.source_artifacts
        for value in (artifact.source_manifest_hash, artifact.artifact_hash)
        if value
    }))


def _dictionary(field: FieldNode) -> dict[str, Any]:
    diagnostic_role = field.axes.get("icd_topology_role") or field.axes.get("diagnostic_role")
    return {
        "field_id": field.id,
        "display_name": field.name,
        "technical_name": field.name,
        "definition": f"EFG field generated by {field.operator or 'substrate admission'}.",
        "estimand_label": next((role for role in field.role if role.endswith("rate")), field.kind),
        "source_systems": list(field.source),
        "carrier": field.carrier,
        "unit": field.unit,
        "support_description": dict(field.support),
        "axis_description": dict(field.axes),
        "provenance_description": list(field.provenance),
        "state": field.state.value,
        "dashboard_safe": field.dashboard_safe,
        "interpretation_warning": ";".join(field.warnings) or None,
        "diagnostic_role": diagnostic_role,
        "topology": diagnostic_role,
        "position": field.axes.get("position"),
        "capacity_index": field.axes.get("capacity_index") or field.axes.get("capacity_family"),
        "cost_component": field.axes.get("cost_component"),
    }


def _event_groups(fields: Iterable[FieldNode]) -> dict[tuple[str, str], list[FieldNode]]:
    groups: dict[tuple[str, str], list[FieldNode]] = {}
    for field in fields:
        artifact = field.support.get("artifact_path")
        if not artifact or field.carrier not in {"Deaths", "HospitalAdmissions", "LiveBirths", "Facilities"}:
            continue
        if field.unit == "ICD10" or "diagnostic_topology" in field.role:
            continue
        groups.setdefault((str(artifact), field.carrier), []).append(field)
    return groups


def _ratio_role(numerator: FieldNode, denominator: FieldNode) -> str:
    roles = {
        ("Deaths", "Population"): "mortality_rate",
        ("HospitalAdmissions", "Population"): "hospitalization_rate",
        ("LiveBirths", "Population"): "birth_rate",
        ("HospitalDeaths", "HospitalAdmissions"): "inpatient_mortality",
        ("InfantDeaths", "LiveBirths"): "infant_mortality",
        ("LowBirthWeightBirths", "LiveBirths"): "birth_outcome_share",
    }
    return roles.get((numerator.carrier, denominator.carrier), "unsupported_ratio")


def _find_field(fields: list[FieldNode], selector: str) -> FieldNode | None:
    return next((field for field in fields if field.id == selector or field.name == selector), None)


def _race_bridge_source(fields: Iterable[FieldNode], plan: dict[str, Any]) -> FieldNode | None:
    source_system = str(plan.get("source_system") or "SIM-DO")
    for field in fields:
        roles = set(field.role or [])
        if source_system not in set(field.source or []):
            continue
        if "administrative_race_axis" in roles or field.support.get("column") == "race_color_admin":
            return field
    return None


def _append_race_bridge_fields(
    *,
    fields: list[FieldNode],
    edges: list[EFGEdge],
    constraints: dict[str, Any],
    warnings: list[str],
) -> None:
    plan = constraints.get("race_bridge_plan")
    if not isinstance(plan, dict) or plan.get("status") != "planned":
        return
    parent = _race_bridge_source(fields, plan)
    if parent is None:
        warnings.append("race_bridge_source_axis_not_found")
        return
    prior_path = plan.get("prior_path")
    prior_hash = plan.get("prior_hash")
    params = {
        "bridge_id": plan.get("bridge_id"),
        "bridge_operator": "Bridge_R_fixedC_dynamic_weight",
        "bridge_mode": plan.get("mode") or "fixedC_dynamic_weight",
        "source_axis": plan.get("source_axis"),
        "target_axis": plan.get("target_axis"),
        "prior_path": prior_path,
        "prior_hash": prior_hash,
        "raw_admin_counts_preserved": True,
        "missing_category_preserved": True,
    }
    support = {
        **dict(parent.support),
        **params,
        "missing_race_share": 0.0,
        "race_bridge_cv": 0.0,
        "sensitivity_width": 0.0,
        "emission_prior_strength": 1.0,
    }
    axes = {
        **dict(parent.axes),
        "numerator_axis_source": plan.get("source_axis"),
        "denominator_axis_target": plan.get("target_axis"),
        "bridge_operator": "Bridge_R_fixedC_dynamic_weight",
        "emission_matrix_registry_version": prior_hash,
        "bridge_mode": plan.get("mode") or "fixedC_dynamic_weight",
        "missing_race_share": 0.0,
        "race_bridge_cv": 0.0,
        "sensitivity_width": 0.0,
        "race_axis_warning": "administrative_race_not_self_declared",
        "bayesian_ecological_bridge_warning": "fixed_C_dynamic_weight_posterior",
        "prior_hash": prior_hash,
        "lower_count": 0.0,
        "upper_count": 0.0,
        "race_axis_target": plan.get("target_axis"),
    }
    lineage = make_lineage(
        parent_ids=[parent.id],
        operator_type="Bridge_R_fixedC_dynamic_weight",
        operator_params=params,
        registry_versions={
            **dict(parent.lineage.registry_versions),
            "race_bridge_prior": str(prior_hash or "unknown"),
        },
        source_manifest_hashes=list(parent.lineage.source_manifest_hashes),
        code_version="slice28y_race_bridge_executor",
    )
    field = make_field_node(
        name=f"SIM race bridge posterior count {plan.get('bridge_id') or 'fixedC'}",
        kind="bridge_module",
        carrier=parent.carrier,
        unit="counts",
        support=support,
        axes=axes,
        aggregation="additive",
        role=["race_bridge_posterior", "self_aligned_race_estimate", "source_field"],
        source=list(dict.fromkeys([*list(parent.source or []), "RaceBridgePrior", str(prior_path or "")])),
        operator="Bridge_R_fixedC_dynamic_weight",
        provenance=list(dict.fromkeys([*list(parent.provenance or []), "BayesianEcologicalRaceBridge"])),
        state="warning",
        warnings=["race_bridge_posterior_not_raw_epidemiological_observation"],
        lineage=lineage,
        materialization_state="metadata_only",
        path=None,
        dashboard_safe="warning",
    ).model_copy(update={"id": f"SIMRaceBridgePosteriorCount_{lineage_hash(lineage)[:24]}"})
    fields.append(field)
    operator = OperatorSpec(
        name="Bridge_R_fixedC_dynamic_weight",
        role="race_bridge_posterior",
        output_kind="bridge_module",
        params=params,
    )
    edges.append(_edge(parent, field, operator))


def _field_domain_summaries(fields: Iterable[FieldNode], constraints: dict[str, Any]) -> dict[str, Any]:
    field_list = list(fields)
    summaries: dict[str, Any] = {}
    plan = constraints.get("race_bridge_plan")
    race_fields = [field for field in field_list if str(field.id).startswith("SIMRaceBridgePosteriorCount_")]
    if isinstance(plan, dict) and (plan.get("status") == "planned" or race_fields):
        race_support = race_fields[0].support if race_fields else {}
        summaries["race_bridge"] = {
            "bridge_id": plan.get("bridge_id") if isinstance(plan, dict) else race_support.get("bridge_id"),
            "mode": (plan.get("mode") if isinstance(plan, dict) else race_support.get("bridge_mode")) or "fixedC_dynamic_weight",
            "prior_hash": plan.get("prior_hash") if isinstance(plan, dict) else race_support.get("prior_hash"),
            "source_axis": plan.get("source_axis") if isinstance(plan, dict) else race_support.get("source_axis"),
            "target_axis": plan.get("target_axis") if isinstance(plan, dict) else race_support.get("target_axis"),
            "missing_race_share": float(race_support.get("missing_race_share") or 0.0),
            "sensitivity_width": float(race_support.get("sensitivity_width") or 0.0),
            "race_bridge_cv": float(race_support.get("race_bridge_cv") or 0.0),
            "raw_admin_counts_preserved": True,
            "missing_category_preserved": True,
            "attach_stage": "efg_executor",
            "field_ids": [field.id for field in race_fields],
        }
    population_fields = [field for field in field_list if str(field.id).startswith("population_tensor_")]
    if population_fields:
        first = population_fields[0]
        summaries["population_tensor"] = {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "standalone_population_tensor",
            "field_id": first.id,
            "tensor_id": first.id,
            "mode": "official_sidra_anchor",
            "solver_id": "sidra_9606_total_anchor",
            "solver_backend": "official_sidra_anchor",
            "denominator_feedback_warning": None,
            "independent_denominator_mode": True,
            "sim_feedback_warning": None,
            "source_hashes": list(first.lineage.source_manifest_hashes),
        }
    cnes_fields = [field for field in field_list if "CNES-ST" in set(field.source or [])]
    sih_fields = [field for field in field_list if "SIH-RD" in set(field.source or [])]
    if cnes_fields or sih_fields:
        summaries["cnes_sih"] = {
            "schema_version": "1.0",
            "source_systems": ["CNES-ST", "SIH-RD"],
            "attach_stage": "she_build",
            "cnes": {
                "field_count": len(cnes_fields),
                "generic_beds_blocked": True,
            },
            "sih": {
                "field_count": len(sih_fields),
                "generic_sih_cost_blocked": True,
                "diagnostic_topology_preserved": True,
            },
        }
    return summaries


def _dedupe_edges(edges: Iterable[EFGEdge], canonical: dict[str, str]) -> tuple[EFGEdge, ...]:
    output: dict[tuple[str, str, str], EFGEdge] = {}
    for item in edges:
        parent = canonical.get(item.parent_field_id, item.parent_field_id)
        child = canonical.get(item.child_field_id, item.child_field_id)
        if parent == child:
            continue
        key = (parent, child, item.operator)
        if key not in output:
            output[key] = EFGEdge(
                edge_id=f"edge_{content_hash(key)[:24]}",
                parent_field_id=parent,
                child_field_id=child,
                operator=item.operator,
                operator_params=item.operator_params,
                registry_versions=item.registry_versions,
                created_at=item.created_at,
            )
    return tuple(output.values())


def _build_efg_base(
    *,
    substrate: SubstrateBundle,
    registry_root: str | Path = "config/registries",
    intent: UserIntent | dict[str, Any] | None = None,
    registries: Any = None,
    compute: Any = None,
    intent_constraints: dict[str, Any] | None = None,
    operator_budget: int = 256,
    operator_mode: str = "standard",
) -> EFGResult:
    """Build the metadata-level autonomous field DAG from a SHE substrate.

    The result is directly serializable and its ``fields`` manifest shape is
    accepted by the existing materialization/promotion planning boundary.
    """

    del compute
    root = Path(registry_root)
    registry_arg = registries or {"registry_root": str(root)}
    constraints = dict(intent_constraints or {})
    materialized = materialize_substrate_bundle(substrate)
    roots, root_compression = precompress_fields(item.field for item in materialized.fields)
    fields: list[FieldNode] = list(roots)
    edges: list[EFGEdge] = []
    failures: list[FailedBranchRecord] = [
        failed_exclusion(exclusion=item)
        for item in materialized.excluded_source_fields
    ]
    warnings = list(dict.fromkeys([*substrate.warnings, *materialized.warnings]))
    legality = {"attempted": 0, "legal": 0, "illegal": len(failures), "blocked": 0}
    expansions = 0

    def expand(operator: OperatorSpec, parents: list[FieldNode], alignment=None) -> FieldNode | None:
        nonlocal expansions
        legality["attempted"] += 1
        if expansions >= operator_budget:
            failures.append(make_failed_branch_record(
                parents=parents,
                operator=operator,
                reason="operator_budget_exhausted",
                disposition="deferred",
                warnings=["operator_budget_exhausted"],
            ))
            legality["blocked"] += 1
            return None
        expansions += 1
        try:
            delta = evaluate_delta(
                parents=parents,
                operator=operator,
                intent=intent,
                registries=registry_arg,
                alignment=alignment,
            )
        except ValueError as exc:
            failures.append(make_failed_branch_record(
                parents=parents,
                operator=operator,
                reason=str(exc),
                disposition="unsupported",
            ))
            legality["illegal"] += 1
            return None
        if not delta.legal:
            failures.append(make_failed_branch_record(
                parents=parents,
                operator=operator,
                delta=delta,
                alignment=alignment,
                reason=(alignment.failure_reason if alignment and alignment.failure_reason else
                        ";".join(delta.failed_terms) or "delta_rejected"),
            ))
            legality["illegal"] += 1
            return None
        result, field = apply_operator(
            operator=operator,
            parents=parents,
            delta=delta,
            alignment=alignment,
            registries=registry_arg,
        )
        if result.status != "success" or field is None:
            failures.append(make_failed_branch_record(
                parents=parents,
                operator=operator,
                delta=delta,
                alignment=alignment,
                reason="operator_application_blocked",
                disposition="blocked",
                warnings=result.warnings,
            ))
            legality["blocked"] += 1
            return None
        fields.append(field)
        if operator.name == EFGOperator.RN.value and len(parents) == 2:
            edges.append(_edge(
                parents[0], field, operator, edge_params={"input_role": "numerator"},
            ))
            edges.append(_edge(
                parents[1],
                field,
                operator,
                edge_operator=EFGOperator.DENOMINATOR_LINK.value,
                edge_params={"input_role": "denominator", "ratio_operator": "RN"},
            ))
        else:
            edges.extend(_edge(parent, field, operator) for parent in parents)
        legality["legal"] += 1
        return field

    count_nodes: list[FieldNode] = []
    for (artifact, carrier), parents in sorted(_event_groups(roots).items()):
        count_parents = [
            parent for parent in parents
            if {"geography", "time", "period"}.intersection(parent.axes)
            or {"geography_axis", "time_axis_candidate"}.intersection(parent.role)
        ] or [parents[0]]
        operator = OperatorSpec(
            name=EFGOperator.COUNT_MEASURE.value,
            role="event_count",
            output_kind="extensive_measure",
            params={
                "artifact_path": artifact,
                "carrier": carrier,
                "name": f"{count_parents[0].source[0]}.{Path(artifact).stem}.count",
            },
        )
        child = expand(operator, count_parents)
        if child is not None:
            count_nodes.append(child)

    denominators = [field for field in roots if field.carrier == "Population"]
    if operator_mode != "raw_only":
        for numerator in count_nodes:
            for denominator in denominators:
                role = _ratio_role(numerator, denominator)
                operator = OperatorSpec(
                    name=EFGOperator.RN.value,
                    role=role,
                    output_kind="intensive_density",
                    params={"ratio_role": role},
                )
                alignment = align_fields(
                    left=numerator,
                    right=denominator,
                    operator=operator,
                    intent=intent,
                    registries=registry_arg,
                )
                expand(operator, [numerator, denominator], alignment)

    bridge_plan = None
    if operator_mode != "raw_only":
        bridge_plan = plan_bridge_candidates(fields, registry_root=root, intent=intent)
        by_id = {field.id: field for field in fields}
        for candidate in bridge_plan.candidates:
            parents = [by_id.get(candidate.numerator_id)]
            if candidate.denominator_id:
                parents.append(by_id.get(candidate.denominator_id))
            parents = [parent for parent in parents if parent is not None]
            operator_name = EFGOperator.RN.value if str(candidate.required_operator).lower() == "ratio" else str(candidate.required_operator)
            operator_role = _ratio_role(parents[0], parents[1]) if operator_name == EFGOperator.RN.value and len(parents) == 2 else str(candidate.bridge_type)
            operator = OperatorSpec(
                name=operator_name,
                role=operator_role,
                output_kind="bridge_module" if not candidate.denominator_id else "intensive_density",
                params={
                    "bridge_id": candidate.bridge_id,
                    "bridge_type": candidate.bridge_type,
                    "support_relation": candidate.support_relation,
                    "registry_evidence": list(candidate.registry_evidence),
                },
            )
            if not parents:
                failures.append(make_failed_branch_record(
                    parents=[],
                    operator=operator,
                    reason="bridge_candidate_parent_missing",
                    disposition="blocked",
                    warnings=list(candidate.warnings),
                ))
                legality["blocked"] += 1
                continue
            alignment = None
            if len(parents) == 2:
                alignment = align_fields(
                    left=parents[0],
                    right=parents[1],
                    operator=operator,
                    intent=intent,
                    registries=registry_arg,
                )
            child = expand(operator, parents, alignment)
            if child is not None:
                by_id[child.id] = child

    _append_race_bridge_fields(fields=fields, edges=edges, constraints=constraints, warnings=warnings)

    for request in constraints.get("ratio_requests", []):
        numerator = _find_field(fields, str(request.get("numerator", "")))
        denominator = _find_field(fields, str(request.get("denominator", "")))
        operator = OperatorSpec(
            name=str(request.get("operator", EFGOperator.RN.value)),
            role=str(request.get("role", "unsupported_ratio")),
            params=dict(request.get("params", {})),
        )
        if numerator is None or denominator is None:
            failures.append(make_failed_branch_record(
                parents=[field for field in (numerator, denominator) if field is not None],
                operator=operator,
                reason="requested_input_field_not_found",
                disposition="unsupported",
            ))
            legality["illegal"] += 1
            continue
        alignment = align_fields(
            left=numerator,
            right=denominator,
            operator=operator,
            intent=intent,
            registries=registry_arg,
        )
        expand(operator, [numerator, denominator], alignment)

    compressed, final_compression = precompress_fields(fields)
    combined_report = PrecompressionReport(
        input_count=root_compression.input_count + max(0, len(fields) - len(roots)),
        output_count=len(compressed),
        suppressed=tuple([*root_compression.suppressed, *final_compression.suppressed]),
        canonical_by_field_id={
            **root_compression.canonical_by_field_id,
            **final_compression.canonical_by_field_id,
        },
        protected_non_equivalences=tuple([
            *root_compression.protected_non_equivalences,
            *final_compression.protected_non_equivalences,
        ]),
    )
    final_edges = _dedupe_edges(edges, combined_report.canonical_by_field_id)
    registry_hashes = _registry_hashes(substrate, root)
    source_hashes = _source_hashes(substrate)
    try:
        seed_set = build_core_seed_set(compressed, registry_root=root, intent=intent)
        core_summary = core_seed_summary(seed_set)
    except Exception as exc:  # pragma: no cover - defensive manifest metadata
        core_summary = {
            **_empty_core_seed_summary(root),
            "blocked": [{"reason": "core_seed_summary_failed", "error": str(exc)}],
            "blocked_count": 1,
        }
    if bridge_plan is None:
        bridge_summary_payload = _empty_bridge_plan_summary()
    else:
        bridge_summary_payload = bridge_summary(bridge_plan)
    domain_summary_payload = _field_domain_summaries(compressed, constraints)
    payload = {
        "substrate_id": substrate.substrate_id,
        "field_ids": [field.id for field in compressed],
        "edge_ids": [edge.edge_id for edge in final_edges],
        "failed_ids": [branch.failed_branch_id for branch in failures],
        "registry_hashes": registry_hashes,
        "core_seed_summary": core_summary,
        "bridge_plan_summary": bridge_summary_payload,
        "domain_summaries": domain_summary_payload,
    }
    return EFGResult(
        schema_version="19A.1",
        efg_id=f"efg_{content_hash(payload)[:24]}",
        substrate_id=substrate.substrate_id,
        fields=compressed,
        edges=final_edges,
        failed_branches=tuple(failures),
        warnings=tuple(dict.fromkeys(warnings)),
        variable_dictionary=tuple(_dictionary(field) for field in compressed),
        precompression=combined_report,
        source_hashes=source_hashes,
        registry_hashes=registry_hashes,
        legality_summary=legality,
        operator_mode=operator_mode,
        core_seed_summary=core_summary,
        bridge_plan_summary=bridge_summary_payload,
        domain_summaries=domain_summary_payload,
    )

def build_efg(*args, **kwargs):
    return _build_efg_base(*args, **kwargs)

