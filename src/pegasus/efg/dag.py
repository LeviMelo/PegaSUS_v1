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
from pegasus.efg.core_seed_registry import core_seed_specs, enforce_mandatory_fields, resolve_core_seeds
from pegasus.efg.diagnostic_strata import (
    ICD_AXIS_BY_LEVEL,
    diagnostic_columns_by_event,
    enforce_health_seeds,
    requested_icd_levels,
)
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


def _event_groups(
    fields: Iterable[FieldNode], *, registry_root: str | Path = "config/registries"
) -> dict[tuple[str, str], list[FieldNode]]:
    from pegasus.registries.events import primary_event_carriers

    countable = primary_event_carriers(root=registry_root)
    groups: dict[tuple[str, str], list[FieldNode]] = {}
    for field in fields:
        artifact = field.support.get("artifact_path")
        carrier = field.carrier
        if carrier == "FacilityCapacityVector" and "Facilities" in countable and "cnes_capacity_component" in set(field.role or []):
            carrier = "Facilities"
        if not artifact or carrier not in countable:
            continue
        if field.unit == "ICD10" or "diagnostic_topology" in field.role:
            continue
        groups.setdefault((str(artifact), carrier), []).append(field)
    return groups


def _ratio_role(
    numerator: FieldNode, denominator: FieldNode, *, registry_root: str | Path = "config/registries"
) -> str:
    """Resolve the estimand label for a (numerator, denominator) carrier pair.

    Driven by the clinical event registry (MSD §2.6) — the engine holds no
    hardcoded carrier→role table.
    """
    from pegasus.registries.events import clinical_ratio_specs

    for spec in clinical_ratio_specs(root=registry_root):
        if spec.numerator_carrier == numerator.carrier and spec.denominator_carrier == denominator.carrier:
            return spec.role
    return "unsupported_ratio"


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
        "bridge_operator": "Bridge_R_localPi_posteriorC",
        "bridge_mode": plan.get("mode") or "localPi_posterior",
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
        "bridge_operator": "Bridge_R_localPi_posteriorC",
        "emission_matrix_registry_version": prior_hash,
        "bridge_mode": plan.get("mode") or "localPi_posterior",
        "missing_race_share": 0.0,
        "race_bridge_cv": 0.0,
        "sensitivity_width": 0.0,
        "race_axis_warning": "administrative_race_not_self_declared",
        "bayesian_ecological_bridge_warning": "local_pi_posterior_crosswalk",
        "prior_hash": prior_hash,
        "lower_count": 0.0,
        "upper_count": 0.0,
        "race_axis_target": plan.get("target_axis"),
    }
    lineage = make_lineage(
        parent_ids=[parent.id],
        operator_type="Bridge_R_localPi_posteriorC",
        operator_params=params,
        registry_versions={
            **dict(parent.lineage.registry_versions),
            "race_bridge_prior": str(prior_hash or "unknown"),
        },
        source_manifest_hashes=list(parent.lineage.source_manifest_hashes),
        code_version="slice28y_race_bridge_executor",
    )
    field = make_field_node(
        name=f"SIM race bridge posterior count {plan.get('bridge_id') or 'localPi'}",
        kind="bridge_module",
        carrier=parent.carrier,
        unit="counts",
        support=support,
        axes=axes,
        aggregation="additive",
        role=["race_bridge_posterior", "self_aligned_race_estimate", "source_field"],
        source=list(dict.fromkeys([*list(parent.source or []), "RaceBridgePrior", str(prior_path or "")])),
        operator="Bridge_R_localPi_posteriorC",
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
        name="Bridge_R_localPi_posteriorC",
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
            "mode": (plan.get("mode") if isinstance(plan, dict) else race_support.get("bridge_mode")) or "localPi_posteriorC",
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
        solver_fields = [field for field in population_fields if field.operator == "population_tensor_solver"]
        first = solver_fields[0] if solver_fields else population_fields[0]
        support = dict(first.support or {})
        mode = str(support.get("PopulationTensorMode") or "official_sidra_anchor")
        solver_id = str(support.get("SolverID") or "sidra_9606_total_anchor")
        solver_backend = "population_tensor_solver" if first.operator == "population_tensor_solver" else "official_sidra_anchor"
        summaries["population_tensor"] = {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "population_solver" if first.operator == "population_tensor_solver" else "standalone_population_tensor",
            "field_id": first.id,
            "tensor_id": first.id,
            "mode": mode,
            "solver_id": solver_id,
            "solver_backend": solver_backend,
            "denominator_feedback_warning": bool("sim_informed_population_feedback_risk" in set(first.warnings or [])) or None,
            "independent_denominator_mode": mode in {"official_sidra_anchor", "independent_denominator"},
            "sim_feedback_warning": "sim_informed_population_feedback_risk" if mode == "sim_informed_denominator" else None,
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
    sidra_context_fields = [field for field in field_list if field.operator == "sidra_context_field"]
    if sidra_context_fields:
        stdfm_fields = [field for field in sidra_context_fields if isinstance(field.support, dict) and field.support.get("stdfm") is not None]
        latent_fields = [field for field in sidra_context_fields if field.kind == "latent_context"]
        bound_fields = [
            field
            for field in sidra_context_fields
            if ((field.support or {}).get("high_dimensional_bound") or {}).get("status") == "bounded"
        ]
        summaries["sidra_context"] = {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "she_build",
            "field_count": len(sidra_context_fields),
            "context_gradient_count": len([field for field in sidra_context_fields if field.kind == "context_gradient"]),
            "latent_context_count": len(latent_fields),
            "stdfm_executed_count": len(stdfm_fields),
            "stdfm_certified_count": len([
                field for field in stdfm_fields
                if (((field.support or {}).get("stdfm") or {}).get("certification") or {}).get("status") in {"verified", "fragile"}
            ]),
            "high_dimensional_bounded_count": len(bound_fields),
            "projection_matrix_ids": sorted({
                str((field.support or {}).get("projection", {}).get("projection_matrix_id") or "")
                for field in sidra_context_fields
                if (field.support or {}).get("projection", {}).get("projection_matrix_id")
            }),
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

    from pegasus.registries.demographic_axis import DEMOGRAPHIC_AXES
    from pegasus.registries.demographic_axis import source_column as _demo_source_column

    def _demographic_axes(node: FieldNode) -> frozenset[str]:
        return frozenset(set(node.axes) & DEMOGRAPHIC_AXES)

    # Demographic axes that have a population denominator tensor admitted (e.g. {sex});
    # only these support demographically-stratified rates (matched denominator exists).
    available_demographic_axes: set[str] = set()
    for node in roots:
        if "demographic_stratified" in set(node.role or []):
            available_demographic_axes |= (set(node.axes) & DEMOGRAPHIC_AXES)

    count_nodes: list[FieldNode] = []
    strata_levels = requested_icd_levels(intent)
    diagnostic_columns = diagnostic_columns_by_event(roots)
    # Geo/time parent context per primary event carrier, reused to build σ-restricted
    # clinical events (MSD §2.6/§3.10.4) from the same source artifact.
    primary_event_context: dict[str, tuple[str, list[FieldNode]]] = {}
    for (artifact, carrier), parents in sorted(_event_groups(roots, registry_root=root).items()):
        count_parents = [
            parent for parent in parents
            if {"geography", "time", "period"}.intersection(parent.axes)
            or {"geography_axis", "time_axis_candidate"}.intersection(parent.role)
        ] or [parents[0]]
        primary_event_context.setdefault(carrier, (artifact, count_parents))
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

        # ICD diagnostic traversal (MSD §3.11): build cause-specific event counts by
        # σ_C restriction of the primary diagnostic observer for this (artifact, carrier).
        # Each becomes a legal additive measure that the RN loop below divides by the
        # population denominator to yield cause-specific mortality / hospitalization.
        icd_column = diagnostic_columns.get((artifact, carrier))
        if icd_column and strata_levels:
            for level, _seed in strata_levels:
                axis_name = ICD_AXIS_BY_LEVEL[level]
                strat_operator = OperatorSpec(
                    name=EFGOperator.COUNT_MEASURE.value,
                    role="cause_specific_event_count",
                    output_kind="extensive_measure",
                    params={
                        "artifact_path": artifact,
                        "carrier": carrier,
                        "stratify_icd": level,
                        "icd_column": icd_column,
                        "icd_axis": axis_name,
                        "icd_source_field": icd_column,
                        "name": f"{count_parents[0].source[0]}.{Path(artifact).stem}.count.{axis_name}",
                    },
                )
                strat_child = expand(strat_operator, count_parents)
                if strat_child is not None:
                    count_nodes.append(strat_child)

        # Demographic stratification (MSD §2.8/§3.7.4): stratify this event count by each
        # demographic axis (sex/age/race) that has a matching population denominator tensor,
        # mapping source category codes to the canonical axis. Enables stratified rates.
        source_system = count_parents[0].source[0] if count_parents[0].source else ""
        for axis_name in sorted(available_demographic_axes):
            column = _demo_source_column(axis_name, source_system, registry_root=root)
            if not column:
                continue
            demo_operator = OperatorSpec(
                name=EFGOperator.COUNT_MEASURE.value,
                role=f"{axis_name}_stratified_event_count",
                output_kind="extensive_measure",
                params={
                    "artifact_path": artifact,
                    "carrier": carrier,
                    "stratify_column": column,
                    "stratify_axis": axis_name,
                    "stratify_source": source_system,
                    "name": f"{source_system}.{Path(artifact).stem}.count.{axis_name}",
                },
            )
            demo_child = expand(demo_operator, count_parents)
            if demo_child is not None:
                count_nodes.append(demo_child)

    # σ-restricted clinical events (MSD §2.6/§3.10.4): infant/neonatal/postneonatal
    # death, inpatient death, low birth weight, prematurity, congenital anomaly. Each is
    # a declarative predicate (from clinical_event_definitions.yaml) applied to the
    # primary carrier's source records — fully registry-driven, no source-specific code.
    from pegasus.registries.events import clinical_ratio_specs, restricted_event_specs

    for rspec in restricted_event_specs(root=root):
        context = primary_event_context.get(rspec.of_carrier)
        if context is None or not rspec.has_conditions():
            continue
        artifact, restrict_parents = context
        restrict_operator = OperatorSpec(
            name=EFGOperator.COUNT_MEASURE.value,
            role=f"{rspec.predicate}_count",
            output_kind="extensive_measure",
            params={
                "artifact_path": artifact,
                "carrier": rspec.event_carrier,
                "restrict_predicate": rspec.predicate,
                "restrict_conditions": [dict(cond) for cond in rspec.conditions],
                "restrict_of_carrier": rspec.of_carrier,
                "restrict_event_id": rspec.event_id,
                "name": f"{restrict_parents[0].source[0]}.{Path(artifact).stem}.{rspec.event_carrier}",
            },
        )
        restrict_child = expand(restrict_operator, restrict_parents)
        if restrict_child is not None:
            count_nodes.append(restrict_child)

    # Statistical-functional fields (MSD §3.10.4-6 Ψ_mean/Ψ_median): mean length of stay,
    # mean cost components, median reporting delay. Registry-driven (functional_fields.yaml);
    # these are intensive marked_functional covariates, not counts (not RN-divided).
    from pegasus.registries.functional import functional_field_specs

    for fspec in functional_field_specs(registry_root=root):
        context = primary_event_context.get(fspec.carrier)
        if context is None:
            continue
        artifact, functional_parents = context
        functional_operator = OperatorSpec(
            name=EFGOperator.PSI_FUNCTIONAL.value,
            role=fspec.role,
            output_kind="marked_functional",
            params={
                "artifact_path": artifact,
                "carrier": fspec.carrier,
                "mark_column": fspec.mark_column,
                "functional": fspec.functional,
                "unit": fspec.unit,
                "functional_id": fspec.field_id,
                "name": f"{functional_parents[0].source[0]}.{Path(artifact).stem}.{fspec.functional}.{fspec.mark_column}",
            },
        )
        expand(functional_operator, functional_parents)

    # Radon–Nikodym ratios, driven by the clinical event registry's declared
    # (numerator_carrier, denominator_carrier, role) pairings. The denominator pool is
    # population anchors plus event counts, so cross-source/cross-event ratios
    # (e.g. InfantDeaths / LiveBirths, HospitalDeaths / HospitalAdmissions) are built
    # generically — the engine holds no hardcoded numerator→denominator map.
    denominators = [field for field in roots if field.carrier == "Population"]
    if operator_mode != "raw_only":
        ratio_specs = clinical_ratio_specs(root=root)
        denominator_pool = [*denominators, *count_nodes]
        for numerator in count_nodes:
            for spec in ratio_specs:
                if spec.numerator_carrier != numerator.carrier:
                    continue
                for denominator in denominator_pool:
                    if denominator.id == numerator.id or denominator.carrier != spec.denominator_carrier:
                        continue
                    # Demographic alignment (§3.7.4): a numerator stratified on a demographic
                    # axis must divide a denominator on the SAME axis, never a marginal total
                    # (and crude/ICD numerators divide the total, not a stratified pop).
                    if _demographic_axes(numerator) != _demographic_axes(denominator):
                        continue
                    operator = OperatorSpec(
                        name=EFGOperator.RN.value,
                        role=spec.role,
                        output_kind="intensive_density",
                        params={"ratio_role": spec.role},
                    )
                    alignment = align_fields(
                        left=numerator,
                        right=denominator,
                        operator=operator,
                        intent=intent,
                        registries=registry_arg,
                    )
                    expand(operator, [numerator, denominator], alignment)

    # Cross-source divergence bridges (§2.11 / relationship surface): registry-declared
    # carrier pairs (e.g. SIH admissions vs SIM deaths) formed into a log-ratio divergence
    # on shared support, at every shared stratifier signature (all-cause and cause-specific).
    # The engine holds no hardcoded divergence pairs — they come from bridge_grammars.yaml.
    if operator_mode != "raw_only":
        from pegasus.registries.bridge import bridge_grammar_entries

        _STRATIFIERS = ("icd_chapter", "icd_block", "curated_cause_group", "sex", "age_group", "race")

        def _signature(node: FieldNode) -> frozenset[str]:
            return frozenset(axis for axis in _STRATIFIERS if axis in node.axes)

        def _event_counts(carrier: str) -> list[FieldNode]:
            return [
                node for node in count_nodes
                if node.carrier == carrier and "source_event_count" in set(node.role or [])
            ]

        for grammar in bridge_grammar_entries(registry_root=root):
            if not str(grammar.get("bridge_type", "")).endswith("divergence"):
                continue
            if "divergence_log_ratio" not in set(grammar.get("operators", []) or []):
                continue
            left_carrier = grammar.get("left_carrier")
            right_carrier = grammar.get("right_carrier")
            if not left_carrier or not right_carrier:
                continue
            bridge_type = str(grammar.get("bridge_type"))
            rights_by_sig: dict[frozenset[str], list[FieldNode]] = {}
            for right in _event_counts(str(right_carrier)):
                rights_by_sig.setdefault(_signature(right), []).append(right)
            for left in _event_counts(str(left_carrier)):
                for right in rights_by_sig.get(_signature(left), []):
                    operator = OperatorSpec(
                        name=EFGOperator.DIVERGENCE.value,
                        role=bridge_type,
                        output_kind="bridge_divergence",
                        params={"name": f"{left_carrier}_vs_{right_carrier}.{bridge_type}", "bridge_type": bridge_type},
                    )
                    alignment = align_fields(
                        left=left, right=right, operator=operator, intent=intent, registries=registry_arg,
                    )
                    expand(operator, [left, right], alignment)

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
            operator_role = _ratio_role(parents[0], parents[1], registry_root=root) if operator_name == EFGOperator.RN.value and len(parents) == 2 else str(candidate.bridge_type)
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
    # Intent-contract enforcement (MSD §3.10): refuse a hollow success — fail the build
    # loudly if the intent's health_seeds are not actually produced by the compiled EFG.
    domain_summary_payload["health_seed_contract"] = enforce_health_seeds(intent, compressed)
    # Named core-seed binding + mandatory_fields enforcement (MSD §3.10): bind canonical
    # V_* / count seed ids to produced fields and fail loudly if the intent's
    # mandatory_fields are not realized.
    seed_resolution = resolve_core_seeds(compressed, registry_root=root)
    domain_summary_payload["core_seed_resolution"] = seed_resolution
    domain_summary_payload["mandatory_field_contract"] = enforce_mandatory_fields(
        intent, compressed, registry_root=root
    )
    # Surface the MSD §3.10 named core-seed surface: set each bound field's display
    # `name` to its canonical seed name (SIMDeathsAll, SINASCLiveBirthsAll, …) so the
    # named V_core fields are discoverable in V_fields / VariableDictionary. The
    # content-addressed `field_id` is preserved untouched, so lineage, edges, and
    # precompression are unaffected (this is a display alias, not a field rename).
    if seed_resolution:
        _spec_by_seed = {spec.seed_id: spec for spec in core_seed_specs(registry_root=root)}
        _canonical_name_by_fid: dict[str, str] = {}
        for seed_id, fid in seed_resolution.items():
            spec = _spec_by_seed.get(seed_id)
            if spec is not None and fid and fid not in _canonical_name_by_fid:
                # Prefer the descriptive composite alias (e.g. SIMCrudeMortalitySIDRAOfficial)
                # over the short MSD shorthand (CrudeMortality) as the public display name —
                # it is the unambiguous, source-qualified name intents declare in
                # mandatory_fields and consumers query by. Seeds without an alias keep their
                # already-descriptive name (e.g. SIMDeathsAll).
                _canonical_name_by_fid[fid] = spec.aliases[0] if spec.aliases else spec.name
        if _canonical_name_by_fid:
            compressed = tuple(
                field.model_copy(update={"name": _canonical_name_by_fid[field.id]})
                if field.id in _canonical_name_by_fid
                else field
                for field in compressed
            )
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

