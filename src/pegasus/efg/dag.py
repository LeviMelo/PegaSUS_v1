"""Autonomous, source-agnostic Epidemiological Field Graph compiler core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.core.schemas import FieldNode, UserIntent
from pegasus.efg.align import align_fields
from pegasus.efg.declaration import OperatorSpec
from pegasus.efg.equivalence import PrecompressionReport, precompress_fields
from pegasus.efg.failed_branch import (
    FailedBranchRecord,
    failed_exclusion,
    make_failed_branch_record,
)
from pegasus.efg.legality import evaluate_delta
from pegasus.efg.lineage import lineage_hash
from pegasus.efg.materialize import materialize_substrate_bundle
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

    @property
    def field_count(self) -> int:
        return len(self.fields)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
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
    payload = {
        "substrate_id": substrate.substrate_id,
        "field_ids": [field.id for field in compressed],
        "edge_ids": [edge.edge_id for edge in final_edges],
        "failed_ids": [branch.failed_branch_id for branch in failures],
        "registry_hashes": registry_hashes,
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
    )

# ---- Slice 28Y EFG semantic manifest integration ----
# The autonomous DAG now records metadata-only core-seed and bridge-plan evidence
# without changing first-class output-bundle keys and without materializing tensors.
from functools import wraps as _slice28y_wraps

from pegasus.efg.bridges import bridge_summary as _slice28y_bridge_summary
from pegasus.efg.bridges import plan_bridge_candidates as _slice28y_plan_bridge_candidates
from pegasus.efg.core_seed import build_core_seed_set as _slice28y_build_core_seed_set
from pegasus.efg.core_seed import core_seed_summary as _slice28y_core_seed_summary


def _slice28y_empty_core_seed_summary() -> dict[str, object]:
    return {
        "registry_root": "config/registries",
        "seed_count": 0,
        "blocked_count": 0,
        "role_counts": {},
        "seeds": [],
        "blocked": [],
    }


def _slice28y_empty_bridge_plan_summary() -> dict[str, object]:
    return {
        "candidate_count": 0,
        "blocked_count": 0,
        "bridge_type_counts": {},
        "candidates": [],
        "blocked": [],
    }


if not hasattr(EFGResult, "_slice28y_base_as_manifest"):
    EFGResult._slice28y_base_as_manifest = EFGResult.as_manifest  # type: ignore[attr-defined]


def _slice28y_efgresult_as_manifest(self):
    payload = self._slice28y_base_as_manifest()  # type: ignore[attr-defined]
    payload.setdefault(
        "core_seed_summary",
        getattr(self, "_slice28y_core_seed_summary", _slice28y_empty_core_seed_summary()),
    )
    payload.setdefault(
        "bridge_plan_summary",
        getattr(self, "_slice28y_bridge_plan_summary", _slice28y_empty_bridge_plan_summary()),
    )
    payload.setdefault("semantic_manifest_schema", "28Y.1")
    return payload


EFGResult.as_manifest = _slice28y_efgresult_as_manifest  # type: ignore[method-assign]


@_slice28y_wraps(_build_efg_base)
def build_efg(*args, **kwargs):
    result = _build_efg_base(*args, **kwargs)
    fields = tuple(getattr(result, "fields", ()) or ())
    registry_root = kwargs.get("registry_root", "config/registries")
    intent = kwargs.get("intent")
    try:
        seed_set = _slice28y_build_core_seed_set(fields, registry_root=registry_root, intent=intent)
        bridge_plan = _slice28y_plan_bridge_candidates(fields, registry_root=registry_root, intent=intent)
        object.__setattr__(result, "_slice28y_core_seed_summary", _slice28y_core_seed_summary(seed_set))
        object.__setattr__(result, "_slice28y_bridge_plan_summary", _slice28y_bridge_summary(bridge_plan))
    except Exception as exc:  # pragma: no cover - defensive metadata guard only
        object.__setattr__(result, "_slice28y_core_seed_summary", _slice28y_empty_core_seed_summary())
        object.__setattr__(result, "_slice28y_bridge_plan_summary", {
            **_slice28y_empty_bridge_plan_summary(),
            "blocked": [{"reason": "semantic_manifest_failed", "error": str(exc)}],
            "blocked_count": 1,
        })
    return result
