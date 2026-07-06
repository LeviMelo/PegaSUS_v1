"""Autonomous, source-agnostic Epidemiological Field Graph compiler core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash
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
from pegasus.efg.lineage import lineage_hash
from pegasus.efg.materialize import materialize_substrate_bundle
from pegasus.efg.operators import EFGOperator, apply_operator
from pegasus.she.substrate import SubstrateBundle

from pegasus.efg.dag.helpers import *

__all__ = [
    "EFGResult",
    "build_efg",
    "_build_efg_base",
]


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
    # Intent-level spatial aggregation (MSD §3.7): coarsen all event-count geography to
    # a denser IBGE region so sparse outcomes accumulate per-cell. Threaded into every
    # count operator's params so numerator/denominator/restricted counts aggregate
    # consistently (the RN/divergence joins then operate on the same region cells).
    _geo_agg_raw = (intent.get("geography_aggregation") if isinstance(intent, dict)
                    else getattr(intent, "geography_aggregation", None))
    geo_agg = str(_geo_agg_raw) if _geo_agg_raw and str(_geo_agg_raw) != "municipality" else None

    def _count_params(**params: Any) -> dict[str, Any]:
        if geo_agg:
            params["geography_aggregation"] = geo_agg
        return params

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
            params=_count_params(
                artifact_path=artifact,
                carrier=carrier,
                name=f"{count_parents[0].source[0]}.{Path(artifact).stem}.count",
            ),
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
                    params=_count_params(
                        artifact_path=artifact,
                        carrier=carrier,
                        stratify_icd=level,
                        icd_column=icd_column,
                        icd_axis=axis_name,
                        icd_source_field=icd_column,
                        name=f"{count_parents[0].source[0]}.{Path(artifact).stem}.count.{axis_name}",
                    ),
                )
                strat_child = expand(strat_operator, count_parents)
                if strat_child is not None:
                    count_nodes.append(strat_child)

        # Demographic stratification (MSD §2.8/§3.7.4): stratify this event count by each
        # demographic axis (sex/age/race) that has a matching population denominator tensor,
        # mapping source category codes to the canonical axis. Enables stratified rates.
        source_system = count_parents[0].source[0] if count_parents[0].source else ""
        # Race-specific rates need the admin-race death/birth count bridged to the census
        # self-declared race axis (admin race != self-declared, MSD §3.7.4). When a Bridge_R
        # prior is configured for this compile, thread its path into the race count so the
        # executor emits SELF-DECLARED race counts that divide the self-declared population;
        # without a prior the race count keeps raw admin codes and forms no rate (align gates it).
        _rb_plan = constraints.get("race_bridge_plan") if isinstance(constraints.get("race_bridge_plan"), dict) else {}
        _race_bridge_prior_path = (
            _rb_plan.get("prior_path") if _rb_plan.get("status") in {"planned", "embedded"} else None
        )
        for axis_name in sorted(available_demographic_axes):
            column = _demo_source_column(axis_name, source_system, registry_root=root)
            if not column:
                continue
            extra: dict[str, Any] = {}
            if axis_name == "race" and _race_bridge_prior_path:
                extra["race_bridge_prior_path"] = str(_race_bridge_prior_path)
                extra["race_bridge_id"] = _rb_plan.get("bridge_id")
            demo_operator = OperatorSpec(
                name=EFGOperator.COUNT_MEASURE.value,
                role=f"{axis_name}_stratified_event_count",
                output_kind="extensive_measure",
                params=_count_params(
                    artifact_path=artifact,
                    carrier=carrier,
                    stratify_column=column,
                    stratify_axis=axis_name,
                    stratify_source=source_system,
                    name=f"{source_system}.{Path(artifact).stem}.count.{axis_name}",
                    **extra,
                ),
            )
            demo_child = expand(demo_operator, count_parents)
            if demo_child is not None:
                count_nodes.append(demo_child)

    # σ-restricted clinical events (MSD §2.6/§3.10.4): infant/neonatal/postneonatal
    # death, inpatient death, low birth weight, prematurity, congenital anomaly. Each is
    # a declarative predicate (from health/clinical_event_definitions.yaml) applied to the
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
            params=_count_params(
                artifact_path=artifact,
                carrier=rspec.event_carrier,
                restrict_predicate=rspec.predicate,
                restrict_conditions=[dict(cond) for cond in rspec.conditions],
                restrict_of_carrier=rspec.of_carrier,
                restrict_event_id=rspec.event_id,
                name=f"{restrict_parents[0].source[0]}.{Path(artifact).stem}.{rspec.event_carrier}",
            ),
        )
        restrict_child = expand(restrict_operator, restrict_parents)
        if restrict_child is not None:
            count_nodes.append(restrict_child)

    # Statistical-functional fields (MSD §3.10.4-6 Ψ_mean/Ψ_median): mean length of stay,
    # mean cost components, median reporting delay. Registry-driven (fields/functional_fields.yaml);
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
    # The engine holds no hardcoded divergence pairs — they come from fields/bridge_grammars.yaml.
    if operator_mode != "raw_only":
        from pegasus.registries.bridge import bridge_grammar_entries

        _STRATIFIERS = ("icd_chapter", "icd_block", "curated_cause_group", "sex", "age_group", "race")

        def _signature(node: FieldNode) -> frozenset[str]:
            return frozenset(axis for axis in _STRATIFIERS if axis in node.axes)

        def _event_counts(carrier: str) -> list[FieldNode]:
            # Match a carrier's additive count nodes for divergence pairing: both
            # primary event counts (source_event_count) and σ-restricted clinical
            # event counts (restricted_count, e.g. ArbovirusHospitalAdmissions,
            # MicrocephalyBirths) qualify, so registry-declared divergences can relate
            # restricted carriers across sources (MSD §2.11) — not just raw carriers.
            return [
                node for node in count_nodes
                if node.carrier == carrier
                and bool({"source_event_count", "restricted_count"} & set(node.role or []))
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
            # Optional registry-declared year lag (MSD §2.11): a lagged divergence
            # pairs left(t-k) with right(t), e.g. arbovirus admissions(t-1) vs a birth
            # outcome(t). 0/absent = the standard contemporaneous divergence.
            try:
                temporal_lag = int(grammar.get("temporal_lag") or 0)
            except (TypeError, ValueError):
                temporal_lag = 0
            lag_suffix = f".lag{temporal_lag}" if temporal_lag > 0 else ""
            rights_by_sig: dict[frozenset[str], list[FieldNode]] = {}
            for right in _event_counts(str(right_carrier)):
                rights_by_sig.setdefault(_signature(right), []).append(right)
            for left in _event_counts(str(left_carrier)):
                for right in rights_by_sig.get(_signature(left), []):
                    params: dict[str, Any] = {
                        "name": f"{left_carrier}_vs_{right_carrier}.{bridge_type}{lag_suffix}",
                        "bridge_type": bridge_type,
                    }
                    if temporal_lag > 0:
                        params["temporal_lag"] = temporal_lag
                    operator = OperatorSpec(
                        name=EFGOperator.DIVERGENCE.value,
                        role=bridge_type,
                        output_kind="bridge_divergence",
                        params=params,
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
