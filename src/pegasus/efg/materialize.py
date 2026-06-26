
"""EFG metadata materialization from SHE SubstrateBundle candidates.

Slice 14A replaces the blocked materialization stub with a typed, metadata-only
boundary.  It does not write output bundles, compute rates, build denominator
ratios, run PIRS/HSIC, or mutate compile outputs.  Its only responsibility is
to convert SHE-admissible source-field candidates into FieldNode objects and
keep SHE exclusions out of the EFG admission surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import FieldNode
from pegasus.efg.lineage import make_lineage, lineage_hash
from pegasus.efg.node import make_field_node
from pegasus.she.population.sidra_anchor import (
    load_sidra_population_total_anchor,
    load_sidra_population_totals_frame,
)
from pegasus.she.substrate import SubstrateBundle, SubstrateFieldCandidate, SubstrateFieldExclusion


FIELD_KINDS: frozenset[str] = frozenset({
    "extensive_measure",
    "intensive_density",
    "marked_functional",
    "context_gradient",
    "bridge_divergence",
    "bridge_module",
    "observer_proxy",
    "latent_context",
    "model_residual",
})
AGGREGATION_LAWS: frozenset[str] = frozenset({
    "additive",
    "weighted_mean",
    "statistical_functional",
    "compositional",
    "non_aggregable",
})
COUNT_UNITS: frozenset[str] = frozenset({"count", "counts", "events", "admissions", "births", "deaths"})
DIAGNOSTIC_CODE_COLUMNS: frozenset[str] = frozenset({
    "underlying_icd_norm",
    "associated_conditions_norm",
    "principal_icd_norm",
    "anomaly_icd_code",
})


class EFGMaterializationError(ValueError):
    """Raised when substrate-to-EFG metadata materialization is invalid."""


@dataclass(frozen=True)
class SubstrateMaterializedField:
    """One metadata-only FieldNode emitted from one SHE substrate candidate."""

    candidate_id: str
    field: FieldNode
    lineage_hash: str
    materialization_reason: str
    warnings: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "field": self.field.model_dump(mode="json"),
            "lineage_hash": self.lineage_hash,
            "materialization_reason": self.materialization_reason,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SubstrateMaterializationResult:
    """Metadata-only EFG admission result for a SHE SubstrateBundle."""

    schema_version: str
    substrate_id: str
    materialization_id: str
    fields: tuple[SubstrateMaterializedField, ...]
    excluded_source_fields: tuple[dict[str, Any], ...]
    registry_hashes: dict[str, str]
    warnings: tuple[str, ...]

    @property
    def field_count(self) -> int:
        return len(self.fields)

    @property
    def excluded_field_count(self) -> int:
        return len(self.excluded_source_fields)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "substrate_id": self.substrate_id,
            "materialization_id": self.materialization_id,
            "field_count": self.field_count,
            "excluded_field_count": self.excluded_field_count,
            "registry_hashes": dict(self.registry_hashes),
            "warnings": list(self.warnings),
            "fields": [field.as_manifest() for field in self.fields],
            "excluded_source_fields": list(self.excluded_source_fields),
        }


def _unique(values: Iterable[str | None]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            seen.setdefault(text, None)
    return list(seen)


def _safe_aggregation(value: str) -> Literal[
    "additive",
    "weighted_mean",
    "statistical_functional",
    "compositional",
    "non_aggregable",
]:
    if value in AGGREGATION_LAWS:
        return value  # type: ignore[return-value]
    return "non_aggregable"


def _is_diagnostic_candidate(candidate: SubstrateFieldCandidate) -> bool:
    roles = {str(role) for role in candidate.role}
    return (
        str(candidate.column) in DIAGNOSTIC_CODE_COLUMNS
        or str(candidate.unit) == "ICD10"
        or "diagnostic_topology" in roles
    )


def classify_substrate_candidate_kind(candidate: SubstrateFieldCandidate) -> Literal[
    "extensive_measure",
    "intensive_density",
    "marked_functional",
    "context_gradient",
    "bridge_divergence",
    "bridge_module",
    "observer_proxy",
    "latent_context",
    "model_residual",
]:
    """Map a SHE substrate candidate to an allowed FieldNode kind.

    Registry-provided FieldNode kinds are respected directly.  Otherwise this
    function performs a deterministic metadata-only classification from unit,
    aggregation law and role.  It never makes epidemiological ratio claims.
    """

    unit = str(candidate.unit)
    aggregation = str(candidate.aggregation)
    if _is_diagnostic_candidate(candidate):
        return "observer_proxy"
    raw = str(candidate.substrate_kind)
    if raw in FIELD_KINDS:
        return raw  # type: ignore[return-value]
    if aggregation == "additive" and unit.lower() in COUNT_UNITS:
        return "extensive_measure"
    if aggregation == "weighted_mean":
        return "intensive_density"
    if aggregation in {"statistical_functional", "compositional"}:
        return "marked_functional"
    return "observer_proxy"


def support_from_substrate_candidate(candidate: SubstrateFieldCandidate) -> dict[str, Any]:
    """Build a conservative support descriptor without fabricating axes."""

    return {
        "support_kind": "source_artifact_column",
        "source_system": candidate.source_system,
        "artifact_path": candidate.artifact_path,
        "column": candidate.column,
        "row_count": candidate.row_count,
        "non_null_count": candidate.non_null_count,
        "unique_non_null_count": candidate.unique_non_null_count,
        "missing_rate": candidate.missing_rate,
        "numeric_min": candidate.numeric_min,
        "numeric_max": candidate.numeric_max,
    }


def materialize_candidate_field(candidate: SubstrateFieldCandidate) -> SubstrateMaterializedField:
    """Materialize one SHE substrate candidate as a metadata-only FieldNode."""

    diagnostic = _is_diagnostic_candidate(candidate)
    aggregation = "non_aggregable" if diagnostic else _safe_aggregation(str(candidate.aggregation))
    unit = "ICD10" if diagnostic else str(candidate.unit)
    role = _unique([*candidate.role, "source_field", "substrate_materialized"])
    if diagnostic and "diagnostic_topology" not in role:
        role.append("diagnostic_topology")
    kind = classify_substrate_candidate_kind(candidate)
    source_manifest_hashes = _unique([candidate.source_manifest_hash, candidate.artifact_hash])
    lineage = make_lineage(
        parent_ids=[],
        operator_type="she_substrate_materialization",
        operator_params={
            "candidate_id": candidate.candidate_id,
            "source_system": candidate.source_system,
            "artifact_path": candidate.artifact_path,
            "column": candidate.column,
            "technical_name": candidate.technical_name,
            "substrate_kind": candidate.substrate_kind,
            "row_count": candidate.row_count,
            "non_null_count": candidate.non_null_count,
            "unique_non_null_count": candidate.unique_non_null_count,
            "missing_rate": candidate.missing_rate,
        },
        registry_versions={candidate.source_system: candidate.registry_hash, "substrate_materializer": "slice14a"},
        source_manifest_hashes=source_manifest_hashes,
        code_version="slice14a",
    )
    warnings = _unique([*candidate.warnings])
    state = "fragile" if warnings or (candidate.missing_rate is not None and candidate.missing_rate > 0.5) else "verified"
    field = make_field_node(
        name=candidate.technical_name,
        kind=kind,
        carrier=str(candidate.carrier),
        unit=unit,
        support=support_from_substrate_candidate(candidate),
        axes=dict(candidate.axes),
        aggregation=aggregation,
        role=role,
        source=_unique([candidate.source_system, candidate.artifact_path, candidate.column]),
        operator="she_substrate_materialization",
        provenance=_unique([*candidate.provenance, "SHE_SubstrateBundle"]),
        state=state,
        warnings=warnings,
        lineage=lineage,
        materialization_state="metadata_only",
        path=None,
        dashboard_safe="warning" if warnings else False,
    )
    return SubstrateMaterializedField(
        candidate_id=candidate.candidate_id,
        field=field,
        lineage_hash=lineage_hash(lineage),
        materialization_reason="metadata_only_substrate_candidate",
        warnings=tuple(warnings),
    )


def _exclusion_manifest(exclusion: SubstrateFieldExclusion) -> dict[str, Any]:
    payload = exclusion.as_manifest()
    payload["efg_materialized"] = False
    payload["efg_exclusion_reason"] = exclusion.reason
    return payload


def _sidra_population_anchor_field(bundle: SubstrateBundle) -> SubstrateMaterializedField | None:
    sidra_artifact = next(
        (
            artifact
            for artifact in bundle.source_artifacts
            if artifact.source_system == "SIDRA" and artifact.artifact_role == "normalized_facts"
        ),
        None,
    )
    if sidra_artifact is None:
        return None

    # Per-municipality population panel (one row per muni/year). Generalizes the
    # old single-locality anchor so state and national grids get a real
    # denominator panel instead of crashing on "expected exactly one total".
    frame = load_sidra_population_totals_frame(sidra_artifact.path)
    if frame.height == 0:
        return None
    years = sorted({int(y) for y in frame["year"].drop_nulls().to_list()})
    period_year = years[0] if len(years) == 1 else None
    single = frame.height == 1
    municipality_cod6 = str(frame["municipality_cod6"][0]) if single else None
    value = float(frame["value"][0]) if single else None
    name_locality = municipality_cod6 if single else "panel"
    period_label = str(period_year) if period_year is not None else "multi"
    support = {
        "support_kind": "sidra_population_total_anchor",
        "source_system": "SIDRA",
        "artifact_path": str(sidra_artifact.path),
        "sidra_facts_path": str(sidra_artifact.path),
        "table_id": "9606",
        "variable_id": "93",
        "period": period_label,
        "year": period_year,
        "locality_level": "N6",
        "municipality_cod6": municipality_cod6,
        "n_localities": int(frame.height),
        "value": value,
        "n_denom": value,
        "unit_raw": "Pessoas",
        "population_tensor_diagnostics": "official_sidra_9606_total_anchor",
        "PopulationTensorMode": "official_sidra_anchor",
        "SolverBackend": "official_sidra_anchor",
        "SolverID": "sidra_9606_total_anchor",
        "SparseJacobian": False,
        "DenominatorFeedbackWarning": None,
    }
    axes = {
        "geography_axis": "N6",
        "time_axis": "period",
        "population_strata_axis": "total",
        "population_tensor_mode": "official_sidra_anchor",
    }
    lineage = make_lineage(
        parent_ids=[],
        operator_type="sidra_population_total_anchor",
        operator_params=support,
        registry_versions={"SIDRA": (sidra_artifact.artifact_hash or sidra_artifact.source_manifest_hash or "unknown"), "sidra_population_anchor": "sidra_9606_total_v1"},
        source_manifest_hashes=_unique([sidra_artifact.source_manifest_hash, sidra_artifact.artifact_hash]),
        code_version="slice28y_sidra_anchor",
    )
    field = make_field_node(
        name=f"population_tensor_sidra_9606_total_{name_locality}_{period_label}",
        kind="extensive_measure",
        carrier="Population",
        unit="persons",
        support=support,
        axes=axes,
        aggregation="additive",
        role=["population_tensor", "population_denominator_seed", "official_sidra_anchor", "source_field"],
        source=["SIDRA", str(sidra_artifact.path), "SIDRA_9606_TOTAL"],
        operator="sidra_population_total_anchor",
        provenance=["SHE_SubstrateBundle", "population_tensor", "official", "SIDRA_9606"],
        state="verified",
        warnings=[],
        lineage=lineage,
        materialization_state="metadata_only",
        path=None,
        dashboard_safe=False,
    ).model_copy(update={"id": f"population_tensor_{lineage_hash(lineage)[:24]}"})
    return SubstrateMaterializedField(
        candidate_id=field.id,
        field=field,
        lineage_hash=lineage_hash(lineage),
        materialization_reason="sidra_population_total_anchor",
        warnings=(),
    )


def materialize_substrate_bundle(bundle: SubstrateBundle) -> SubstrateMaterializationResult:
    """Convert SHE-admissible substrate candidates into metadata-only FieldNodes.

    Exclusions remain exclusions.  No excluded field is promoted into EFG nodes,
    and no output bundle files are written here.
    """

    fields_list = [materialize_candidate_field(candidate) for candidate in bundle.candidates]
    sidra_anchor = _sidra_population_anchor_field(bundle)
    if sidra_anchor is not None:
        fields_list.append(sidra_anchor)
    fields = tuple(fields_list)
    excluded = tuple(_exclusion_manifest(exclusion) for exclusion in bundle.exclusions)
    payload = {
        "substrate_id": bundle.substrate_id,
        "field_ids": [item.field.id for item in fields],
        "excluded_ids": [item.get("exclusion_id") for item in excluded],
        "registry_hashes": dict(bundle.registry_hashes),
    }
    warnings = _unique([*bundle.warnings])
    return SubstrateMaterializationResult(
        schema_version="1.0",
        substrate_id=bundle.substrate_id,
        materialization_id=f"efg_materialization_{content_hash(payload)[:20]}",
        fields=fields,
        excluded_source_fields=excluded,
        registry_hashes=dict(bundle.registry_hashes),
        warnings=tuple(warnings),
    )


def materialized_field_nodes(bundle: SubstrateBundle) -> tuple[FieldNode, ...]:
    """Return only FieldNode objects for SHE-admissible substrate candidates."""

    return tuple(item.field for item in materialize_substrate_bundle(bundle).fields)
