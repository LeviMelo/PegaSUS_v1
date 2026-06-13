
"""EFG metadata materialization from SHE SubstrateBundle candidates.

Slice 14A replaces the blocked materialization stub with a typed, metadata-only
boundary.  It does not write output bundles, compute rates, build denominator
ratios, run PIRS/HSIC, or mutate compile outputs.  Its only responsibility is
to convert SHE-admissible source-field candidates into FieldNode objects and
keep SHE exclusions out of the EFG admission surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import FieldNode
from pegasus.efg.lineage import make_lineage, lineage_hash
from pegasus.efg.node import make_field_node
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

    raw = str(candidate.substrate_kind)
    if raw in FIELD_KINDS:
        return raw  # type: ignore[return-value]
    roles = set(str(role) for role in candidate.role)
    unit = str(candidate.unit)
    aggregation = str(candidate.aggregation)
    if unit == "ICD10" or "diagnostic_topology" in roles:
        return "observer_proxy"
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

    aggregation = _safe_aggregation(str(candidate.aggregation))
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
        unit=str(candidate.unit),
        support=support_from_substrate_candidate(candidate),
        axes=dict(candidate.axes),
        aggregation=aggregation,
        role=_unique([*candidate.role, "source_field", "substrate_materialized"]),
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


def materialize_substrate_bundle(bundle: SubstrateBundle) -> SubstrateMaterializationResult:
    """Convert SHE-admissible substrate candidates into metadata-only FieldNodes.

    Exclusions remain exclusions.  No excluded field is promoted into EFG nodes,
    and no output bundle files are written here.
    """

    fields = tuple(materialize_candidate_field(candidate) for candidate in bundle.candidates)
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
