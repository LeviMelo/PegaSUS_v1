
"""EFG metadata materialization from SHE SubstrateBundle candidates.

Slice 14A replaces the blocked materialization stub with a typed, metadata-only
boundary.  It does not write output bundles, compute rates, build denominator
ratios, run PIRS/HSIC, or mutate compile outputs.  Its only responsibility is
to convert SHE-admissible source-field candidates into FieldNode objects and
keep SHE exclusions out of the EFG admission surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import FieldNode
from pegasus.efg.lineage import make_lineage, lineage_hash
from pegasus.efg.node import make_field_node
from pegasus.she.substrate import SubstrateBundle, SubstrateFieldCandidate, SubstrateFieldExclusion
from pegasus.denominators.population.anchor import (
    load_sidra_population_total_anchor,
    load_sidra_population_totals_frame,
)


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


def _is_diagnostic_candidate(candidate: SubstrateFieldCandidate) -> bool:
    roles = {str(role) for role in candidate.role}
    return (
        str(candidate.quality_role) == "diagnostic_code"
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

    support = {
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
    roles = {str(role) for role in candidate.role}
    if "time" in candidate.axes or "time_axis_candidate" in roles:
        support["time_column"] = candidate.column
    if "geography" in candidate.axes or "geography_axis" in roles:
        support["geography_column"] = candidate.column
    return support


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


def _sidra_context_materialized_fields(bundle: SubstrateBundle) -> list[SubstrateMaterializedField]:
    """Admit SIDRA context facts (curated socioeconomic compendium) as context_gradient
    fields (MSD §3.10.7 V_X), regime-classified at the §2.9 boundary."""
    from pegasus.she.sidra_context import build_sidra_context_fields

    out: list[SubstrateMaterializedField] = []
    for artifact in bundle.source_artifacts:
        if artifact.source_system != "SIDRA" or artifact.artifact_role != "context_facts":
            continue
        context_fields = build_sidra_context_fields(
            artifact.path,
            artifact_hash=artifact.artifact_hash,
            source_manifest_hash=artifact.source_manifest_hash,
        )
        for field in context_fields:
            out.append(SubstrateMaterializedField(
                candidate_id=field.id,
                field=field,
                lineage_hash=lineage_hash(field.lineage),
                materialization_reason="sidra_context_field",
                warnings=tuple(field.warnings),
            ))
    return out


def _sidra_demographic_population_materialized_fields(bundle: SubstrateBundle) -> list[SubstrateMaterializedField]:
    """Admit disaggregated SIDRA population (by sex/race/age) as a demographic-stratified
    Population field (MSD §2.8), for SIDRA artifacts with role 'population_strata'.

    Denominator-mode reconciliation (§2.8.2): in independent/sim-informed tensor modes a
    solver `population_tensor` artifact is the *authoritative reconstructed* denominator and
    the raw `population_strata` is merely its input anchor. When a solver tensor is present
    we therefore do NOT also admit the raw strata as a competing demographic denominator
    (that would yield duplicate/competing stratified rates). In official_sidra_anchor mode
    no solver tensor exists, so the observed strata are admitted directly.
    """
    from pegasus.denominators.population.fields import build_sidra_demographic_population_fields

    has_solver_tensor = any(
        artifact.source_system == "SIDRA" and artifact.artifact_role == "population_tensor"
        for artifact in bundle.source_artifacts
    )
    if has_solver_tensor:
        return []

    out: list[SubstrateMaterializedField] = []
    for artifact in bundle.source_artifacts:
        if artifact.source_system != "SIDRA" or artifact.artifact_role != "population_strata":
            continue
        try:
            fields = build_sidra_demographic_population_fields(
                artifact.path,
                artifact_hash=artifact.artifact_hash,
                source_manifest_hash=artifact.source_manifest_hash,
            )
        except Exception:  # pragma: no cover - defensive; malformed strata artifact
            continue
        for field in fields:
            out.append(SubstrateMaterializedField(
                candidate_id=field.id,
                field=field,
                lineage_hash=lineage_hash(field.lineage),
                materialization_reason="sidra_demographic_population",
                warnings=tuple(field.warnings),
            ))
    return out


def _population_solver_materialized_fields(bundle: SubstrateBundle) -> list[SubstrateMaterializedField]:
    """Admit materialized population solver tensors as Population denominator fields.

    These tensors are produced by ``sidra.population_cube.build`` for
    ``independent_population_tensor`` / ``sim_informed_population_tensor`` intents. They
    enter through the source-artifact boundary instead of compile side channels.
    """
    out: list[SubstrateMaterializedField] = []
    for artifact in bundle.source_artifacts:
        if artifact.source_system != "SIDRA" or artifact.artifact_role != "population_tensor":
            continue
        try:
            import polars as pl

            frame = pl.read_parquet(artifact.path, n_rows=5)
        except Exception:
            continue
        if "population_tensor_mode" not in frame.columns or "solver_id" not in frame.columns:
            continue
        mode = str(frame.get_column("population_tensor_mode")[0])
        solver_id = str(frame.get_column("solver_id")[0])

        def _first(col: str, default: Any) -> Any:
            if col in frame.columns:
                try:
                    v = frame.get_column(col).drop_nulls().to_list()
                    if v:
                        return v[0]
                except Exception:
                    pass
            return default

        demographic_axes: list[str] = []
        for axis in ("age_group", "sex", "race"):
            if axis in frame.columns:
                try:
                    values = [str(v) for v in frame.get_column(axis).drop_nulls().unique().to_list()]
                except Exception:
                    values = []
                if values and values != ["__total__"]:
                    demographic_axes.append(axis)

        solver_backend = str(_first("solver_backend", "projected_gradient"))
        sparse_jacobian = bool(_first("sparse_jacobian", False))
        feedback = bool(_first("denominator_feedback_warning", mode == "sim_informed_denominator"))

        # Emit one denominator field per marginal: a `total` crude denominator (tensor
        # summed over all demographic axes) plus one per demographic axis (marginalized
        # over the others). Each is a proper same-axis match for its numerator -- a
        # sex-stratified death count divides by the sex-marginal population (§3.7.4);
        # the RN pairing (dag.py) requires the numerator/denominator demographic axes to
        # be equal, so a single all-axes field would never pair with a single-axis count.
        for target_axis in [None, *demographic_axes]:
            axes = {
                "geography_axis": "N6",
                "time_axis": "period",
                "population_tensor_mode": mode,
                "population_strata_axis": target_axis or "total",
            }
            roles = ["population_tensor", "population_denominator_seed", "population_solver", "source_field"]
            if target_axis:
                axes[target_axis] = "stratified"
                roles.insert(3, "demographic_stratified")
                # The SIDRA 9606 population race axis IS IBGE self-declared census race;
                # declare it explicitly so a Bridge_R self-declared death/birth count can be
                # divided by it (declaration.evaluate_declaration_compatibility fails closed
                # when one RN operand's race axis is unlabeled, §3.7.4/EFG-DECL-02).
                if target_axis == "race":
                    axes["race_axis_type"] = "ibge_self_declared"
            support = {
                "support_kind": "population_tensor_solver_output",
                "source_system": "SIDRA",
                "artifact_path": str(artifact.path),
                "population_tensor_path": str(artifact.path),
                "marginal_demographic_axis": target_axis,
                "PopulationTensorMode": mode,
                "SolverID": solver_id,
                "SolverBackend": solver_backend,
                "SparseJacobian": sparse_jacobian,
                "DenominatorFeedbackWarning": feedback,
                "population_tensor_diagnostics": "solver_materialized_tensor",
            }
            lineage = make_lineage(
                parent_ids=[],
                operator_type="population_tensor_solver",
                operator_params=support,
                registry_versions={"SIDRA": artifact.artifact_hash or artifact.source_manifest_hash or "unknown", "population_solver": solver_id},
                source_manifest_hashes=_unique([artifact.source_manifest_hash, artifact.artifact_hash]),
                code_version="population_tensor_solver_v1",
            )
            field = make_field_node(
                name=f"population_tensor_solver_{mode}_{target_axis or 'total'}",
                kind="extensive_measure",
                carrier="Population",
                unit="persons",
                support=support,
                axes=axes,
                aggregation="additive",
                role=roles,
                source=["SIDRA", str(artifact.path), "SIDRA_9606_POPULATION_SOLVER"],
                operator="population_tensor_solver",
                provenance=["SHE_SubstrateBundle", "population_tensor", "solver", "SIDRA_9606"],
                state="warning" if mode == "sim_informed_denominator" else "verified",
                warnings=["sim_informed_population_feedback_risk"] if mode == "sim_informed_denominator" else [],
                lineage=lineage,
                materialization_state="metadata_only",
                path=None,
                dashboard_safe="warning" if mode == "sim_informed_denominator" else False,
            ).model_copy(update={"id": f"population_tensor_solver_{target_axis or 'total'}_{lineage_hash(lineage)[:20]}"})
            out.append(SubstrateMaterializedField(
                candidate_id=field.id,
                field=field,
                lineage_hash=lineage_hash(field.lineage),
                materialization_reason="population_tensor_solver",
                warnings=tuple(field.warnings),
            ))
    return out


def _cnes_facility_stock_materialized_fields(bundle: SubstrateBundle) -> list[SubstrateMaterializedField]:
    """Admit CNES-ST facility-period stock as a first-class Facilities count.

    CNES-ST bed/capacity columns are marks over a facility registry row, not the
    facility stock itself.  The core seed `CNESFacilitiesAll` therefore needs an
    explicit source-artifact event count over the real CNES-ST processed table.
    """

    out: list[SubstrateMaterializedField] = []
    for artifact in bundle.source_artifacts:
        if artifact.source_system != "CNES-ST" or artifact.artifact_role != "processed_events":
            continue
        artifact_path = Path(artifact.path)
        if not artifact_path.exists():
            continue
        try:
            import polars as pl

            frame = pl.read_parquet(artifact_path, n_rows=25)
        except Exception:
            continue
        columns = set(frame.columns)
        if "facility_id" not in columns and not any(col.startswith("cnpj_") for col in columns):
            continue
        support: dict[str, Any] = {
            "support_kind": "source_artifact_event_count",
            "source_system": "CNES-ST",
            "artifact_path": str(artifact_path),
            "source_columns": sorted(columns),
            "facility_identifier_column": "facility_id" if "facility_id" in columns else None,
            "row_count": None,
        }
        axes: dict[str, Any] = {}
        if "mun_facility_cod6" in columns:
            support["geography_column"] = "mun_facility_cod6"
            axes["geography"] = "mun_facility_cod6"
        elif "municipality_cod6" in columns:
            support["geography_column"] = "municipality_cod6"
            axes["geography"] = "municipality_cod6"
        if "year" in columns:
            support["time_column"] = "year"
            axes["time"] = "year"
        elif "competence_year" in columns:
            support["time_column"] = "competence_year"
            axes["time"] = "competence_year"
        lineage = make_lineage(
            parent_ids=[],
            operator_type="count_measure",
            operator_params={
                "carrier": "Facilities",
                "role": "source_event_count",
                "source_system": "CNES-ST",
                "artifact_path": str(artifact_path),
                "facility_identifier_column": support.get("facility_identifier_column"),
                "geography_column": support.get("geography_column"),
                "time_column": support.get("time_column"),
            },
            registry_versions={
                "CNES-ST": artifact.artifact_hash or artifact.source_manifest_hash or "unknown",
                "cnes_facility_stock": "source_artifact_event_count_v1",
            },
            source_manifest_hashes=_unique([artifact.source_manifest_hash, artifact.artifact_hash]),
            code_version="cnes_facility_stock_v1",
        )
        field = make_field_node(
            name=f"CNES facility stock count {artifact_path.stem}",
            kind="extensive_measure",
            carrier="Facilities",
            unit="counts",
            support=support,
            axes=axes,
            aggregation="additive",
            role=["source_event_count", "facility_stock", "source_field"],
            source=["CNES-ST", str(artifact_path), support.get("facility_identifier_column") or "facility_record"],
            operator="count_measure",
            provenance=["SHE_SubstrateBundle", "source_normalized", "facility_registry", "CNES-ST"],
            state="verified",
            warnings=[],
            lineage=lineage,
            materialization_state="metadata_only",
            path=None,
            dashboard_safe=False,
        ).model_copy(update={"id": f"cnes_facility_stock_{lineage_hash(lineage)[:24]}"})
        out.append(SubstrateMaterializedField(
            candidate_id=field.id,
            field=field,
            lineage_hash=lineage_hash(field.lineage),
            materialization_reason="cnes_facility_stock_count",
            warnings=(),
        ))
    return out


def materialize_substrate_bundle(bundle: SubstrateBundle) -> SubstrateMaterializationResult:
    """Convert SHE-admissible substrate candidates into metadata-only FieldNodes.

    Exclusions remain exclusions.  No excluded field is promoted into EFG nodes,
    and no output bundle files are written here.
    """

    fields_list = [materialize_candidate_field(candidate) for candidate in bundle.candidates]
    sidra_anchor = _sidra_population_anchor_field(bundle)
    if sidra_anchor is not None:
        fields_list.append(sidra_anchor)
    fields_list.extend(_sidra_context_materialized_fields(bundle))
    fields_list.extend(_sidra_demographic_population_materialized_fields(bundle))
    fields_list.extend(_population_solver_materialized_fields(bundle))
    fields_list.extend(_cnes_facility_stock_materialized_fields(bundle))
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
