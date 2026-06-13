from __future__ import annotations

from pegasus.efg.align import align_fields
from pegasus.efg.dag import build_efg
from pegasus.efg.declaration import OperatorSpec
from pegasus.efg.equivalence import precompress_fields
from pegasus.efg.legality import evaluate_delta
from pegasus.efg.materialize import materialize_candidate_field
from pegasus.efg.promotion_plan import materialized_fields_from_manifest
from pegasus.she.substrate import (
    SubstrateBundle,
    SubstrateFieldCandidate,
    SubstrateFieldExclusion,
)


def _candidate(**overrides) -> SubstrateFieldCandidate:
    base = dict(
        candidate_id="candidate_year",
        source_system="SIM-DO",
        artifact_path="tests/fixtures/sim.parquet",
        column="year",
        technical_name="SIM-DO.year",
        carrier="Deaths",
        unit="counts",
        aggregation="additive",
        role=("time_axis_candidate",),
        axes={"time": "year"},
        provenance=("source_normalized", "fixture"),
        substrate_kind="extensive_measure",
        registry_hash="source-registry-hash",
        row_count=10,
        non_null_count=10,
        unique_non_null_count=2,
        missing_rate=0.0,
        numeric_min=2021.0,
        numeric_max=2022.0,
        source_manifest_hash="source-manifest-hash",
        artifact_hash="artifact-hash",
        warnings=(),
    )
    base.update(overrides)
    return SubstrateFieldCandidate(**base)


def _exclusion(column: str, reason: str) -> SubstrateFieldExclusion:
    return SubstrateFieldExclusion(
        exclusion_id=f"excluded_{column}",
        source_system="SIM-DO",
        artifact_path="tests/fixtures/sim.parquet",
        column=column,
        reason=reason,
        registry_reason=None,
        row_count=10,
        non_null_count=0 if reason == "all_missing" else 10,
        unique_non_null_count=0 if reason == "all_missing" else 1,
        missing_rate=1.0 if reason == "all_missing" else 0.0,
        structural_role="excluded",
        warnings=(reason,),
    )


def _bundle(*candidates: SubstrateFieldCandidate) -> SubstrateBundle:
    return SubstrateBundle(
        schema_version="1.0",
        substrate_id="substrate_slice19a",
        source_reality_mode="fixture_only",
        source_artifacts=(),
        candidates=tuple(candidates),
        exclusions=(
            _exclusion("constant_col", "zero_variance_constant"),
            _exclusion("missing_col", "all_missing"),
            _exclusion("event_id", "structural_or_audit_only"),
        ),
        table_profiles=(),
        registry_hashes={"SIM-DO": "source-registry-hash"},
        warnings=(),
    )


def test_slice19a_builds_source_nodes_count_measure_edges_and_legal_rate() -> None:
    population = _candidate(
        candidate_id="candidate_population",
        source_system="SIDRA",
        artifact_path="tests/fixtures/population.parquet",
        column="population",
        technical_name="SIDRA.population",
        carrier="Population",
        unit="persons",
        role=("denominator_measure",),
        axes={"time": "year", "geography": "mun_residence_cod6"},
        provenance=("sidra_contextual", "fixture"),
    )
    geography = _candidate(
        candidate_id="candidate_geography",
        column="mun_residence_cod6",
        technical_name="SIM-DO.mun_residence_cod6",
        role=("geography_axis", "residence"),
        axes={"geography": "mun_residence_cod6"},
    )
    result = build_efg(substrate=_bundle(_candidate(), geography, population))

    names = {field.name for field in result.fields}
    assert "SIM-DO.sim.count" in names
    rate = next(field for field in result.fields if "mortality_rate" in field.role)
    assert rate.unit == "rate"
    assert {edge.parent_field_id for edge in result.edges if edge.child_field_id == rate.id}
    assert any(
        edge.child_field_id == rate.id and edge.operator == "denominator_link"
        for edge in result.edges
    )
    assert result.legality_summary["legal"] >= 2
    assert materialized_fields_from_manifest(result.as_manifest())


def test_slice19a_exclusions_never_enter_nodes_and_are_auditable_failures() -> None:
    result = build_efg(substrate=_bundle(_candidate()), operator_mode="raw_only")
    node_columns = {field.support.get("column") for field in result.fields}
    assert "constant_col" not in node_columns
    assert "missing_col" not in node_columns
    assert "event_id" not in node_columns
    reasons = {branch.reason for branch in result.failed_branches}
    assert {"zero_variance_constant", "all_missing", "structural_or_audit_only"} <= reasons


def test_slice19a_icd_fields_are_topology_observers_not_additive_measures() -> None:
    diagnostic = _candidate(
        candidate_id="candidate_icd",
        column="underlying_icd_norm",
        technical_name="SIM-DO.underlying_icd_norm",
        role=("diagnostic_topology", "underlying_cause"),
        axes={"icd_topology_role": "underlying_cause"},
    )
    field = materialize_candidate_field(diagnostic).field
    assert field.kind == "observer_proxy"
    assert field.unit == "ICD10"
    assert field.aggregation == "non_aggregable"
    assert "diagnostic_topology" in field.role


def test_slice19a_illegal_raw_ratio_and_support_mismatch_fail_explicitly() -> None:
    numerator = materialize_candidate_field(_candidate(
        axes={"time": "year", "geography": "mun_residence_cod6"},
    )).field
    denominator = materialize_candidate_field(_candidate(
        candidate_id="candidate_population",
        source_system="SIDRA",
        artifact_path="tests/fixtures/population.parquet",
        column="population",
        technical_name="SIDRA.population",
        carrier="Population",
        unit="persons",
        role=("denominator_measure",),
        axes={"time": "year", "geography": "state"},
    )).field
    operator = OperatorSpec(name="RN", role="mortality_rate")
    alignment = align_fields(left=numerator, right=denominator, operator=operator)
    delta = evaluate_delta(parents=[numerator, denominator], operator=operator, alignment=alignment)

    assert alignment.ok is False
    assert "geospatial_transform_required" in str(alignment.failure_reason)
    assert delta.legal is False
    assert {"support", "axes", "aggregation"} <= set(delta.failed_terms)


def test_slice19a_race_mismatch_requires_bridge_and_precompression_suppresses_duplicates() -> None:
    left = materialize_candidate_field(_candidate(
        candidate_id="race_num",
        column="race_count",
        technical_name="SIM-DO.race_count",
        axes={"race_axis_type": "administrative_death_declaration"},
    )).field
    right = materialize_candidate_field(_candidate(
        candidate_id="race_den",
        source_system="SIDRA",
        artifact_path="tests/fixtures/population.parquet",
        column="population",
        technical_name="SIDRA.population",
        carrier="Population",
        unit="persons",
        axes={"race_axis_type": "self_declared"},
    )).field
    operator = OperatorSpec(name="RN", role="mortality_rate")
    alignment = align_fields(left=left, right=right, operator=operator)
    assert alignment.ok is False
    assert "required_module:Bridge_R" in alignment.warnings

    duplicate = materialize_candidate_field(_candidate(candidate_id="candidate_year_copy")).field
    compressed, report = precompress_fields([materialize_candidate_field(_candidate()).field, duplicate])
    assert len(compressed) == 1
    assert report.suppressed_count == 1
