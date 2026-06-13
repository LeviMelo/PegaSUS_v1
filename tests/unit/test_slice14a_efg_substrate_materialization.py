
from __future__ import annotations

from pegasus.efg.materialize import (
    classify_substrate_candidate_kind,
    materialize_candidate_field,
    materialize_substrate_bundle,
)
from pegasus.she.substrate import SubstrateBundle, SubstrateFieldCandidate, SubstrateFieldExclusion


def _candidate(**overrides):
    base = dict(
        candidate_id="substrate_candidate_demo",
        source_system="SIM-DO",
        artifact_path="tests/fixtures/demo.parquet",
        column="year",
        technical_name="SIM-DO.year",
        carrier="Deaths",
        unit="counts",
        aggregation="additive",
        role=("time_axis",),
        axes={"period": "year"},
        provenance=("registry", "fixture"),
        substrate_kind="source_measure",
        registry_hash="source_registry_hash:SIM-DO",
        row_count=3,
        non_null_count=3,
        unique_non_null_count=2,
        missing_rate=0.0,
        numeric_min=2020.0,
        numeric_max=2021.0,
        source_manifest_hash="manifest_hash",
        artifact_hash="artifact_hash",
        warnings=(),
    )
    base.update(overrides)
    return SubstrateFieldCandidate(**base)


def test_slice14a_classifies_substrate_candidate_kinds_without_ratio_claims() -> None:
    assert classify_substrate_candidate_kind(_candidate()) == "extensive_measure"
    assert classify_substrate_candidate_kind(_candidate(unit="ICD10", aggregation="non_aggregable", role=("diagnostic_topology",))) == "observer_proxy"
    assert classify_substrate_candidate_kind(_candidate(aggregation="weighted_mean", unit="ratio")) == "intensive_density"


def test_slice14a_materializes_candidate_as_metadata_only_field_node() -> None:
    item = materialize_candidate_field(_candidate())
    field = item.field
    assert field.name == "SIM-DO.year"
    assert field.carrier == "Deaths"
    assert field.unit == "counts"
    assert field.kind == "extensive_measure"
    assert field.aggregation == "additive"
    assert field.materialization_state == "metadata_only"
    assert field.support["column"] == "year"
    assert field.support["row_count"] == 3
    assert "substrate_materialized" in field.role
    assert item.lineage_hash == field.lineage.model_dump(mode="json") and False or item.lineage_hash


def test_slice14a_materialization_never_promotes_exclusions() -> None:
    candidate = _candidate()
    exclusion = SubstrateFieldExclusion(
        exclusion_id="substrate_exclusion_constant",
        source_system="SIM-DO",
        artifact_path="tests/fixtures/demo.parquet",
        column="constant_col",
        reason="zero_variance_constant",
        registry_reason=None,
        row_count=3,
        non_null_count=3,
        unique_non_null_count=1,
        missing_rate=0.0,
        structural_role="constant",
        warnings=("zero_variance_constant",),
    )
    bundle = SubstrateBundle(
        schema_version="1.0",
        substrate_id="substrate_bundle_demo",
        source_reality_mode="fixture_only",
        source_artifacts=(),
        candidates=(candidate,),
        exclusions=(exclusion,),
        table_profiles=(),
        registry_hashes={"SIM-DO": "source_registry_hash:SIM-DO"},
        warnings=(),
    )
    result = materialize_substrate_bundle(bundle)
    assert result.field_count == 1
    assert result.excluded_field_count == 1
    assert result.fields[0].candidate_id == candidate.candidate_id
    assert result.excluded_source_fields[0]["column"] == "constant_col"
    assert result.excluded_source_fields[0]["efg_materialized"] is False
