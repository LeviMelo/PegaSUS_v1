
from __future__ import annotations

from pathlib import Path

from pegasus.efg.materialization_manifest import (
    build_efg_materialization_manifest,
    substrate_bundle_from_manifest,
    write_efg_materialization_manifest,
)
from pegasus.she.substrate import (
    SubstrateBundle,
    SubstrateFieldCandidate,
    SubstrateFieldExclusion,
    write_substrate_bundle_manifest,
)


def _candidate(column: str = "underlying_icd_norm") -> SubstrateFieldCandidate:
    return SubstrateFieldCandidate(
        candidate_id=f"substrate_candidate_{column}",
        source_system="SIM-DO",
        artifact_path="tests/fixtures/demo.parquet",
        column=column,
        technical_name=f"SIM-DO.{column}",
        carrier="Deaths",
        unit="ICD10",
        aggregation="non_aggregable",
        role=("diagnostic_topology",),
        axes={"diagnosis": "icd10"},
        provenance=("registry", "fixture"),
        substrate_kind="source_measure",
        registry_hash="source_registry_hash:SIM-DO",
        row_count=3,
        non_null_count=3,
        unique_non_null_count=3,
        missing_rate=0.0,
        numeric_min=None,
        numeric_max=None,
        source_manifest_hash="manifest_hash",
        artifact_hash="artifact_hash",
        warnings=(),
    )


def _bundle() -> SubstrateBundle:
    exclusion = SubstrateFieldExclusion(
        exclusion_id="substrate_exclusion_constant_col",
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
    return SubstrateBundle(
        schema_version="1.0",
        substrate_id="substrate_bundle_demo",
        source_reality_mode="fixture_only",
        source_artifacts=(),
        candidates=(_candidate(),),
        exclusions=(exclusion,),
        table_profiles=(),
        registry_hashes={"SIM-DO": "source_registry_hash:SIM-DO"},
        warnings=(),
    )


def test_slice14b_reconstructs_substrate_bundle_from_manifest_dict() -> None:
    payload = _bundle().as_manifest()
    reconstructed = substrate_bundle_from_manifest(payload)
    assert reconstructed.substrate_id == "substrate_bundle_demo"
    assert len(reconstructed.candidates) == 1
    assert len(reconstructed.exclusions) == 1
    assert reconstructed.candidates[0].column == "underlying_icd_norm"
    assert reconstructed.exclusions[0].column == "constant_col"


def test_slice14b_writes_metadata_only_efg_manifest(tmp_path: Path) -> None:
    substrate_path = tmp_path / "substrate_manifest.json"
    output = tmp_path / "efg_substrate_materialization.json"
    write_substrate_bundle_manifest(_bundle(), substrate_path)
    payload = write_efg_materialization_manifest(substrate_manifest=substrate_path, output=output)

    assert output.exists()
    assert payload["metadata_only"] is True
    assert payload["writes_v_fields"] is False
    assert payload["writes_e_dag"] is False
    assert payload["field_count"] == 1
    assert payload["excluded_field_count"] == 1
    assert payload["fields"][0]["field"]["unit"] == "ICD10"
    assert payload["fields"][0]["field"]["kind"] == "observer_proxy"
    assert payload["excluded_source_fields"][0]["efg_materialized"] is False
    assert payload["summary"]["field_columns"] == ["underlying_icd_norm"]
    assert payload["summary"]["excluded_columns"] == ["constant_col"]


def test_slice14b_build_manifest_is_deterministic(tmp_path: Path) -> None:
    substrate_path = tmp_path / "substrate_manifest.json"
    write_substrate_bundle_manifest(_bundle(), substrate_path)
    first = build_efg_materialization_manifest(substrate_manifest=substrate_path)
    second = build_efg_materialization_manifest(substrate_manifest=substrate_path)
    assert first["materialization_id"] == second["materialization_id"]
