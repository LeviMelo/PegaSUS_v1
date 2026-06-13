from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.efg.compile_attach import attach_autonomous_efg_to_run
from pegasus.efg.dag import build_efg
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.she.substrate import SubstrateBundle, SubstrateFieldCandidate, SubstrateFieldExclusion


def _candidate() -> SubstrateFieldCandidate:
    return SubstrateFieldCandidate(
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
        row_count=3,
        non_null_count=3,
        unique_non_null_count=2,
        missing_rate=0.0,
        numeric_min=2021.0,
        numeric_max=2022.0,
        source_manifest_hash="source-manifest-hash",
        artifact_hash="artifact-hash",
        warnings=(),
    )


def _bundle() -> SubstrateBundle:
    exclusion = SubstrateFieldExclusion(
        exclusion_id="excluded_event_id",
        source_system="SIM-DO",
        artifact_path="tests/fixtures/sim.parquet",
        column="event_id",
        reason="structural_or_audit_only",
        registry_reason="registry_not_admissible",
        row_count=3,
        non_null_count=3,
        unique_non_null_count=3,
        missing_rate=0.0,
        structural_role="identifier",
        warnings=("registry_not_admissible",),
    )
    return SubstrateBundle(
        schema_version="1.0",
        substrate_id="substrate_19b",
        source_reality_mode="fixture_only",
        source_artifacts=(),
        candidates=(_candidate(),),
        exclusions=(exclusion,),
        table_profiles=(),
        registry_hashes={"SIM-DO": "source-registry-hash"},
        warnings=(),
    )


def test_slice19b_attaches_graph_tables_without_removing_existing_fields(tmp_path: Path) -> None:
    run = create_empty_output_bundle(tmp_path / "run")
    result = build_efg(substrate=_bundle())
    attached = attach_autonomous_efg_to_run(run_dir=run, result=result, validate=False)

    fields = pq.read_table(run / "V_fields.parquet").to_pylist()
    field_ids = {str(row["field_id"]) for row in fields}
    assert "slice0_scaffold_field" in field_ids
    assert {field.id for field in result.fields} <= field_ids
    assert all(
        row["state"] == "quarantined_descriptive"
        for row in fields
        if row["field_id"] in {field.id for field in result.fields}
    )
    assert pq.read_table(run / "E_DAG.parquet").num_rows == result.edge_count
    assert pq.read_table(run / "FailedBranches.parquet").num_rows == len(result.failed_branches)
    assert (run / attached.manifest_path).exists()
    payload = json.loads((run / attached.manifest_path).read_text(encoding="utf-8"))
    assert payload["efg_id"] == result.efg_id
    assert set(path.name for path in run.iterdir()) == {
        "V_fields.parquet", "E_DAG.parquet", "Q_tensor.parquet", "P_vector.json",
        "UserIntent.json", "Warnings.parquet", "ModelAssociations.parquet",
        "ResidualAssociations.parquet", "Hypotheses.parquet", "Tables", "Maps",
        "VariableDictionary.parquet", "FailedBranches.parquet", "QuarantinedFields.parquet",
        "ForcedFields.parquet", "RunConfig.json", "ReproducibilityManifest.json",
    }
