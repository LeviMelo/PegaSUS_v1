from __future__ import annotations

import json

import pyarrow.parquet as pq

from pegasus.output.bundle_manager import OutputBundleManager
from pegasus.output.validate import validate_output_bundle


def test_core_vital_bundle_declares_profile_optional_empty_artifacts(tmp_path):
    run_dir = tmp_path / "profile_bundle"
    manager = OutputBundleManager(run_dir=run_dir)
    manager.set_json("UserIntent", {"run_profile": "core_vital", "budget": "fast"})
    manager.set_json("RunConfig", {"schema_version": "1.0", "run_profile": "core_vital"})
    manager.set_json(
        "ReproducibilityManifest",
        {
            "schema_version": "1.0",
            "run_profile": "core_vital",
            "telemetry": {"total_wall_seconds": 0.0, "stage_status": {}, "stage_wall_seconds": {}},
        },
    )
    manager.set_json("P_vector", {"schema_version": "1.0"})
    manager.set_table(
        "V_fields",
        [
            {
                "field_id": "field_a",
                "name": "Field A",
                "kind": "extensive_measure",
                "carrier": "Deaths",
                "unit": "counts",
                "aggregation": "additive",
                "role": json.dumps(["outcome"]),
                "source": json.dumps(["test"]),
                "support_json": "{}",
                "axes_json": "{}",
                "operator": "test",
                "provenance": json.dumps(["fixture"]),
                "state": "verified",
                "dashboard_safe": "False",
                "warnings": "[]",
                "lineage_hash": "lineage",
                "registry_hash": "registry",
                "materialization_state": "metadata_only",
                "path": "",
            }
        ],
    )
    manager.set_table(
        "E_DAG",
        [
            {
                "edge_id": "edge_a",
                "parent_field_id": "field_a",
                "child_field_id": "field_a",
                "operator": "identity",
                "operator_params_json": "{}",
                "registry_versions_json": "{}",
                "created_at": "2026-06-27T00:00:00+00:00",
            }
        ],
    )
    manager.set_table(
        "Q_tensor",
        [
            {
                "field_id": "field_a",
                "n_events": 1.0,
                "n_denom": 1.0,
                "n_eff": 1.0,
                "cov_S": 1.0,
                "cov_T": 1.0,
                "missingness": 0.0,
                "zero_inflation": 0.0,
                "denom_fragility": 0.0,
                "cv": 0.0,
                "moran_i": 0.0,
                "temporal_roughness": 0.0,
                "spatial_entropy": 0.0,
                "provenance_risk": 0.0,
                "state": "verified",
                "dashboard_safe": "False",
                "warnings": "[]",
                "computed_at": "2026-06-27T00:00:00+00:00",
                "q_schema_version": "1.0",
            }
        ],
    )
    manager.set_table(
        "VariableDictionary",
        [
            {
                "field_id": "field_a",
                "display_name": "Field A",
                "technical_name": "field_a",
                "definition": "Test field.",
                "estimand_label": "count",
                "source_systems": "[]",
                "carrier": "Deaths",
                "unit": "counts",
                "support_description": "{}",
                "axis_description": "{}",
                "provenance_description": "[]",
                "state": "verified",
                "dashboard_safe": "False",
                "interpretation_warning": "",
            }
        ],
    )

    manager.flush_to_disk(run_dir)

    result = validate_output_bundle(run_dir=str(run_dir))
    assert result.ok, result.errors
    warnings = pq.read_table(run_dir / "Warnings.parquet").to_pylist()
    warning_ids = {row["warning_id"] for row in warnings}
    assert "empty_by_profile::core_vital::ModelAssociations" in warning_ids
    assert "empty_by_profile::core_vital::Maps" in warning_ids

