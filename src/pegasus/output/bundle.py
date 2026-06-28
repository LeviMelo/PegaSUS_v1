"""Compatibility entrypoints for creating a canonical output bundle."""

from __future__ import annotations

from pathlib import Path
import json

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.schema_seed import create_schema_seed_output_bundle
from pegasus.output.table_io import append_replace_rows


def _scaffold_rows() -> tuple[list[dict], list[dict], list[dict]]:
    field = {
        "field_id": "slice0_scaffold_field",
        "name": "Slice0ScaffoldField",
        "kind": "scaffold",
        "carrier": "Audit",
        "unit": "dimensionless",
        "aggregation": "non_aggregable",
        "role": '["schema_seed"]',
        "source": '["PegaSUS"]',
        "support_json": '{"support":"run"}',
        "axes_json": "{}",
        "operator": "schema_seed",
        "provenance": '["schema_seed"]',
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": '["schema_seed_scaffold"]',
        "lineage_hash": "schema_seed",
        "registry_hash": "schema_seed",
        "materialization_state": "metadata_only",
        "path": "",
    }
    q = {
        "field_id": "slice0_scaffold_field",
        "n_events": 0.0,
        "n_denom": 0.0,
        "n_eff": 0.0,
        "cov_S": 0.0,
        "cov_T": 0.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0,
        "cv": 0.0,
        "moran_i": 0.0,
        "temporal_roughness": 0.0,
        "spatial_entropy": 0.0,
        "provenance_risk": 0.0,
        "race_axis_source": "",
        "race_axis_target": "",
        "missing_race_share": 0.0,
        "emission_prior_strength": 0.0,
        "race_bridge_cv": 0.0,
        "sensitivity_width": 0.0,
        "bridge_mode": "",
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": '["schema_seed_scaffold"]',
        "computed_at": "schema_seed",
        "q_schema_version": "1.0",
    }
    vd = {
        "field_id": "slice0_scaffold_field",
        "display_name": "Slice0ScaffoldField",
        "technical_name": "schema_seed.slice0_scaffold_field",
        "definition": "Schema seed field for empty output bundle initialization.",
        "estimand_label": "schema_seed",
        "source_systems": '["PegaSUS"]',
        "carrier": "Audit",
        "unit": "dimensionless",
        "support_description": '{"support":"run"}',
        "axis_description": "{}",
        "provenance_description": '["schema_seed"]',
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "interpretation_warning": "schema_seed_scaffold",
    }
    return [field], [q], [vd]


def create_empty_output_bundle(run_dir: str | Path) -> Path:
    """Create the canonical empty 17-key run bundle."""
    root = create_schema_seed_output_bundle(run_dir)
    fields, q_rows, vd_rows = _scaffold_rows()
    append_replace_rows(root / "V_fields.parquet", fields, id_column="field_id")
    append_replace_rows(root / "Q_tensor.parquet", q_rows, id_column="field_id")
    append_replace_rows(root / "VariableDictionary.parquet", vd_rows, id_column="field_id")
    warnings = []
    required = {"V_fields", "E_DAG", "Q_tensor", "P_vector", "UserIntent", "VariableDictionary", "RunConfig", "ReproducibilityManifest"}
    for key in OUTPUT_BUNDLE_FILES:
        if key in required or key == "Warnings":
            continue
        warnings.append({
            "warning_id": f"empty_by_profile::core_vital::{key}",
            "field_id": "run",
            "source": "output_profile",
            "severity": "info",
            "code": "empty_by_profile",
            "message": f"{key} is empty because run_profile=core_vital does not require it.",
            "inherited_from": "[]",
            "created_at": "schema_seed",
        })
    append_replace_rows(root / "Warnings.parquet", warnings, id_column="warning_id")
    for name in ("UserIntent.json", "RunConfig.json", "ReproducibilityManifest.json"):
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_profile"] = "core_vital"
        if name == "ReproducibilityManifest.json":
            payload["source_hashes"] = {"schema_seed": "schema_seed"}
            payload["registry_hashes"] = {"schema_seed": "schema_seed"}
            payload["telemetry"] = {
                "total_wall_seconds": 0.0,
                "stage_status": {"schema_seed": "success"},
                "stage_wall_seconds": {"schema_seed": 0.0},
            }
        if name == "RunConfig.json":
            payload["source_hashes"] = {"schema_seed": "schema_seed"}
            payload["registry_hashes"] = {"schema_seed": "schema_seed"}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


__all__ = ["create_empty_output_bundle"]
