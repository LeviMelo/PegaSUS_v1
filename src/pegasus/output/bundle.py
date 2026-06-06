from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_table(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def create_empty_output_bundle(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Tables").mkdir(exist_ok=True)
    (run_dir / "Maps").mkdir(exist_ok=True)

    field_id = "slice0_scaffold_field"

    v_schema = pa.schema([
        ("field_id", pa.string()),
        ("name", pa.string()),
        ("kind", pa.string()),
        ("carrier", pa.string()),
        ("unit", pa.string()),
        ("aggregation", pa.string()),
        ("role", pa.string()),
        ("source", pa.string()),
        ("support_json", pa.string()),
        ("axes_json", pa.string()),
        ("operator", pa.string()),
        ("provenance", pa.string()),
        ("state", pa.string()),
        ("dashboard_safe", pa.string()),
        ("warnings", pa.string()),
        ("lineage_hash", pa.string()),
        ("registry_hash", pa.string()),
        ("materialization_state", pa.string()),
        ("path", pa.string()),
    ])
    _write_table(run_dir / "V_fields.parquet", [{
        "field_id": field_id,
        "name": "Slice 0 Scaffold Field",
        "kind": "observer_proxy",
        "carrier": "none",
        "unit": "none",
        "aggregation": "non_aggregable",
        "role": json.dumps(["model_only"]),
        "source": json.dumps(["scaffold"]),
        "support_json": json.dumps({}),
        "axes_json": json.dumps({}),
        "operator": None,
        "provenance": json.dumps(["synthetic"]),
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": json.dumps(["slice0_scaffold_only"]),
        "lineage_hash": "slice0",
        "registry_hash": "uncomputed",
        "materialization_state": "metadata_only",
        "path": None,
    }], v_schema)

    edge_schema = pa.schema([
        ("edge_id", pa.string()),
        ("parent_field_id", pa.string()),
        ("child_field_id", pa.string()),
        ("operator", pa.string()),
        ("operator_params_json", pa.string()),
        ("registry_versions_json", pa.string()),
        ("created_at", pa.string()),
    ])
    _write_table(run_dir / "E_DAG.parquet", [], edge_schema)

    q_schema = pa.schema([
        ("field_id", pa.string()),
        ("n_events", pa.float64()),
        ("n_denom", pa.float64()),
        ("n_eff", pa.float64()),
        ("cov_S", pa.float64()),
        ("cov_T", pa.float64()),
        ("missingness", pa.float64()),
        ("zero_inflation", pa.float64()),
        ("denom_fragility", pa.float64()),
        ("cv", pa.float64()),
        ("moran_i", pa.float64()),
        ("temporal_roughness", pa.float64()),
        ("spatial_entropy", pa.float64()),
        ("provenance_risk", pa.float64()),
        ("race_axis_source", pa.string()),
        ("race_axis_target", pa.string()),
        ("missing_race_share", pa.float64()),
        ("emission_prior_strength", pa.float64()),
        ("race_bridge_cv", pa.float64()),
        ("sensitivity_width", pa.float64()),
        ("bridge_mode", pa.string()),
        ("state", pa.string()),
        ("dashboard_safe", pa.string()),
        ("warnings", pa.string()),
        ("computed_at", pa.string()),
        ("q_schema_version", pa.string()),
    ])
    _write_table(run_dir / "Q_tensor.parquet", [{
        "field_id": field_id,
        "n_events": 0.0,
        "n_denom": 0.0,
        "n_eff": 0.0,
        "cov_S": 0.0,
        "cov_T": 0.0,
        "missingness": 1.0,
        "zero_inflation": 0.0,
        "denom_fragility": 1.0,
        "cv": None,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 1.0,
        "race_axis_source": None,
        "race_axis_target": None,
        "missing_race_share": None,
        "emission_prior_strength": None,
        "race_bridge_cv": None,
        "sensitivity_width": None,
        "bridge_mode": None,
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": json.dumps(["slice0_scaffold_only"]),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }], q_schema)

    warn_schema = pa.schema([
        ("warning_id", pa.string()),
        ("field_id", pa.string()),
        ("source", pa.string()),
        ("severity", pa.string()),
        ("code", pa.string()),
        ("message", pa.string()),
        ("inherited_from", pa.string()),
        ("created_at", pa.string()),
    ])
    _write_table(run_dir / "Warnings.parquet", [{
        "warning_id": "slice0_scaffold_only",
        "field_id": field_id,
        "source": "pegasus.output.bundle",
        "severity": "info",
        "code": "slice0_scaffold_only",
        "message": "Slice 0 scaffold bundle; no domain computation has run.",
        "inherited_from": json.dumps([]),
        "created_at": _now(),
    }], warn_schema)

    empty_assoc_schema = pa.schema([
        ("id", pa.string()),
        ("status", pa.string()),
        ("warnings", pa.string()),
    ])
    for name in ["ModelAssociations", "ResidualAssociations"]:
        _write_table(run_dir / f"{name}.parquet", [], empty_assoc_schema)

    hyp_schema = pa.schema([
        ("hypothesis_id", pa.string()),
        ("outcome_field_id", pa.string()),
        ("covariate_field_id", pa.string()),
        ("residual_field_id", pa.string()),
        ("statistic", pa.float64()),
        ("p_value", pa.float64()),
        ("q_value", pa.float64()),
        ("hsic_mode", pa.string()),
        ("residual_mode", pa.string()),
        ("fold_scheme", pa.string()),
        ("bootstrap_count", pa.int64()),
        ("residual_uncertainty", pa.string()),
        ("null_strategy", pa.string()),
        ("fdr_method", pa.string()),
        ("n_eff", pa.float64()),
        ("state", pa.string()),
        ("warnings", pa.string()),
        ("approximation_diagnostics_json", pa.string()),
    ])
    _write_table(run_dir / "Hypotheses.parquet", [], hyp_schema)

    vd_schema = pa.schema([
        ("field_id", pa.string()),
        ("display_name", pa.string()),
        ("technical_name", pa.string()),
        ("definition", pa.string()),
        ("estimand_label", pa.string()),
        ("source_systems", pa.string()),
        ("carrier", pa.string()),
        ("unit", pa.string()),
        ("support_description", pa.string()),
        ("axis_description", pa.string()),
        ("provenance_description", pa.string()),
        ("state", pa.string()),
        ("dashboard_safe", pa.string()),
        ("interpretation_warning", pa.string()),
    ])
    _write_table(run_dir / "VariableDictionary.parquet", [{
        "field_id": field_id,
        "display_name": "Slice 0 Scaffold Field",
        "technical_name": "slice0_scaffold_field",
        "definition": "Non-analytic placeholder used to validate the output contract.",
        "estimand_label": "scaffold_non_estimand",
        "source_systems": json.dumps(["scaffold"]),
        "carrier": "none",
        "unit": "none",
        "support_description": "No epidemiological support.",
        "axis_description": "No epidemiological axes.",
        "provenance_description": "Synthetic scaffold artifact.",
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "interpretation_warning": "Not an epidemiological field.",
    }], vd_schema)

    fb_schema = pa.schema([
        ("failed_branch_id", pa.string()),
        ("attempted_operator", pa.string()),
        ("parent_field_ids", pa.string()),
        ("failure_stage", pa.string()),
        ("failed_terms", pa.string()),
        ("reason", pa.string()),
        ("warnings", pa.string()),
        ("created_at", pa.string()),
    ])
    _write_table(run_dir / "FailedBranches.parquet", [], fb_schema)

    qf_schema = pa.schema([
        ("field_id", pa.string()),
        ("state", pa.string()),
        ("reason", pa.string()),
        ("warnings", pa.string()),
    ])
    _write_table(run_dir / "QuarantinedFields.parquet", [{
        "field_id": field_id,
        "state": "quarantined_descriptive",
        "reason": "slice0_scaffold_only",
        "warnings": json.dumps(["synthetic_placeholder"]),
    }], qf_schema)
    _write_table(run_dir / "ForcedFields.parquet", [], qf_schema)

    (run_dir / "P_vector.json").write_text(json.dumps({
        "schema_version": "1.0",
        "provenance": {"slice0_scaffold_field": ["synthetic"]},
    }, indent=2), encoding="utf-8")

    (run_dir / "UserIntent.json").write_text(json.dumps({
        "frozen": True,
        "intent_source": "slice0_scaffold",
    }, indent=2), encoding="utf-8")

    (run_dir / "RunConfig.json").write_text(json.dumps({
        "frozen": True,
        "budget": "fast",
        "geo_mode": "native",
        "slice": "0",
    }, indent=2), encoding="utf-8")

    stages = [
        "config_load", "registry_validation", "datasus_acquire", "datasus_profile",
        "datasus_normalize", "sidra_metadata", "sidra_plan", "sidra_fetch",
        "sidra_normalize", "geo_support", "she_build", "population_solver",
        "stdfm", "efg_build", "q_tensor", "pirs_model", "pirs_hsic",
        "output_serialization", "output_validation",
    ]
    manifest = {
        "run_id": run_dir.name,
        "created_at": _now(),
        "completed_at": _now(),
        "status": "success",
        "code_version": {
            "package_version": "0.1.0",
            "git_commit": "uninitialized",
            "git_dirty": False,
        },
        "environment": {
            "python_version": sys.version,
            "os": platform.platform(),
            "duckdb_version": None,
            "polars_version": None,
            "pyarrow_version": pa.__version__,
            "torch_version": None,
            "torch_cuda_available": False,
            "cuda_device_name": None,
            "r_version": None,
            "microdatasus_version": None,
            "read_dbc_version": None,
        },
        "registry_hashes": {},
        "source_manifest_hashes": [],
        "random_seeds": {},
        "telemetry": {
            "total_wall_seconds": 0.0,
            "stage_wall_seconds": {f"{s}_seconds": 0.0 for s in stages},
            "stage_status": {
                s: ("success" if s in {"config_load", "registry_validation", "output_serialization", "output_validation"} else "skipped")
                for s in stages
            },
            "resource_summary": {
                "peak_rss_mb": None,
                "peak_vram_mb": None,
                "duckdb_temp_bytes": None,
                "rows_read": {},
                "rows_written": {},
                "parquet_bytes_written": 0,
            },
        },
    }
    (run_dir / "ReproducibilityManifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    expected = set(OUTPUT_BUNDLE_FILES.values())
    for child in run_dir.iterdir():
        if child.name not in expected:
            raise RuntimeError(f"Unexpected first-class output artifact: {child}")

    return run_dir
