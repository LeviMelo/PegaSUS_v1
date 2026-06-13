from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from pegasus.core.hashing import sha256_file
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.population import registry_manifest
from pegasus.she.population.schema import PopulationTensorResult
from pegasus.she.population.solvers import solve_population_tensor_from_sidra_anchor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pq.read_table(path).schema
    shaped = [{name: row.get(name) for name in schema.names} for row in rows]
    table = pa.Table.from_pylist(shaped, schema=schema) if shaped else pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
    pq.write_table(table, path)



def _field(result: PopulationTensorResult) -> dict[str, Any]:
    support = {
        "PopulationTensorMode": result.mode,
        "SolverBackend": result.solver_backend,
        "SolverID": result.solver_id,
        "SparseJacobian": result.sparse_jacobian,
        "DenominatorFeedbackWarning": result.denominator_feedback_warning,
        "reconstruction_uncertainty": result.reconstruction_uncertainty,
        "locality_id": result.locality_id,
        "period": result.period,
        "n_events": result.value,
        "n_eff": result.value,
        "missingness": 0.0,
        "denom_fragility": result.reconstruction_uncertainty,
        "population_tensor_diagnostics": result.diagnostics.as_manifest(),
    }
    axes = {
        "geography_axis": "IBGE_COD7",
        "time_axis": "year",
        "population_strata_axis": "total",
        "population_tensor_mode": result.mode,
    }
    role = ["population_denominator_tensor", result.mode]
    source = ["SIDRA", "population_tensor"]
    provenance = ["official_sidra_anchor", "population_tensor", result.solver_id]
    warnings = list(result.warnings)
    return {
        "field_id": f"population_tensor_{result.mode}",
        "name": "PopulationTensorOptimizedIndependent" if result.mode == "independent_denominator" else "PopulationTensorOptimizedSIMInformed",
        "kind": "latent_context" if result.mode == "sim_informed_denominator" else "extensive_measure",
        "carrier": "Population",
        "unit": result.unit,
        "support_json": _compact(support),
        "axes_json": _compact(axes),
        "aggregation": "additive",
        "role": _compact(role),
        "role_json": _compact(role),
        "source": _compact(source),
        "source_json": _compact(source),
        "operator": "PopulationTensor/ProjectedGradientSmall",
        "provenance": _compact(provenance),
        "provenance_json": _compact(provenance),
        "state": result.state,
        "warnings": _compact(warnings),
        "warnings_json": _compact(warnings),
        "lineage_json": _compact({"parent_ids": [result.source_anchor_field_id], "operator": "population_tensor_solver", "created_at": _now(), "tensor_id": result.tensor_id}),
        "materialization_state": "materialized",
        "path": "Tables/population_tensor_diagnostics.parquet",
        "dashboard_safe": "warning" if result.warnings else "true",
    }


def _q_row(field: dict[str, Any], result: PopulationTensorResult) -> dict[str, Any]:
    warnings = list(result.warnings)
    return {
        "field_id": field["field_id"],
        "n_events": result.value,
        "n_denom": result.value,
        "n_eff": result.value,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": result.reconstruction_uncertainty,
        "cv": result.reconstruction_uncertainty,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.05 if result.mode == "independent_denominator" else 0.35,
        "race_axis_source": None,
        "race_axis_target": None,
        "missing_race_share": None,
        "emission_prior_strength": None,
        "race_bridge_cv": None,
        "sensitivity_width": None,
        "bridge_mode": None,
        "state": result.state,
        "dashboard_safe": field["dashboard_safe"],
        "warnings": _compact(warnings),
        "warnings_json": _compact(warnings),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }

def _vd_row(field: dict[str, Any], result: PopulationTensorResult) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "name": field["name"],
        "definition": "Population denominator tensor optimized under nonnegativity, closure, and registered demographic loss terms from an official SIDRA anchor.",
        "estimand": result.mode,
        "interpretation_warning": "Independent denominator mode fixes lambda_D=0 and does not use SIM feedback." if result.mode == "independent_denominator" else "SIM-informed mode requires feedback-risk handling; SIDRA-only requests record that the SIM prior was unavailable.",
        "unit": field["unit"],
        "source_system": "SIDRA",
        "carrier": field["carrier"],
        "role_json": field["role_json"],
        "state": field["state"],
        "dashboard_safe": field["dashboard_safe"],
    }


def _warning_rows(field: dict[str, Any], result: PopulationTensorResult) -> list[dict[str, Any]]:
    rows = [
        {
            "warning_id": "population_tensor_official_sidra_anchor_used",
            "field_id": field["field_id"],
            "source": "population_tensor",
            "severity": "info",
            "code": "independent_population_denominator_mode",
            "message": "Population tensor uses official SIDRA denominator independently from SIM events.",
            "inherited_from_json": "[]",
            "created_at": _now(),
        }
    ]
    if result.denominator_feedback_warning:
        rows.append(
            {
                "warning_id": "population_tensor_sim_feedback_risk",
                "field_id": field["field_id"],
                "source": "population_tensor",
                "severity": "warning",
                "code": "sim_informed_population_feedback_risk",
                "message": "SIM-informed denominator mode can feed numerator measurement noise back into denominator reconstruction; this SIDRA-only request had no SIM death prior.",
                "inherited_from_json": "[]",
                "created_at": _now(),
            }
        )
    return rows


def _failed_dense_branch(field: dict[str, Any]) -> dict[str, Any]:
    return {
        "failed_branch_id": "failed_dense_national_population_tensor_above_threshold",
        "attempted_operator": "dense_national_projected_gradient_population_tensor",
        "parent_field_ids": _compact([field["field_id"]]),
        "failure_stage": "SHE.population_tensor",
        "failed_terms": _compact(["dense_national_solver", "population_tensor_scale_gate"]),
        "reason": "Dense national population tensor projected-gradient path must abort above the configured scale threshold; sparse/block solver planning is required.",
        "warnings_json": _compact(["dense_national_population_tensor_aborted_above_threshold"]),
        "created_at": _now(),
    }


def write_population_tensor_fixture_bundle(
    *,
    sidra_facts_path: str | Path,
    run_dir: str | Path,
    mode: str = "independent_denominator",
) -> Path:
    sidra_facts_path = Path(sidra_facts_path)
    run_dir = Path(run_dir)
    create_empty_output_bundle(run_dir)
    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=sidra_facts_path, mode=mode)
    field = _field(result)
    q_rows = [_q_row(field, result)]
    vd_rows = [_vd_row(field, result)]
    warning_rows = _warning_rows(field, result)
    failed_rows = [_failed_dense_branch(field)]

    _write_rows_like(run_dir / "V_fields.parquet", [field])
    _write_rows_like(run_dir / "Q_tensor.parquet", q_rows)
    _write_rows_like(run_dir / "VariableDictionary.parquet", vd_rows)
    _write_rows_like(run_dir / "Warnings.parquet", warning_rows)
    _write_rows_like(run_dir / "FailedBranches.parquet", failed_rows)
    _write_rows_like(run_dir / "E_DAG.parquet", [])
    for name in [
        "ModelAssociations.parquet",
        "ResidualAssociations.parquet",
        "Hypotheses.parquet",
        "QuarantinedFields.parquet",
        "ForcedFields.parquet",
    ]:
        _write_rows_like(run_dir / name, [])

    (run_dir / "Tables").mkdir(exist_ok=True)
    pl.DataFrame([result.as_manifest()]).write_parquet(run_dir / "Tables" / "population_tensor_diagnostics.parquet")

    stage_status = {stage: "skipped" for stage in COMPILE_TELEMETRY_STAGES}
    stage_wall_seconds = {stage: 0.0 for stage in COMPILE_TELEMETRY_STAGES}
    for stage in ["sidra_fetch", "she_build", "population_solver", "efg_build", "q_tensor", "output_serialization", "output_validation"]:
        if stage in stage_status:
            stage_status[stage] = "success"
    for stage in ["stdfm", "pirs_model", "pirs_hsic"]:
        if stage in stage_status:
            stage_status[stage] = "blocked"
    telemetry = {
        "total_wall_seconds": 0.0,
        "stage_status": stage_status,
        "stage_wall_seconds": stage_wall_seconds,
        "stage_errors": {
            "stdfm": "ST-DFM is outside Slice 6A population tensor foundation.",
            "pirs_model": "PIRS modeling is outside Slice 6A population tensor foundation.",
            "pirs_hsic": "HSIC scanning is outside Slice 6A population tensor foundation.",
        },
        "resource_summary": {
            "peak_rss_mb": None,
            "peak_vram_mb": None,
            "duckdb_temp_bytes": None,
            "rows_read": {"sidra_facts": 1},
            "rows_written": {"V_fields": 1, "Q_tensor": 1, "VariableDictionary": 1, "FailedBranches": 1},
            "parquet_bytes_written": 0,
        },
    }

    population_manifest = result.as_manifest()
    population_manifest.update(
        {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "standalone_population_tensor",
            "field_id": field["field_id"],
            "field_name": field["name"],
            "source_hashes": {"sidra_facts": sha256_file(sidra_facts_path)},
            "independent_denominator_mode": result.mode == "independent_denominator",
            "sim_feedback_warning": bool(result.denominator_feedback_warning),
            "dashboard_safe": field.get("dashboard_safe"),
            "materialization_state": field.get("materialization_state"),
            "table_paths": {"diagnostics": "Tables/population_tensor_diagnostics.parquet"},
        }
    )
    user_intent = {
        "workflow": "slice6a_population_tensor_fixture",
        "population_tensor_mode": mode,
        "source_systems": ["SIDRA"],
    }
    run_config = {
        "schema_version": "1.0",
        "workflow": "slice6a_population_tensor_fixture",
        "source_systems": ["SIDRA"],
        "source_hashes": {"sidra_facts": sha256_file(sidra_facts_path)},
        "registry_hashes": {"population": registry_manifest()["registry_version"]},
        "population_tensor": population_manifest,
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "generated_at": _now(),
        "workflow": "slice6a_population_tensor_fixture",
        "source_hashes": run_config["source_hashes"],
        "registry_hashes": run_config["registry_hashes"],
        "telemetry": telemetry,
        "population_tensor": population_manifest,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    p_vector = {
        "source_systems": ["SIDRA"],
        "field_count": 1,
        "blocked_outputs": [row["failed_branch_id"] for row in failed_rows],
        "population_tensor": population_manifest,
        "provenance": {field["field_id"]: json.loads(field["provenance_json"])},
    }
    (run_dir / "UserIntent.json").write_text(_json(user_intent), encoding="utf-8")
    (run_dir / "RunConfig.json").write_text(_json(run_config), encoding="utf-8")
    (run_dir / "ReproducibilityManifest.json").write_text(_json(manifest), encoding="utf-8")
    (run_dir / "P_vector.json").write_text(_json(p_vector), encoding="utf-8")
    validate_output_bundle(run_dir=str(run_dir))
    return run_dir
