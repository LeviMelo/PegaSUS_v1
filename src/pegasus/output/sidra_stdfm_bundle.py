from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import sha256_file, sha256_text, stable_json
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.cnes_sih_efg_bundle import write_rows_like
from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES
from pegasus.sidra.projection import load_projection_matrix, projection_metadata
from pegasus.sidra.stitching import SIDRASegment, stitch_sidra_longitudinal_segments
from pegasus.she.high_dimensional import bound_high_dimensional_sidra_exposure
from pegasus.she.stdfm.artifacts import materialize_stdfm_fit
from pegasus.she.stdfm.objective import stdfm_objective_pseudocode_contract
from pegasus.she.stdfm.schema import STDFMFitResult, STDFMProblem, build_stdfm_input_schema
from pegasus.she.stdfm.torch_solver import solve_stdfm


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _bool_string(value: bool) -> str:
    return "True" if bool(value) else "False"


def _support(periods: list[str], localities: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "years": [int(p) for p in periods if str(p).isdigit()],
        "periods": [str(p) for p in periods],
        "municipalities_ibge_cod7": [str(x) for x in localities],
        "n_events": None,
        "n_denom": None,
        "missingness": 0.0,
        "denom_fragility": 0.0,
    }
    if extra:
        payload.update(extra)
    return payload


def _field(
    *,
    field_id: str,
    name: str,
    kind: str,
    carrier: str,
    unit: str,
    aggregation: str,
    role: list[str],
    source: list[str],
    support: dict[str, Any],
    axes: dict[str, Any],
    provenance: list[str],
    warnings: list[str],
    state: str,
    dashboard_safe: bool,
    materialization_state: str,
    operator: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    schema_support = dict(support)
    if metadata:
        schema_support["field_metadata"] = metadata
    lineage = {
        "field_id": field_id,
        "operator": operator,
        "support": schema_support,
        "axes": axes,
        "metadata": metadata,
    }
    return {
        "field_id": field_id,
        "name": name,
        "technical_name": field_id,
        "kind": kind,
        "carrier": carrier,
        "unit": unit,
        "aggregation": aggregation,
        "role": _json(role),
        "role_json": _json(role),
        "source": _json(source),
        "source_json": _json(source),
        "support": _json(schema_support),
        "support_json": _json(schema_support),
        "axes": _json(axes),
        "axes_json": _json(axes),
        "lineage": _json(lineage),
        "lineage_json": _json(lineage),
        "lineage_hash": sha256_text(stable_json(lineage)),
        "registry_hash": "slice7a_sidra_stdfm_contract_v1",
        "provenance": _json(provenance),
        "provenance_json": _json(provenance),
        "warnings": _json(warnings),
        "warnings_json": _json(warnings),
        "state": state,
        "dashboard_safe": _bool_string(dashboard_safe),
        "materialization_state": materialization_state,
        "operator": operator,
        "metadata": _json(metadata),
        "metadata_json": _json(metadata),
        "path": None,
    }


def _q(field: dict[str, Any], *, provenance_risk: float, warnings: list[str]) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "n_events": None,
        "n_denom": None,
        "n_eff": None,
        "cov_S": None,
        "cov_T": None,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0,
        "cv": None,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": float(provenance_risk),
        "race_axis_source": None,
        "race_axis_target": None,
        "missing_race_share": None,
        "emission_prior_strength": None,
        "race_bridge_cv": None,
        "sensitivity_width": None,
        "bridge_mode": None,
        "state": field["state"],
        "dashboard_safe": field["dashboard_safe"],
        "warnings": _json(warnings),
        "warnings_json": _json(warnings),
        "q_json": _json({"provenance_risk": provenance_risk, "warnings": warnings}),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def _vd(field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "display_name": field["name"],
        "name": field["name"],
        "technical_name": field["technical_name"],
        "definition": definition,
        "estimand_label": estimand,
        "estimand": estimand,
        "source_systems": field["source"],
        "carrier": field["carrier"],
        "unit": field["unit"],
        "support_description": field["support_json"],
        "axis_description": field["axes_json"],
        "provenance_description": field["provenance"],
        "state": field["state"],
        "dashboard_safe": field["dashboard_safe"],
        "interpretation_warning": warning,
    }


def _warning(
    warning_id: str,
    field_id: str,
    code: str,
    message: str,
    severity: str = "warning",
    inherited_from: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "warning_id": warning_id,
        "field_id": field_id,
        "source": "slice7a_sidra_stdfm",
        "severity": severity,
        "code": code,
        "message": message,
        "inherited_from": _json(inherited_from or []),
        "created_at": _now(),
    }


def _failed(
    branch_id: str,
    operator: str,
    parents: list[str],
    reason: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "failed_branch_id": branch_id,
        "branch_id": branch_id,
        "attempted_operator": operator,
        "operator": operator,
        "parent_field_ids": _json(parents),
        "parents": _json(parents),
        "failure_stage": "SHE",
        "stage": "SHE",
        "failed_terms": _json(parents),
        "terms": _json(parents),
        "reason": reason,
        "warnings": _json(warnings),
        "created_at": _now(),
    }


def _empty_optional_tables(run_dir: Path) -> None:
    for name in [
        "ModelAssociations.parquet",
        "ResidualAssociations.parquet",
        "Hypotheses.parquet",
        "QuarantinedFields.parquet",
        "ForcedFields.parquet",
    ]:
        write_rows_like(run_dir / name, [])


def _write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pl.DataFrame(rows).write_parquet(path)
    else:
        pl.DataFrame([]).write_parquet(path)


def solve_sidra_stdfm_fixture(payload: dict[str, Any]):
    periods = sorted(
        {
            str(period)
            for segment in payload["stitching"]["segments"]
            for period in segment["periods"]
        }
    )
    localities = [str(value) for value in payload.get("localities", ["2704302"])]
    spec = payload["stdfm"]
    field_ids = tuple(str(value) for value in spec["field_ids"])
    raw_observations = spec["observations"]
    observations = tuple(0.0 if value is None else float(value) for value in raw_observations)
    input_schema = build_stdfm_input_schema(
        field_id="sidra_stdfm_candidate",
        concept_id="sidra_stdfm_context",
        support=_support(periods, localities),
        periods=periods,
        localities=localities,
        transform="mixed_registered",
        dynamics="continuous",
        projection_matrix_id=payload["projection"]["matrix_id"],
        stitch_metadata={"status": "stitched"},
        warnings=["stdfm_certification_required"],
    )
    problem = STDFMProblem(
        field_ids=field_ids,
        shape=(len(localities), len(periods), len(field_ids)),
        observations=observations,
        observed_mask=tuple(bool(value) for value in spec["observed_mask"]),
        validation_mask=(
            tuple(bool(value) for value in spec["validation_mask"])
            if spec.get("validation_mask") is not None
            else None
        ),
        link_function_by_field=tuple(str(value) for value in spec["link_function_by_field"]),
        n_factors=int(spec.get("n_factors", 1)),
        spatial_laplacian=(
            tuple(float(value) for value in spec["spatial_laplacian"])
            if spec.get("spatial_laplacian") is not None
            else None
        ),
        gamma_temporal=float(spec.get("gamma_temporal", 0.1)),
        gamma_spatial=float(spec.get("gamma_spatial", 0.1)),
        gamma_transition=float(spec.get("gamma_transition", 0.1)),
        multi_starts=int(spec.get("multi_starts", 3)),
        seed=int(spec.get("seed", 1729)),
    )
    result = solve_stdfm(input_schema, problem=problem, allow_uncertified=True)
    if not isinstance(result, STDFMFitResult):
        raise RuntimeError(f"ST-DFM fixture unexpectedly blocked: {result.reason}")
    return input_schema, problem, result


def build_sidra_stdfm_fixture_bundle(*, input_path: str | Path, run_dir: str | Path) -> Path:
    input_path = Path(input_path)
    run_dir = Path(run_dir)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    create_empty_output_bundle(run_dir)

    segments = [SIDRASegment.from_mapping(x) for x in payload["stitching"]["segments"]]
    stitch = stitch_sidra_longitudinal_segments(segments)
    projection_matrix = load_projection_matrix(payload["projection"])
    projection = projection_metadata(measure_kind="additive", has_denominator=False, matrix=projection_matrix)
    high_dim = bound_high_dimensional_sidra_exposure(
        raw_axes=payload["high_dimensional"]["raw_axes"],
        demanded_axes=payload["high_dimensional"]["demanded_axes"],
        axis_cardinalities={k: int(v) for k, v in payload["high_dimensional"]["axis_cardinalities"].items()},
        aggregation=payload["high_dimensional"].get("aggregation", "additive"),
        high_dimensional=True,
    )
    periods = list(stitch.stitched_periods)
    localities = [str(x) for x in payload.get("localities", ["2704302"])]
    base_support = _support(periods, localities)
    stitch_metadata = stitch.as_manifest()

    stdfm_input, stdfm_problem, stdfm_fit = solve_sidra_stdfm_fixture(payload)
    stdfm_output = stdfm_fit.output
    certification = stdfm_fit.certification

    fields: list[dict[str, Any]] = [
        _field(
            field_id="sidra_stitched_gdp_context",
            name="SIDRA stitched GDP context",
            kind="latent_context",
            carrier="municipality_year",
            unit="R$",
            aggregation="additive",
            role=["context", "sidra_stitched"],
            source=["SIDRA"],
            support=base_support,
            axes={"locality": "IBGE7", "time": "year", "concept": stitch.concept_id},
            provenance=["SIDRA", "sidra_longitudinal_stitching"],
            warnings=list(stitch.warnings),
            state="fragile" if stitch.warnings else "verified",
            dashboard_safe=not bool(stitch.warnings),
            materialization_state="materialized",
            operator="sidra_longitudinal_stitch",
            metadata={"stitch_metadata": stitch_metadata},
        ),
        _field(
            field_id="sidra_projected_labor_context",
            name="SIDRA projected labor context",
            kind="latent_context",
            carrier="municipality_year_sector",
            unit="count",
            aggregation="additive",
            role=["context", "sidra_projection"],
            source=["SIDRA"],
            support=base_support,
            axes={"locality": "IBGE7", "time": "year", "projection_axis": projection_matrix.target_axis},
            provenance=["SIDRA", "sidra_classification_projection"],
            warnings=list(projection["warnings"]),
            state="fragile" if projection["warnings"] else "verified",
            dashboard_safe=False if projection["warnings"] else True,
            materialization_state="materialized",
            operator="sidra_classification_projection",
            metadata={"projection_matrix_id": projection_matrix.matrix_id, "projection": projection},
        ),
        _field(
            field_id="sidra_highdim_bounded_context",
            name="SIDRA high-dimensional bounded context",
            kind="latent_context",
            carrier="municipality_year_bounded_context",
            unit="index",
            aggregation="additive",
            role=["context", "sidra_high_dimensional_bounded"],
            source=["SIDRA"],
            support=base_support,
            axes={"raw_axes": list(high_dim.raw_axes), "exposed_axes": list(high_dim.exposed_axes)},
            provenance=["SIDRA", "high_dimensional_bounded_pushforward"],
            warnings=list(high_dim.warnings),
            state="fragile",
            dashboard_safe=False,
            materialization_state="materialized",
            operator="high_dimensional_bounded_pushforward",
            metadata={"high_dimensional_bound": high_dim.as_manifest()},
        ),
        _field(
            field_id=stdfm_input.field_id,
            name="ST-DFM latent reconstruction candidate",
            kind="latent_context",
            carrier="municipality_year_context",
            unit="index",
            aggregation="model_based",
            role=["context", "stdfm_candidate"],
            source=["SIDRA"],
            support=base_support,
            axes={"locality": "IBGE7", "time": "year", "latent_factor": "F"},
            provenance=["SIDRA", "ST-DFM", stdfm_output.solver_backend],
            warnings=list(stdfm_output.warnings),
            state="verified" if stdfm_output.status == "verified" else "fragile",
            dashboard_safe=False,
            materialization_state="materialized",
            operator="ST-DFM",
            metadata={"stdfm_input": stdfm_input.as_manifest(), "stdfm_output": stdfm_output.as_manifest()},
        ),
    ]

    q_rows = [
        _q(fields[0], provenance_risk=0.35 if fields[0]["state"] == "fragile" else 0.05, warnings=json.loads(fields[0]["warnings"])),
        _q(fields[1], provenance_risk=0.45 if json.loads(fields[1]["warnings"]) else 0.05, warnings=json.loads(fields[1]["warnings"])),
        _q(fields[2], provenance_risk=0.55, warnings=json.loads(fields[2]["warnings"])),
        _q(fields[3], provenance_risk=0.95, warnings=json.loads(fields[3]["warnings"])),
    ]
    vd_rows = [
        _vd(fields[0], "SIDRA field stitched across table/variable segments with explicit segment provenance.", "stitched SIDRA contextual field", "Use with segment-provenance warning when table identity changes."),
        _vd(fields[1], "SIDRA field projected across classification axes with registered projection matrix.", "projected SIDRA contextual field", "Fractional projections are fragile unless externally validated."),
        _vd(fields[2], "High-dimensional SIDRA context exposed only after bounded pushforward.", "bounded high-dimensional context", "Dropped axes must remain visible in metadata."),
        _vd(fields[3], "ST-DFM latent reconstruction from masked SIDRA context observations.", "latent factor reconstruction", "Uncertified fits remain fragile and non-dashboard-safe."),
    ]
    warnings = [
        _warning("w_sidra_stitch", fields[0]["field_id"], "sidra_stitch_segment_provenance", "SIDRA stitching preserved table/variable segment provenance."),
        _warning("w_sidra_projection", fields[1]["field_id"], "sidra_fractional_classification_projection", "Fractional classification projection emits fragile state."),
        _warning("w_sidra_highdim", fields[2]["field_id"], "high_dimensional_bounded_pushforward", "High-dimensional SIDRA field was bounded before EFG exposure."),
        _warning("w_stdfm_certification", fields[3]["field_id"], "stdfm_certification_required", "ST-DFM fit completed without independent holdout certification."),
    ]
    failed = [
        _failed(
            "fb_unbounded_high_dimensional_sidra",
            "direct_high_dimensional_efg_exposure",
            [fields[2]["field_id"]],
            "High-dimensional SIDRA exposure without bounded pushforward is illegal.",
            ["high_dimensional_bounded_pushforward"],
        ),
    ]

    write_rows_like(run_dir / "V_fields.parquet", fields)
    write_rows_like(run_dir / "Q_tensor.parquet", q_rows)
    write_rows_like(run_dir / "VariableDictionary.parquet", vd_rows)
    write_rows_like(run_dir / "Warnings.parquet", warnings)
    write_rows_like(run_dir / "FailedBranches.parquet", failed)
    write_rows_like(run_dir / "E_DAG.parquet", [])
    _empty_optional_tables(run_dir)

    tables = run_dir / "Tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_parquet_rows(tables / "sidra_stitching_segments.parquet", [dict(x) for x in stitch.segment_provenance])
    _write_parquet_rows(tables / "sidra_projection_matrix.parquet", projection_matrix.as_rows())
    _write_parquet_rows(tables / "sidra_high_dimensional_bounds.parquet", [high_dim.as_manifest()])
    stdfm_fit = materialize_stdfm_fit(stdfm_fit, stdfm_problem, output_dir=tables)
    stdfm_output = stdfm_fit.output
    _write_parquet_rows(
        tables / "stdfm_objective_contract.parquet",
        [{"contract_json": _json(stdfm_objective_pseudocode_contract())}],
    )

    stage_status = {stage: "skipped" for stage in COMPILE_TELEMETRY_STAGES}
    stage_wall_seconds = {stage: 0.0 for stage in COMPILE_TELEMETRY_STAGES}
    for stage in ["sidra_fetch", "she_build", "efg_build", "q_tensor", "output_serialization", "output_validation"]:
        if stage in stage_status:
            stage_status[stage] = "success"
    if "stdfm" in stage_status:
        stage_status["stdfm"] = "success"
    telemetry = {
        "total_wall_seconds": 0.0,
        "stage_status": stage_status,
        "stage_wall_seconds": stage_wall_seconds,
        "stage_errors": {},
        "resource_summary": {
            "peak_rss_mb": None,
            "peak_vram_mb": None,
            "duckdb_temp_bytes": None,
            "rows_read": {"sidra_context_fixture": 1},
            "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows)},
            "parquet_bytes_written": 0,
        },
    }
    source_hash = sha256_file(input_path)
    sidra_context = {
        "schema_version": "1.0",
        "stitching": stitch.as_manifest(),
        "projection": projection,
        "high_dimensional_bound": high_dim.as_manifest(),
        "stdfm_input": stdfm_input.as_manifest(),
        "stdfm_output": stdfm_output.as_manifest(),
        "certification": certification,
        "table_paths": {
            "stitching": "Tables/sidra_stitching_segments.parquet",
            "projection": "Tables/sidra_projection_matrix.parquet",
            "high_dimensional": "Tables/sidra_high_dimensional_bounds.parquet",
            "stdfm_certification": "Tables/stdfm_certification.parquet",
            "stdfm_objective_contract": "Tables/stdfm_objective_contract.parquet",
            "stdfm_latent_factors": "Tables/stdfm_latent_factors.parquet",
            "stdfm_loadings": "Tables/stdfm_loadings.parquet",
            "stdfm_reconstructed_fields": "Tables/stdfm_reconstructed_fields.parquet",
            "stdfm_uncertainty": "Tables/stdfm_uncertainty.parquet",
        },
    }
    run_config = {
        "schema_version": "1.0",
        "workflow": "slice7a_sidra_stdfm_context_fixture",
        "source_systems": ["SIDRA"],
        "source_hashes": {"sidra_context_fixture": source_hash},
        "registry_hashes": {"sidra_stdfm": "slice7a_contract_v1"},
        "sidra_context": sidra_context,
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "generated_at": _now(),
        "workflow": "slice7a_sidra_stdfm_context_fixture",
        "source_hashes": run_config["source_hashes"],
        "registry_hashes": run_config["registry_hashes"],
        "telemetry": telemetry,
        "sidra_context": sidra_context,
        "environment": {"python": sys.version.split()[0]},
    }
    (run_dir / "UserIntent.json").write_text(
        json.dumps({"workflow": "slice7a_sidra_stdfm_context_fixture", "source_systems": ["SIDRA"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "RunConfig.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "ReproducibilityManifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "P_vector.json").write_text(json.dumps({"schema_version": "1.0", "sidra_context": sidra_context}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir
