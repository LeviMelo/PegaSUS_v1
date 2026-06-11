"""Standalone Slice 9A HSIC residual-scanner run bundle writer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.core.hashing import sha256_file, content_hash
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.pirs.fdr import correct_p_values
from pegasus.pirs.hsic import run_hsic_scan, select_hsic_mode, residual_mode_for_hsic
from pegasus.pirs.nulls import select_null_regime, assert_monthly_null_preserves_season, descriptive_only_when_insufficient_blocks
from pegasus.pirs.nystrom import nystrom_diagnostics
from pegasus.pirs.rff import rff_diagnostics


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_schema(path: Path) -> pa.Schema:
    return pq.read_schema(path)


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = _read_schema(path)
    shaped = [{name: row.get(name) for name in schema.names} for row in rows]
    table = pa.Table.from_pylist(shaped, schema=schema)
    pq.write_table(table, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _support(payload: dict[str, Any]) -> dict[str, Any]:
    years = sorted({int(row["year"]) for row in payload["observations"]})
    municipalities = sorted({str(row["municipality_cod6"]) for row in payload["observations"]})
    n = len(payload["observations"])
    return {
        "geo_level": "municipality",
        "geo_code_type": "DATASUS_COD6",
        "municipalities": municipalities,
        "years": years,
        "n_events": n,
        "n_denom": n,
        "n_eff": float(n),
        "temporal_resolution": payload.get("panel_type", "annual_municipal_panel"),
    }


def _field(
    field_id: str,
    *,
    name: str,
    kind: str,
    carrier: str,
    unit: str,
    aggregation: str,
    role: list[str],
    source: list[str],
    support: dict[str, Any],
    axes: dict[str, Any],
    operator: str,
    provenance: list[str],
    warnings: list[str],
    state: str,
    dashboard_safe: str,
    path: str | None = None,
) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "name": name,
        "kind": kind,
        "carrier": carrier,
        "unit": unit,
        "aggregation": aggregation,
        "role": _json(role),
        "source": _json(source),
        "support_json": _json(support),
        "axes_json": _json(axes),
        "operator": operator,
        "provenance": _json(provenance),
        "state": state,
        "dashboard_safe": dashboard_safe,
        "warnings": _json(warnings),
        "lineage_hash": content_hash({"field_id": field_id, "support": support, "axes": axes, "operator": operator}),
        "registry_hash": content_hash({"slice": "9A", "field_id": field_id}),
        "materialization_state": "materialized" if path else "virtual",
        "path": path,
    }


def _q(field: dict[str, Any], *, n_eff: float, warnings: list[str], state: str, dashboard_safe: str) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "n_events": n_eff,
        "n_denom": n_eff,
        "n_eff": n_eff,
        "cov_S": 0.0,
        "cov_T": 0.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0,
        "provenance_risk": 0.2 if state != "verified" else 0.0,
        "state": state,
        "dashboard_safe": dashboard_safe,
        "warnings": _json(warnings),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def _vd(field: dict[str, Any], *, definition: str, estimand: str, warning: str) -> dict[str, Any]:
    support = json.loads(field.get("support_json") or "{}")
    axes = json.loads(field.get("axes_json") or "{}")
    provenance = json.loads(field.get("provenance") or "[]")
    return {
        "field_id": field["field_id"],
        "display_name": field["name"],
        "technical_name": field["field_id"],
        "definition": definition,
        "estimand_label": estimand,
        "source_systems": field["source"],
        "carrier": field["carrier"],
        "unit": field["unit"],
        "support_description": _json(support),
        "axis_description": _json(axes),
        "provenance_description": _json(provenance),
        "state": field["state"],
        "dashboard_safe": field["dashboard_safe"],
        "interpretation_warning": warning,
        "updated_at": _now(),
    }


def _warning(warning_id: str, field_id: str, code: str, message: str, severity: str = "warning") -> dict[str, Any]:
    return {
        "warning_id": warning_id,
        "field_id": field_id,
        "source": "PIRS-HSIC",
        "severity": severity,
        "code": code,
        "message": message,
        "created_at": _now(),
        "inherited_from": None,
    }


def _failed(branch_id: str, reason: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "failed_branch_id": branch_id,
        "attempted_operator": "pirs_hsic_scan",
        "failure_stage": "residual_nonlinear_scan",
        "failed_terms": _json(["cuda_backend"]),
        "parent_field_ids": _json(["hsic_outcome_residual_deviance", "hsic_covariate_context"]),
        "reason": reason,
        "warnings": _json(warnings),
        "created_at": _now(),
    }


def _load_fixture(input_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(input_path).read_text(encoding="utf-8"))


def _vectors(payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    residuals = [float(row["residual"]) for row in payload["observations"]]
    covariate = [float(row["covariate"]) for row in payload["observations"]]
    return residuals, covariate


def plan_hsic_fixture(*, input_path: str | Path, budget: str = "standard", cuda_required: bool = False) -> dict[str, Any]:
    payload = _load_fixture(input_path)
    support = _support(payload)
    panel_type = str(payload.get("panel_type", "annual_municipal_panel"))
    null = select_null_regime(panel_type)
    assert_monthly_null_preserves_season(null)
    descriptive_only = descriptive_only_when_insufficient_blocks(
        spatial_blocks=int(payload.get("spatial_blocks", len(support["municipalities"]))),
        temporal_blocks=int(payload.get("temporal_blocks", len(support["years"]))),
    )
    n_eff = float(payload.get("n_eff_override", support["n_eff"]))
    mode = select_hsic_mode(n_eff=n_eff, budget=budget, cuda_required=cuda_required, cuda_available=False)
    residual_mode = residual_mode_for_hsic(budget=budget)
    return {
        "hsic_mode": mode,
        "null_strategy": "descriptive_association_only" if descriptive_only else null.null_strategy,
        "fdr_method": null.fdr_method,
        "permutations": null.permutations,
        "residual_mode": residual_mode,
        "n_eff": n_eff,
        "descriptive_only": descriptive_only,
        "cuda_required": cuda_required,
        "panel_type": panel_type,
    }


def write_hsic_fixture_bundle(*, input_path: str | Path, run_dir: str | Path, budget: str = "standard", cuda_required: bool = False) -> Path:
    input_path = Path(input_path)
    run = Path(run_dir)
    create_empty_output_bundle(run)
    payload = _load_fixture(input_path)
    support = _support(payload)
    plan = plan_hsic_fixture(input_path=input_path, budget=budget, cuda_required=cuda_required)
    residuals, covariate = _vectors(payload)
    null = select_null_regime(str(payload.get("panel_type", "annual_municipal_panel")))
    output = run_hsic_scan(
        outcome_residual_field_id="hsic_outcome_residual_deviance",
        covariate_field_id="hsic_covariate_context",
        residuals=residuals,
        covariate=covariate,
        support_intersection=support,
        budget=budget,
        null_strategy=plan["null_strategy"],
        fdr_method=plan["fdr_method"],
        permutations=plan["permutations"],
        cuda_required=cuda_required,
        cuda_available=False,
    )
    fdr = correct_p_values([output.p_value], method=plan["fdr_method"])
    q_value = fdr.q_values[0]
    output = type(output)(**{**output.__dict__, "q_value": q_value})
    approx = output.approximation_diagnostics.copy()
    if output.hsic_mode == "nystrom":
        approx.update(nystrom_diagnostics(n_eff=int(output.n_eff), budget=budget).as_manifest())
    if output.hsic_mode == "rff":
        approx.update(rff_diagnostics(n_eff=int(output.n_eff), budget=budget).as_manifest())

    base_axes = {"time_axis": "year", "geo_axis": "municipality_cod6", "support_role": "hsic_fixture"}
    residual_field = _field(
        "hsic_outcome_residual_deviance",
        name="Cross-fitted deviance residual outcome for HSIC",
        kind="model_residual",
        carrier="municipality_year_panel",
        unit="residual",
        aggregation="not_aggregable",
        role=["outcome_residual"],
        source=["PIRS"],
        support=support,
        axes={**base_axes, "residual_type": "deviance", "residual_mode": output.residual_mode},
        operator="pirs_residual_extraction",
        provenance=["model_derived"],
        warnings=["hsic_consumes_cross_fitted_residuals"] if output.residual_mode.startswith("cross_fitted") else [],
        state="verified",
        dashboard_safe="False",
    )
    covariate_field = _field(
        "hsic_covariate_context",
        name="Context covariate for residual HSIC scan",
        kind="latent_context",
        carrier="municipality_year_panel",
        unit="index",
        aggregation="mean",
        role=["covariate"],
        source=["fixture"],
        support=support,
        axes={**base_axes, "covariate_role": "context_exposure"},
        operator="fixture_context_field",
        provenance=["fixture"],
        warnings=[],
        state="verified",
        dashboard_safe="False",
    )
    hsic_state = "fragile" if output.hsic_mode in {"nystrom", "rff", "disabled", "cuda_unavailable_abort"} else "verified"
    hsic_field = _field(
        "hsic_residual_association_score",
        name="HSIC residual association score",
        kind="marked_functional",
        carrier="support_intersection",
        unit="hsic_statistic",
        aggregation="not_aggregable",
        role=["nonlinear_residual_association"],
        source=["PIRS-HSIC"],
        support={**support, "field_metadata": output.as_manifest(), "approximation_diagnostics": approx, "fdr": fdr.as_manifest()},
        axes={**base_axes, "hsic_mode": output.hsic_mode, "null_strategy": output.null_strategy, "fdr_method": output.fdr_method},
        operator="hsic_residual_scan",
        provenance=["model_derived", "residual_scan"],
        warnings=output.warnings,
        state=hsic_state,
        dashboard_safe="False",
    )
    fields = [residual_field, covariate_field, hsic_field]
    q_rows = [
        _q(residual_field, n_eff=output.n_eff, warnings=json.loads(residual_field["warnings"]), state="verified", dashboard_safe="False"),
        _q(covariate_field, n_eff=output.n_eff, warnings=[], state="verified", dashboard_safe="False"),
        _q(hsic_field, n_eff=output.n_eff, warnings=output.warnings, state=hsic_field["state"], dashboard_safe="False"),
    ]
    vd_rows = [
        _vd(residual_field, definition="Residual field consumed by HSIC; not a raw epidemiological outcome.", estimand="model residual", warning="model-derived residual; not directly interpretable as an event count"),
        _vd(covariate_field, definition="Context covariate aligned to residual support.", estimand="context covariate", warning="fixture covariate used for Slice 9A scanner validation"),
        _vd(hsic_field, definition="HSIC nonlinear dependence statistic between a residual outcome and a covariate.", estimand="residual nonlinear association", warning="hypothesis-generating score; not causal identification"),
    ]
    warning_rows = [_warning(f"slice9a_warning_{i+1}", hsic_field["field_id"], code, code.replace("_", " ")) for i, code in enumerate(output.warnings)]
    failed_rows = []
    if output.hsic_mode == "cuda_unavailable_abort":
        failed_rows.append(_failed("slice9a_cuda_required_unavailable", "CUDA-required HSIC requested but CUDA unavailable", output.warnings))
    if plan["descriptive_only"]:
        warning_rows.append(_warning("slice9a_insufficient_blocks", hsic_field["field_id"], "insufficient_hsic_blocks", "fewer than five spatial or temporal blocks; descriptive association only"))

    _write_rows_like(run / "V_fields.parquet", fields)
    _write_rows_like(run / "Q_tensor.parquet", q_rows)
    _write_rows_like(run / "VariableDictionary.parquet", vd_rows)
    _write_rows_like(run / "Warnings.parquet", warning_rows)
    _write_rows_like(run / "FailedBranches.parquet", failed_rows)
    _write_rows_like(run / "E_DAG.parquet", [
        {
            "edge_id": "edge_hsic_residual_to_score",
            "parent_field_id": residual_field["field_id"],
            "child_field_id": hsic_field["field_id"],
            "operator": "hsic_residual_scan",
            "operator_params_json": _json(output.as_manifest()),
            "created_at": _now(),
            "registry_versions_json": _json({"hsic_registry": "slice9a"}),
        },
        {
            "edge_id": "edge_hsic_covariate_to_score",
            "parent_field_id": covariate_field["field_id"],
            "child_field_id": hsic_field["field_id"],
            "operator": "hsic_residual_scan",
            "operator_params_json": _json(output.as_manifest()),
            "created_at": _now(),
            "registry_versions_json": _json({"hsic_registry": "slice9a"}),
        },
    ])
    _write_rows_like(run / "Hypotheses.parquet", [{
        "hypothesis_id": output.hypothesis_id,
        "outcome_field_id": output.outcome_field_id,
        "exposure_field_id": output.covariate_field_id,
        "covariate_field_id": output.covariate_field_id,
        "residual_field_id": output.residual_field_id,
        "method": "HSIC",
        "statistic": output.statistic,
        "p_value": output.p_value,
        "q_value": output.q_value,
        "state": "fragile" if output.warnings else "verified",
        "warnings": _json(output.warnings),
        "metadata_json": _json(output.as_manifest()),
        "created_at": _now(),
    }])
    _write_rows_like(run / "ModelAssociations.parquet", [])
    _write_rows_like(run / "ResidualAssociations.parquet", [{
        "association_id": "residual_assoc_slice9a_hsic",
        "id": "residual_assoc_slice9a_hsic",
        "model_id": "pirs_model_fixture",
        "residual_field_id": residual_field["field_id"],
        "parent_field_id": residual_field["field_id"],
        "status": "materialized",
        "residual_type": "deviance",
        "provenance": _json(["model_derived"]),
        "diagnostics_json": _json({"residual_mode": output.residual_mode}),
        "warnings": _json([]),
        "created_at": _now(),
    }])
    _write_rows_like(run / "QuarantinedFields.parquet", [])
    _write_rows_like(run / "ForcedFields.parquet", [])

    tables_dir = run / "Tables"
    tables_dir.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist([output.as_manifest()]), tables_dir / "hsic_outputs.parquet")
    pq.write_table(pa.Table.from_pylist([approx]), tables_dir / "hsic_approximation_diagnostics.parquet")
    pq.write_table(pa.Table.from_pylist([null.as_manifest()]), tables_dir / "hsic_null_regime.parquet")
    pq.write_table(pa.Table.from_pylist([fdr.as_manifest()]), tables_dir / "hsic_fdr_correction.parquet")

    source_hashes = {"hsic_fixture": sha256_file(input_path)}
    registry_hashes = {"pirs_hsic": content_hash({"slice": "9A", "metadata": output.as_manifest(), "null": null.as_manifest()})}
    metadata = {
        "schema_version": "1.0",
        "slice": "9A",
        "workflow_mode": "standalone_hsic_fixture",
        "budget": budget,
        "hsic_mode": output.hsic_mode,
        "residual_mode": output.residual_mode,
        "null_strategy": output.null_strategy,
        "fdr_method": output.fdr_method,
        "n_eff": output.n_eff,
        "cuda_required": cuda_required,
        "cuda_available": False,
        "approximation_diagnostics_emitted": True,
        "standard_deep_cross_fitted_residuals": budget not in {"standard", "deep"} or output.residual_mode.startswith("cross_fitted"),
        "source_hashes": source_hashes,
        "table_hashes": {
            "hsic_outputs": sha256_file(tables_dir / "hsic_outputs.parquet"),
            "hsic_approximation_diagnostics": sha256_file(tables_dir / "hsic_approximation_diagnostics.parquet"),
            "hsic_null_regime": sha256_file(tables_dir / "hsic_null_regime.parquet"),
            "hsic_fdr_correction": sha256_file(tables_dir / "hsic_fdr_correction.parquet"),
        },
    }
    run_config = {
        "run_id": run.name,
        "created_at": _now(),
        "workflow_mode": "standalone_hsic_fixture",
        "user_intent_frozen": True,
        "run_config_frozen": True,
        "source_hashes": source_hashes,
        "registry_hashes": registry_hashes,
        "pirs_hsic": metadata,
    }
    manifest = {
        "run_id": run.name,
        "created_at": _now(),
        "workflow_mode": "standalone_hsic_fixture",
        "registry_versions": {"hsic_registry": "slice9a", "null_registry": "slice9a", "fdr_registry": "slice9a"},
        "source_hashes": source_hashes,
        "registry_hashes": registry_hashes,
        "source_manifest_hashes": [source_hashes["hsic_fixture"]],
        "pirs_hsic": metadata,
        "telemetry": {
            "total_wall_seconds": 0.0,
            "stage_status": {
                "pirs_model": "skipped",
                "pirs_hsic": "success" if output.hsic_mode != "cuda_unavailable_abort" else "blocked",
                "output_serialization": "success",
                "output_validation": "success",
            },
            "stage_wall_seconds": {
                "pirs_model": 0.0,
                "pirs_hsic": 0.0,
                "output_serialization": 0.0,
                "output_validation": 0.0,
            },
            "stage_errors": {"pirs_hsic": output.warnings if output.hsic_mode == "cuda_unavailable_abort" else []},
            "resource_summary": {"engine": "polars_arrow_fixture", "cuda_available": False},
        },
    }
    p_vector = {
        "schema_version": "1.0",
        "pirs_hsic": metadata,
        "statistic": output.statistic,
        "p_value": output.p_value,
        "q_value": output.q_value,
    }
    user_intent = {"budget": budget, "requested_modules": ["pirs_hsic"], "cuda_required": cuda_required, "frozen": True}
    _write_json(run / "RunConfig.json", run_config)
    _write_json(run / "ReproducibilityManifest.json", manifest)
    _write_json(run / "P_vector.json", p_vector)
    _write_json(run / "UserIntent.json", user_intent)
    return run
