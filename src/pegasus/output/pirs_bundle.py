from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.pirs.crossfit import assert_standard_deep_not_in_sample, fold_scheme_for_budget
from pegasus.pirs.design import build_design_matrix
from pegasus.pirs.diagnostics import build_pirs_diagnostics
from pegasus.pirs.families import exposure_offset_source, family_for_outcome
from pegasus.pirs.field_selection import select_fields_for_pirs
from pegasus.pirs.models import fit_parametric_model
from pegasus.pirs.residuals import assert_model_derived_provenance, residual_field_from_model
from pegasus.pirs.schemas import FieldCandidate, ModelInput


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pq.read_schema(path)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _field(*, field_id: str, name: str, kind: str, carrier: str, unit: str, aggregation: str, role: list[str], source: list[str], support: dict[str, Any], axes: dict[str, Any], operator: str | None, provenance: list[str], state: str, dashboard_safe: str, warnings: list[str], materialization_state: str = "materialized", path: str | None = None) -> dict[str, Any]:
    return {
        "field_id": field_id, "name": name, "kind": kind, "carrier": carrier, "unit": unit,
        "aggregation": aggregation, "role": _json(role), "source": _json(source),
        "support_json": _json(support), "axes_json": _json(axes), "operator": operator,
        "provenance": _json(provenance), "state": state, "dashboard_safe": dashboard_safe,
        "warnings": _json(warnings),
        "lineage_hash": content_hash({"field_id": field_id, "support": support, "axes": axes, "operator": operator}),
        "registry_hash": content_hash({"pirs": "slice8a"}),
        "materialization_state": materialization_state, "path": path,
    }


def _q(field: dict[str, Any], *, n_events: float | None, n_denom: float | None, n_eff: float, state: str, dashboard_safe: str, warnings: list[str], provenance_risk: float = 0.35) -> dict[str, Any]:
    return {
        "field_id": field["field_id"], "n_events": n_events, "n_denom": n_denom, "n_eff": n_eff,
        "cov_S": 1.0, "cov_T": 1.0, "missingness": 0.0, "zero_inflation": 0.0,
        "denom_fragility": 0.1, "cv": None, "moran_i": None, "temporal_roughness": None,
        "spatial_entropy": None, "provenance_risk": provenance_risk, "race_axis_source": None,
        "race_axis_target": None, "missing_race_share": None, "emission_prior_strength": None,
        "race_bridge_cv": None, "sensitivity_width": None, "bridge_mode": None, "state": state,
        "dashboard_safe": dashboard_safe, "warnings": _json(warnings), "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def _vd(field: dict[str, Any], *, definition: str, estimand: str, warning: str) -> dict[str, Any]:
    return {
        "field_id": field["field_id"], "display_name": field["name"], "technical_name": field["field_id"],
        "definition": definition, "estimand_label": estimand, "source_systems": field["source"],
        "carrier": field["carrier"], "unit": field["unit"], "support_description": field["support_json"],
        "axis_description": field["axes_json"], "provenance_description": field["provenance"],
        "state": field["state"], "dashboard_safe": field["dashboard_safe"], "interpretation_warning": warning,
    }


def _edge(edge_id: str, parent: str, child: str, operator: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"edge_id": edge_id, "parent_field_id": parent, "child_field_id": child, "operator": operator, "operator_params_json": _json(params), "registry_versions_json": _json({"pirs": "slice8a"}), "created_at": _now()}


def _warning(warning_id: str, field_id: str, code: str, message: str, severity: str = "warning") -> dict[str, Any]:
    return {"warning_id": warning_id, "field_id": field_id, "source": "pegasus.pirs", "severity": severity, "code": code, "message": message, "inherited_from": _json([]), "created_at": _now()}


def candidates_from_fixture(path: str | Path) -> list[FieldCandidate]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FieldCandidate(**item) for item in payload["candidates"]]


def pirs_plan_from_fixture(*, input_path: str | Path, budget: str = "standard") -> dict[str, Any]:
    candidates = candidates_from_fixture(input_path)
    selection = select_fields_for_pirs(candidates, budget=budget)
    fold = fold_scheme_for_budget(budget=budget)
    assert_standard_deep_not_in_sample(budget, fold.residual_mode)
    family = family_for_outcome(outcome=selection.selected_outcome, offset=selection.selected_offset) if selection.selected_outcome else None
    offset_source = exposure_offset_source(family=family, offset=selection.selected_offset) if family else None
    return {"status": "planned", "budget": budget, "residual_mode": fold.residual_mode, "fold_scheme": fold.fold_scheme, "outcome_field_id": selection.selected_outcome.field_id if selection.selected_outcome else None, "covariate_field_ids": [c.field_id for c in selection.selected_covariates], "offset_field_id": selection.selected_offset.field_id if selection.selected_offset else None, "family": family, "exposure_offset_source": offset_source, "rejected": list(selection.rejected)}


def write_pirs_fixture_bundle(*, input_path: str | Path, run_dir: str | Path, budget: str = "standard") -> Path:
    input_path = Path(input_path); run_dir = Path(run_dir)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if "municipalities" not in payload or "years" not in payload:
        raise ValueError("PIRS fixture payload must declare municipalities and years explicitly.")
    create_empty_output_bundle(run_dir)
    for name in ["ModelAssociations.parquet", "ResidualAssociations.parquet", "Hypotheses.parquet", "QuarantinedFields.parquet", "ForcedFields.parquet"]:
        _write_rows_like(run_dir / name, [])
    candidates = candidates_from_fixture(input_path)
    selection = select_fields_for_pirs(candidates, budget=budget)
    if selection.selected_outcome is None:
        raise ValueError("Slice 8A fixture did not select a legal outcome")
    family = family_for_outcome(outcome=selection.selected_outcome, offset=selection.selected_offset)
    offset_source = exposure_offset_source(family=family, offset=selection.selected_offset)
    fold = fold_scheme_for_budget(budget=budget)
    assert_standard_deep_not_in_sample(budget, fold.residual_mode)
    tables_dir = run_dir / "Tables"; pirs_table_dir = tables_dir / "pirs"
    design = build_design_matrix(observations_path=input_path, selection=selection, output_dir=pirs_table_dir)
    model_input = ModelInput(selection.selected_outcome.field_id, design.covariate_field_ids, design.offset_field_id, family, design.support_index_path, design.design_matrix_path, design.outcome_vector_path, {"exclude": ["illegal_excluded", "blocked"], "zero_variance": "exclude"}, {"pirs": "slice8a"}, fold.residual_mode)
    model_output = fit_parametric_model(model_input, output_dir=pirs_table_dir)
    residual = residual_field_from_model(model_output); assert_model_derived_provenance(residual)
    diagnostics = build_pirs_diagnostics(selection=selection, fold_scheme=fold, exposure_offset_source=offset_source)
    support = {"municipality_cod6": payload["municipalities"], "years": payload["years"], "n_rows": design.rows}
    axes = {"geo": "DATASUS_COD6", "time": "year", "support_kind": "annual_municipal_panel"}
    outcome_field = _field(field_id=selection.selected_outcome.field_id, name="PIRS fixture all deaths outcome", kind="extensive_measure", carrier=selection.selected_outcome.carrier, unit=selection.selected_outcome.unit, aggregation="sum", role=["model_outcome"], source=["fixture_efg"], support=support, axes=axes, operator="fixture_field_import", provenance=list(selection.selected_outcome.provenance or ("fixture",)), state="verified", dashboard_safe="True", warnings=list(selection.selected_outcome.warnings))
    offset_field = _field(field_id=selection.selected_offset.field_id, name="PIRS fixture population exposure offset", kind="extensive_measure", carrier=selection.selected_offset.carrier, unit=selection.selected_offset.unit, aggregation="sum", role=["model_offset"], source=["SIDRA"], support=support, axes=axes, operator="fixture_field_import", provenance=list(selection.selected_offset.provenance or ("official",)), state="verified", dashboard_safe="True", warnings=list(selection.selected_offset.warnings)) if selection.selected_offset else None
    cov_fields = [_field(field_id=c.field_id, name=f"PIRS fixture covariate {c.field_id}", kind="context_gradient", carrier=c.carrier, unit=c.unit, aggregation="mean", role=["model_covariate"], source=["CNES-ST" if "capacity" in c.field_id else "SIDRA"], support=support, axes=axes, operator="fixture_field_import", provenance=list(c.provenance or ("processed",)), state="verified", dashboard_safe="True", warnings=list(c.warnings)) for c in selection.selected_covariates]
    residual_field = _field(field_id=residual.field_id, name="PIRS Pearson residual field for all deaths model", kind="model_residual", carrier="residual", unit="dimensionless", aggregation="non_aggregable", role=["model_residual", "hsic_candidate_residual"], source=["PIRS"], support={**support, "model_id": model_output.model_id}, axes={**axes, "residual_type": residual.residual_type}, operator="pirs_residual_extraction", provenance=list(residual.provenance), state=residual.state, dashboard_safe=residual.dashboard_safe, warnings=list(residual.warnings), path=model_output.diagnostics["residual_values_path"])
    fields = [outcome_field] + ([offset_field] if offset_field else []) + cov_fields + [residual_field]
    _write_rows_like(run_dir / "V_fields.parquet", fields)
    q_rows = [_q(outcome_field, n_events=sum(obs[selection.selected_outcome.field_id] for obs in payload["observations"]), n_denom=None, n_eff=float(design.rows), state="verified", dashboard_safe="True", warnings=[])]
    if offset_field:
        q_rows.append(_q(offset_field, n_events=None, n_denom=sum(obs[selection.selected_offset.field_id] for obs in payload["observations"]), n_eff=float(design.rows), state="verified", dashboard_safe="True", warnings=[]))
    q_rows.extend(_q(c, n_events=None, n_denom=None, n_eff=float(design.rows), state="verified", dashboard_safe="True", warnings=[]) for c in cov_fields)
    q_rows.append(_q(residual_field, n_events=None, n_denom=None, n_eff=float(design.rows), state="warning", dashboard_safe="False", warnings=list(residual.warnings), provenance_risk=0.55))
    _write_rows_like(run_dir / "Q_tensor.parquet", q_rows)
    _write_rows_like(run_dir / "VariableDictionary.parquet", [_vd(outcome_field, definition="Legal PIRS outcome fixture field.", estimand="count_outcome_for_parametric_model", warning="Fixture-only model outcome."), *([_vd(offset_field, definition="Population exposure offset used by the count model.", estimand="exposure_offset", warning="Offset comes from carrier registry semantics.")] if offset_field else []), *[_vd(c, definition="Legal covariate fixture field admitted to the design matrix.", estimand="model_covariate", warning="Fixture-only covariate.") for c in cov_fields], _vd(residual_field, definition="Model-derived Pearson residual emitted by PIRS after parametric fitting.", estimand="model_residual", warning="Not a raw epidemiological variable; eligible for residual scanner only under later HSIC slice.")])
    _write_rows_like(run_dir / "E_DAG.parquet", [_edge(f"edge_{parent['field_id']}_to_{residual_field['field_id']}", parent["field_id"], residual_field["field_id"], "pirs_model_fit_and_residual_extract", {"model_id": model_output.model_id, "family": family}) for parent in [outcome_field] + ([offset_field] if offset_field else []) + cov_fields])
    warnings = [_warning("pirs_fixture_model_not_for_inference", residual.field_id, "pirs_fixture_model_not_for_inference", "Slice 8A deterministic fixture model validates contracts but is not inferential output."), _warning("model_derived_residual_not_raw", residual.field_id, "model_derived_residual_not_raw_epidemiological_variable", "Residual fields are model-derived and must not be interpreted as raw epidemiological measures.")]
    if diagnostics.zero_variance_rejections:
        warnings.append(_warning("zero_variance_fields_excluded", "run", "zero_variance_fields_excluded_from_design_matrix", "Zero-variance candidates were excluded before model design materialization."))
    _write_rows_like(run_dir / "Warnings.parquet", warnings)
    _write_rows_like(run_dir / "ModelAssociations.parquet", [{"id": model_output.model_id, "status": model_output.status, "warnings": _json(list(model_output.warnings))}])
    _write_rows_like(run_dir / "ResidualAssociations.parquet", [{"id": residual.field_id, "status": "materialized", "warnings": _json(list(residual.warnings))}])
    failed = [{"failed_branch_id": f"pirs_rejected_{r['field_id']}", "attempted_operator": "pirs_design_matrix_admission", "parent_field_ids": _json([outcome_field["field_id"]]), "failure_stage": "pirs_field_selection", "failed_terms": _json([r["field_id"]]), "reason": r["reason"], "warnings": _json([r["reason"]]), "created_at": _now()} for r in selection.rejected]
    _write_rows_like(run_dir / "FailedBranches.parquet", failed)
    _write_table(tables_dir / "pirs_model_associations_detail.parquet", [{"model_id": model_output.model_id, "outcome_field_id": model_input.outcome_field_id, "covariate_field_ids": _json(list(model_input.covariate_field_ids)), "offset_field_id": model_input.offset_field_id, "family": model_output.family, "status": model_output.status, "coefficients_path": model_output.coefficients_path, "fitted_values_path": model_output.fitted_values_path, "residual_field_id": model_output.residual_field_id, "diagnostics_json": _json(model_output.diagnostics), "warnings": _json(list(model_output.warnings))}])
    _write_table(tables_dir / "pirs_residual_associations_detail.parquet", [{"residual_field_id": residual.field_id, "parent_model_id": residual.parent_model_id, "residual_type": residual.residual_type, "provenance": _json(list(residual.provenance)), "state": residual.state, "dashboard_safe": residual.dashboard_safe, "path": model_output.diagnostics["residual_values_path"], "warnings": _json(list(residual.warnings))}])
    _write_table(tables_dir / "pirs_design_matrix_contract.parquet", [{"design_matrix_path": design.design_matrix_path, "outcome_vector_path": design.outcome_vector_path, "support_index_path": design.support_index_path, "rows": design.rows, "covariate_field_ids": _json(list(design.covariate_field_ids)), "offset_field_id": design.offset_field_id, "status": design.status}])
    selected_ids = {x.field_id for x in [selection.selected_outcome, selection.selected_offset] if x} | {x.field_id for x in selection.selected_covariates}
    _write_table(tables_dir / "pirs_field_selection.parquet", [{"field_id": c.field_id, "role": c.role, "utility": c.utility, "q_state": c.q_state, "variance": c.variance, "selected": c.field_id in selected_ids, "rejection_reason": next((r["reason"] for r in selection.rejected if r["field_id"] == c.field_id), None)} for c in candidates])
    _write_table(tables_dir / "pirs_diagnostics.parquet", [{**diagnostics.as_manifest(), "warnings": _json(list(diagnostics.warnings))}])
    run_config = {"frozen": True, "slice": "8A", "budget": budget, "pirs": {"schema_version": "1.0", "field_selection": {"budget": budget, "top_k": selection.top_k}, "model_id": model_output.model_id, "family": family, "residual_mode": fold.residual_mode, "fold_scheme": fold.fold_scheme, "exposure_offset_source": offset_source, "hsic_enabled": False}}
    (run_dir / "RunConfig.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "UserIntent.json").write_text(json.dumps({"frozen": True, "intent_source": "slice8a_fixture", "budget": budget}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "P_vector.json").write_text(json.dumps({"schema_version": "1.0", "provenance": {row["field_id"]: json.loads(row["provenance"]) for row in fields}, "pirs": run_config["pirs"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stages = ["config_load", "registry_validation", "datasus_acquire", "datasus_profile", "datasus_normalize", "sidra_metadata", "sidra_plan", "sidra_fetch", "sidra_normalize", "geo_support", "she_build", "population_solver", "stdfm", "efg_build", "q_tensor", "pirs_model", "pirs_hsic", "output_serialization", "output_validation"]
    manifest = {"run_id": run_dir.name, "created_at": _now(), "completed_at": _now(), "status": "success", "code_version": {"package_version": "0.1.0", "git_commit": "uninitialized", "git_dirty": False}, "environment": {"python_version": sys.version, "os": platform.platform(), "duckdb_version": None, "polars_version": None, "pyarrow_version": pa.__version__, "torch_version": None, "torch_cuda_available": False, "cuda_device_name": None, "r_version": None, "microdatasus_version": None, "read_dbc_version": None}, "registry_hashes": {"pirs": content_hash({"slice": "8A", "family": family})}, "source_manifest_hashes": [sha256_file(input_path)], "source_hashes": {"pirs_fixture": sha256_file(input_path)}, "random_seeds": {"pirs_fixture_seed": payload.get("seed", 20260611)}, "pirs": run_config["pirs"], "telemetry": {"total_wall_seconds": 0.0, "stage_wall_seconds": {f"{s}_seconds": 0.0 for s in stages}, "stage_status": {s: ("success" if s in {"config_load", "registry_validation", "efg_build", "q_tensor", "pirs_model", "output_serialization", "output_validation"} else ("blocked" if s == "pirs_hsic" else "skipped")) for s in stages}, "resource_summary": {"peak_rss_mb": None, "peak_vram_mb": None, "duckdb_temp_bytes": None, "rows_read": {"pirs_fixture": len(payload["observations"])}, "rows_written": {"V_fields": len(fields), "Q_tensor": len(fields), "ModelAssociations": 1, "ResidualAssociations": 1}, "parquet_bytes_written": 0}}}
    (run_dir / "ReproducibilityManifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = validate_output_bundle(run_dir=str(run_dir))
    if not validation.ok:
        raise RuntimeError("Slice 8A PIRS bundle invalid: " + "; ".join(validation.errors))
    return run_dir
