from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES, TERMINAL_STAGE_STATUSES
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES, OutputSchemaRegistry, OutputValidationResult

REQUIRED_V_FIELDS_COLUMNS = {"field_id", "name", "kind", "carrier", "unit", "aggregation", "role", "source", "support_json", "axes_json", "operator", "provenance", "state", "dashboard_safe", "warnings", "lineage_hash", "registry_hash", "materialization_state", "path"}
REQUIRED_E_DAG_COLUMNS = {"edge_id", "parent_field_id", "child_field_id", "operator", "operator_params_json", "registry_versions_json", "created_at"}
REQUIRED_Q_TENSOR_COLUMNS = {"field_id", "n_events", "n_denom", "n_eff", "cov_S", "cov_T", "missingness", "zero_inflation", "denom_fragility", "provenance_risk", "state", "dashboard_safe", "warnings", "computed_at", "q_schema_version"}
OPTIONAL_RACE_Q_COLUMNS = {"race_axis_source", "race_axis_target", "missing_race_share", "race_bridge_cv", "sensitivity_width", "bridge_mode"}
REQUIRED_VARIABLE_DICTIONARY_COLUMNS = {"field_id", "display_name", "technical_name", "definition", "estimand_label", "source_systems", "carrier", "unit", "support_description", "axis_description", "provenance_description", "state", "dashboard_safe", "interpretation_warning"}
FIELD_REFERENCE_COLUMNS = {"field_id", "parent_field_id", "child_field_id", "outcome_field_id", "covariate_field_id", "residual_field_id"}
RACE_BRIDGE_POSTERIOR_KEYS = {"numerator_axis_source", "denominator_axis_target", "bridge_operator", "emission_matrix_registry_version", "bridge_mode", "missing_race_share", "race_bridge_cv", "sensitivity_width", "race_axis_warning", "bayesian_ecological_bridge_warning", "prior_hash", "lower_count", "upper_count"}
RUN_CONFIG_RACE_BRIDGE_KEYS = {"bridge_id", "mode", "prior_hash", "source_axis", "target_axis", "missing_race_share", "sensitivity_width", "race_bridge_cv", "raw_admin_counts_preserved", "missing_category_preserved", "attach_stage"}


def _read(path: Path):
    return pq.read_table(path)


def _column_values(table, column: str) -> list[Any]:
    if column not in table.column_names:
        return []
    return table.column(column).to_pylist()


def _nonnull(values: list[Any]) -> set[Any]:
    return {value for value in values if value is not None and value != ""}


def _load_json_file(path: Path, *, errors: list[str], name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid {name}: {exc}")
        return None


def _load_json_cell(value: Any, *, errors: list[str], context: str) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception as exc:
        errors.append(f"invalid JSON cell at {context}: {exc}")
        return None


def _require_columns(*, table_name: str, actual: set[str], required: set[str], errors: list[str]) -> None:
    missing = sorted(required - actual)
    if missing:
        errors.append(f"{table_name} missing required columns: {missing}")


def _validate_first_class_keys(root: Path, schema_registry: OutputSchemaRegistry, errors: list[str]) -> None:
    expected_names = {OUTPUT_BUNDLE_FILES[key] for key in schema_registry.required_keys}
    found_names = {p.name for p in root.iterdir()}
    for name in sorted(expected_names - found_names):
        errors.append(f"missing first-class artifact: {name}")
    for name in sorted(found_names - expected_names):
        errors.append(f"extra first-class artifact: {name}")
    for key, name in OUTPUT_BUNDLE_FILES.items():
        if key not in schema_registry.required_keys:
            continue
        path = root / name
        if not path.exists():
            continue
        if key in {"Tables", "Maps"}:
            if not path.is_dir():
                errors.append(f"first-class artifact is not a directory: {name}")
        elif not path.is_file():
            errors.append(f"first-class artifact is not a file: {name}")


def _validate_manifest_and_config(*, root: Path, errors: list[str], warnings: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    user_intent = _load_json_file(root / "UserIntent.json", errors=errors, name="UserIntent.json")
    run_config = _load_json_file(root / "RunConfig.json", errors=errors, name="RunConfig.json")
    manifest = _load_json_file(root / "ReproducibilityManifest.json", errors=errors, name="ReproducibilityManifest.json")
    p_vector = _load_json_file(root / "P_vector.json", errors=errors, name="P_vector.json")
    if not isinstance(user_intent, dict):
        errors.append("UserIntent.json is not a frozen JSON object")
        user_intent = {}
    if not isinstance(run_config, dict):
        errors.append("RunConfig.json is not a frozen JSON object")
        run_config = {}
    if not isinstance(manifest, dict):
        errors.append("ReproducibilityManifest.json is not a JSON object")
        manifest = {}
    if not isinstance(p_vector, (dict, list)):
        errors.append("P_vector.json must be a JSON object or list")
    compile_mode = run_config.get("compile_mode") or manifest.get("compile_mode")
    is_compile_run = bool(compile_mode)
    source_hashes = manifest.get("source_hashes")
    registry_hashes = manifest.get("registry_hashes")
    if is_compile_run:
        if not isinstance(source_hashes, dict) or not source_hashes:
            errors.append("ReproducibilityManifest.json missing nonempty source_hashes for compile run")
        if not isinstance(registry_hashes, dict) or not registry_hashes:
            errors.append("ReproducibilityManifest.json missing nonempty registry_hashes for compile run")
        if not isinstance(run_config.get("source_hashes"), dict) or not run_config.get("source_hashes"):
            errors.append("RunConfig.json missing nonempty source_hashes for compile run")
        if not isinstance(run_config.get("registry_hashes"), dict) or not run_config.get("registry_hashes"):
            errors.append("RunConfig.json missing nonempty registry_hashes for compile run")
    else:
        if not isinstance(source_hashes, dict) or not source_hashes:
            warnings.append("ReproducibilityManifest.json has empty or missing source_hashes on non-compile run")
        if not isinstance(registry_hashes, dict) or not registry_hashes:
            warnings.append("ReproducibilityManifest.json has empty or missing registry_hashes on non-compile run")
    _validate_telemetry(manifest=manifest, is_compile_run=is_compile_run, errors=errors)
    return user_intent, run_config, manifest


def _validate_telemetry(*, manifest: dict[str, Any], is_compile_run: bool, errors: list[str]) -> None:
    telemetry = manifest.get("telemetry")
    if not isinstance(telemetry, dict):
        errors.append("ReproducibilityManifest.json missing global telemetry object")
        return
    total_wall_seconds = telemetry.get("total_wall_seconds")
    if not isinstance(total_wall_seconds, (int, float)) or total_wall_seconds < 0:
        errors.append("telemetry.total_wall_seconds missing or negative")
    stage_status = telemetry.get("stage_status")
    stage_wall_seconds = telemetry.get("stage_wall_seconds")
    if not isinstance(stage_status, dict):
        errors.append("telemetry.stage_status missing or not a mapping")
        stage_status = {}
    if not isinstance(stage_wall_seconds, dict):
        errors.append("telemetry.stage_wall_seconds missing or not a mapping")
        stage_wall_seconds = {}
    if is_compile_run:
        missing_status = sorted(set(COMPILE_TELEMETRY_STAGES) - set(stage_status))
        missing_duration = sorted(set(COMPILE_TELEMETRY_STAGES) - set(stage_wall_seconds))
        if missing_status:
            errors.append(f"telemetry.stage_status missing compile stages: {missing_status}")
        if missing_duration:
            errors.append(f"telemetry.stage_wall_seconds missing compile stages: {missing_duration}")
    for stage, status in stage_status.items():
        if status not in TERMINAL_STAGE_STATUSES:
            errors.append(f"invalid telemetry stage status: {stage}={status}")
    for stage, duration in stage_wall_seconds.items():
        if not isinstance(duration, (int, float)) or duration < 0:
            errors.append(f"invalid telemetry stage duration: {stage}={duration}")


def _validate_race_bridge_contract(*, root: Path, v, q, run_config: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
    rows = v.to_pylist()
    q_rows = {str(row.get("field_id")): row for row in q.to_pylist() if row.get("field_id") is not None}
    bridge_rows = [row for row in rows if str(row.get("field_id", "")).startswith("SIMRaceBridge") or str(row.get("field_id", "")).startswith("SIMRaceAdminRawCount_")]
    if not bridge_rows:
        return
    run_bridge = run_config.get("race_bridge")
    manifest_bridge = manifest.get("race_bridge")
    if not isinstance(run_bridge, dict):
        errors.append("RunConfig.json missing race_bridge metadata while race bridge fields exist")
        run_bridge = {}
    if not isinstance(manifest_bridge, dict):
        errors.append("ReproducibilityManifest.json missing race_bridge metadata while race bridge fields exist")
        manifest_bridge = {}
    missing_run_keys = sorted(RUN_CONFIG_RACE_BRIDGE_KEYS - set(run_bridge))
    if missing_run_keys:
        errors.append(f"RunConfig.race_bridge missing keys: {missing_run_keys}")
    if run_bridge.get("raw_admin_counts_preserved") is not True:
        errors.append("RunConfig.race_bridge.raw_admin_counts_preserved must be true")
    if run_bridge.get("missing_category_preserved") is not True:
        errors.append("RunConfig.race_bridge.missing_category_preserved must be true")
    if manifest_bridge.get("prior_hash") != run_bridge.get("prior_hash"):
        errors.append("ReproducibilityManifest.race_bridge.prior_hash must match RunConfig.race_bridge.prior_hash")
    if not (root / "Tables" / "race_bridge_summary.parquet").exists():
        errors.append("race bridge fields exist but Tables/race_bridge_summary.parquet is missing")
    for row in bridge_rows:
        fid = str(row.get("field_id"))
        q_row = q_rows.get(fid)
        if q_row is None:
            errors.append(f"race bridge field missing Q_tensor row: {fid}")
            continue
        if fid.startswith("SIMRaceBridgePosteriorCount_"):
            axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
            if not isinstance(axes, dict):
                errors.append(f"race bridge posterior axes_json is not an object: {fid}")
                continue
            absent = sorted(RACE_BRIDGE_POSTERIOR_KEYS - set(axes))
            if absent:
                errors.append(f"race bridge posterior field missing metadata keys: {fid} {absent}")
            if axes.get("bridge_operator") != "Bridge_R_fixedC_dynamic_weight":
                errors.append(f"race bridge posterior field has wrong bridge_operator: {fid} {axes.get('bridge_operator')}")
            sensitivity = float(axes.get("sensitivity_width") or 0.0)
            if row.get("dashboard_safe") == "True" and sensitivity > 0.05:
                errors.append(f"race bridge posterior dashboard safety not downgraded despite sensitivity width: {fid}")
            if q_row.get("dashboard_safe") == "True" and sensitivity > 0.05:
                errors.append(f"race bridge posterior Q dashboard safety not downgraded despite sensitivity width: {fid}")
            for column in OPTIONAL_RACE_Q_COLUMNS & set(q.column_names):
                if q_row.get(column) is None:
                    errors.append(f"race bridge posterior Q_tensor missing {column}: {fid}")
        if fid == "SIMRaceBridgeMissingRaceObserver":
            axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
            if isinstance(axes, dict) and axes.get("missing_category_preserved") is not True:
                errors.append("SIMRaceBridgeMissingRaceObserver must declare missing_category_preserved=true")


CNES_CAPACITY_METADATA_KEYS = {"capacity_vector_index", "capacity_family", "generic_beds_forbidden"}
SIH_COST_METADATA_KEYS = {"cost_component", "economic_component_id", "generic_sih_cost_forbidden"}
RUN_CONFIG_CNES_SIH_KEYS = {"schema_version", "source_systems", "attach_stage", "cnes", "sih"}
POPULATION_TENSOR_SUPPORT_KEYS = {"PopulationTensorMode", "SolverBackend", "SolverID", "SparseJacobian", "DenominatorFeedbackWarning", "population_tensor_diagnostics"}
POPULATION_TENSOR_AXIS_KEYS = {"geography_axis", "time_axis", "population_strata_axis", "population_tensor_mode"}
RUN_CONFIG_POPULATION_TENSOR_KEYS = {"schema_version", "source_systems", "attach_stage", "field_id", "tensor_id", "mode", "solver_id", "solver_backend", "denominator_feedback_warning", "independent_denominator_mode", "sim_feedback_warning", "source_hashes"}


def _validate_cnes_sih_contract(*, root: Path, v, q, run_config: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
    rows = v.to_pylist()
    q_rows = {str(row.get("field_id")): row for row in q.to_pylist() if row.get("field_id") is not None}
    cnes_capacity_rows = [row for row in rows if str(row.get("field_id", "")).startswith("cnes_capacity_")]
    sih_cost_rows = [row for row in rows if str(row.get("field_id", "")).startswith("sih_cost_")]
    cnes_sih_rows = cnes_capacity_rows + sih_cost_rows
    if not cnes_sih_rows:
        return

    run_meta = run_config.get("cnes_sih")
    manifest_meta = manifest.get("cnes_sih")
    if not isinstance(run_meta, dict):
        errors.append("RunConfig.json missing cnes_sih metadata while CNES/SIH fields exist")
        run_meta = {}
    if not isinstance(manifest_meta, dict):
        errors.append("ReproducibilityManifest.json missing cnes_sih metadata while CNES/SIH fields exist")
        manifest_meta = {}

    missing_run_keys = sorted(RUN_CONFIG_CNES_SIH_KEYS - set(run_meta))
    if missing_run_keys:
        errors.append(f"RunConfig.cnes_sih missing keys: {missing_run_keys}")
    if run_meta.get("source_systems") != ["CNES-ST", "SIH-RD"]:
        errors.append("RunConfig.cnes_sih.source_systems must equal ['CNES-ST', 'SIH-RD']")
    if run_meta.get("attach_stage") != "she_build":
        errors.append("RunConfig.cnes_sih.attach_stage must be she_build")
    if manifest_meta.get("cnes", {}).get("generic_beds_blocked") is not True:
        errors.append("ReproducibilityManifest.cnes_sih.cnes.generic_beds_blocked must be true")
    if manifest_meta.get("sih", {}).get("generic_sih_cost_blocked") is not True:
        errors.append("ReproducibilityManifest.cnes_sih.sih.generic_sih_cost_blocked must be true")
    if manifest_meta.get("sih", {}).get("diagnostic_topology_preserved") is not True:
        errors.append("ReproducibilityManifest.cnes_sih.sih.diagnostic_topology_preserved must be true")

    if not (root / "Tables" / "slice5a_cnes_capacity_summary.parquet").exists():
        errors.append("CNES capacity fields exist but Tables/slice5a_cnes_capacity_summary.parquet is missing")
    if not (root / "Tables" / "slice5a_sih_cost_summary.parquet").exists():
        errors.append("SIH cost fields exist but Tables/slice5a_sih_cost_summary.parquet is missing")

    for row in cnes_capacity_rows:
        fid = str(row.get("field_id"))
        axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
        support = _load_json_cell(row.get("support_json"), errors=errors, context=f"V_fields.support_json[{fid}]")
        if not isinstance(axes, dict):
            errors.append(f"CNES capacity field axes_json is not an object: {fid}")
            continue
        missing_axes = sorted(CNES_CAPACITY_METADATA_KEYS - set(axes))
        if missing_axes:
            errors.append(f"CNES capacity field missing axes metadata: {fid} {missing_axes}")
        if not isinstance(support, dict) or support.get("capacity_vector_index") != axes.get("capacity_vector_index"):
            errors.append(f"CNES capacity field support must preserve capacity_vector_index: {fid}")
        if q_rows.get(fid) is None:
            errors.append(f"CNES capacity field missing Q_tensor row: {fid}")
        if row.get("unit") == "beds" and not axes.get("capacity_vector_index"):
            errors.append(f"CNES generic beds field emitted without capacity_vector_index: {fid}")

    for row in sih_cost_rows:
        fid = str(row.get("field_id"))
        axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
        support = _load_json_cell(row.get("support_json"), errors=errors, context=f"V_fields.support_json[{fid}]")
        if not isinstance(axes, dict):
            errors.append(f"SIH cost field axes_json is not an object: {fid}")
            continue
        missing_axes = sorted(SIH_COST_METADATA_KEYS - set(axes))
        if missing_axes:
            errors.append(f"SIH cost field missing axes metadata: {fid} {missing_axes}")
        if not isinstance(support, dict) or support.get("cost_component") != axes.get("cost_component"):
            errors.append(f"SIH cost field support must preserve cost_component: {fid}")
        if q_rows.get(fid) is None:
            errors.append(f"SIH cost field missing Q_tensor row: {fid}")



def _validate_population_tensor_contract(*, root: Path, v, q, run_config: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
    rows = v.to_pylist()
    q_rows = {str(row.get("field_id")): row for row in q.to_pylist() if row.get("field_id") is not None}
    population_rows = [row for row in rows if str(row.get("field_id", "")).startswith("population_tensor_")]
    if not population_rows and "population_tensor" not in run_config and "population_tensor" not in manifest:
        return

    run_meta = run_config.get("population_tensor")
    manifest_meta = manifest.get("population_tensor")
    if not isinstance(run_meta, dict):
        errors.append("RunConfig.json missing population_tensor metadata while population tensor is present")
        run_meta = {}
    if not isinstance(manifest_meta, dict):
        errors.append("ReproducibilityManifest.json missing population_tensor metadata while population tensor is present")
        manifest_meta = {}

    missing_run_keys = sorted(RUN_CONFIG_POPULATION_TENSOR_KEYS - set(run_meta))
    if missing_run_keys:
        errors.append(f"RunConfig.population_tensor missing keys: {missing_run_keys}")
    if run_meta.get("source_systems") != ["SIDRA"]:
        errors.append("RunConfig.population_tensor.source_systems must equal ['SIDRA']")
    if run_meta.get("attach_stage") not in {"population_solver", "standalone_population_tensor"}:
        errors.append("RunConfig.population_tensor.attach_stage must be population_solver or standalone_population_tensor")
    if manifest_meta.get("field_id") and run_meta.get("field_id") and manifest_meta.get("field_id") != run_meta.get("field_id"):
        errors.append("RunConfig.population_tensor.field_id and ReproducibilityManifest.population_tensor.field_id disagree")
    if not (root / "Tables" / "population_tensor_diagnostics.parquet").exists():
        errors.append("population tensor metadata exists but Tables/population_tensor_diagnostics.parquet is missing")

    for row in population_rows:
        fid = str(row.get("field_id"))
        support = _load_json_cell(row.get("support_json"), errors=errors, context=f"V_fields.support_json[{fid}]")
        axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
        provenance_cell = row.get("provenance_json")
        if provenance_cell in (None, ""):
            provenance_cell = row.get("provenance")
        warnings_source_cell = row.get("warnings_json")
        if warnings_source_cell in (None, ""):
            warnings_source_cell = row.get("warnings")
        provenance = _load_json_cell(provenance_cell, errors=errors, context=f"V_fields.provenance[{fid}]")
        warnings_cell = _load_json_cell(warnings_source_cell, errors=errors, context=f"V_fields.warnings[{fid}]")

        if not isinstance(support, dict):
            errors.append(f"population tensor field support_json is not an object: {fid}")
            support = {}
        if not isinstance(axes, dict):
            errors.append(f"population tensor field axes_json is not an object: {fid}")
            axes = {}
        if not isinstance(provenance, list) or "population_tensor" not in provenance:
            errors.append(f"population tensor field provenance must include population_tensor: {fid}")

        missing_support = sorted(POPULATION_TENSOR_SUPPORT_KEYS - set(support))
        if missing_support:
            errors.append(f"population tensor field missing support metadata: {fid} {missing_support}")
        missing_axes = sorted(POPULATION_TENSOR_AXIS_KEYS - set(axes))
        if missing_axes:
            errors.append(f"population tensor field missing axes metadata: {fid} {missing_axes}")
        if axes.get("population_tensor_mode") != support.get("PopulationTensorMode"):
            errors.append(f"population tensor support/axes mode mismatch: {fid}")
        if q_rows.get(fid) is None:
            errors.append(f"population tensor field missing Q_tensor row: {fid}")

        sim_feedback = bool(support.get("DenominatorFeedbackWarning"))
        if sim_feedback:
            if row.get("dashboard_safe") in {True, "true", "True"}:
                errors.append(f"SIM-informed population tensor field cannot be dashboard_safe=true: {fid}")
            if row.get("state") not in {"fragile", "experimental", "blocked", "quarantined"}:
                errors.append(f"SIM-informed population tensor field must be fragile/experimental or worse: {fid}")
            if isinstance(warnings_cell, list) and "sim_informed_population_feedback_risk" not in warnings_cell:
                errors.append(f"SIM-informed population tensor field missing feedback-risk warning: {fid}")


def _validate_parquet_contracts(*, root: Path, run_config: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
    try:
        v = _read(root / "V_fields.parquet")
        q = _read(root / "Q_tensor.parquet")
        vd = _read(root / "VariableDictionary.parquet")
        edges = _read(root / "E_DAG.parquet")
        warnings_table = _read(root / "Warnings.parquet")
        failed_branches = _read(root / "FailedBranches.parquet")
        quarantined = _read(root / "QuarantinedFields.parquet")
        forced = _read(root / "ForcedFields.parquet")
        model_assoc = _read(root / "ModelAssociations.parquet")
        residual_assoc = _read(root / "ResidualAssociations.parquet")
        hypotheses = _read(root / "Hypotheses.parquet")
    except Exception as exc:
        errors.append(f"parquet read failure: {exc}")
        return
    _require_columns(table_name="V_fields", actual=set(v.column_names), required=REQUIRED_V_FIELDS_COLUMNS, errors=errors)
    _require_columns(table_name="E_DAG", actual=set(edges.column_names), required=REQUIRED_E_DAG_COLUMNS, errors=errors)
    _require_columns(table_name="Q_tensor", actual=set(q.column_names), required=REQUIRED_Q_TENSOR_COLUMNS, errors=errors)
    _require_columns(table_name="VariableDictionary", actual=set(vd.column_names), required=REQUIRED_VARIABLE_DICTIONARY_COLUMNS, errors=errors)
    if q.num_rows == 0:
        errors.append("Q_tensor is empty")
    v_ids = _nonnull(_column_values(v, "field_id"))
    q_ids = _nonnull(_column_values(q, "field_id"))
    vd_ids = _nonnull(_column_values(vd, "field_id"))
    if not v_ids:
        errors.append("V_fields has no field_id values")
    if not v_ids.issubset(vd_ids):
        errors.append(f"VariableDictionary does not cover all V_fields: {sorted(v_ids - vd_ids)}")
    if not v_ids.issubset(q_ids):
        errors.append(f"Q_tensor does not cover all V_fields: {sorted(v_ids - q_ids)}")
    if edges.num_rows:
        for col in ["parent_field_id", "child_field_id"]:
            bad = _nonnull(_column_values(edges, col)) - v_ids
            if bad:
                errors.append(f"E_DAG {col} contains IDs absent from V_fields: {sorted(bad)}")
    if warnings_table.num_rows and "field_id" in warnings_table.column_names:
        bad_warnings = {x for x in warnings_table.column("field_id").to_pylist() if x is not None and x not in {"", "run"} and x not in v_ids}
        if bad_warnings:
            errors.append(f"Warnings link to invalid field IDs: {sorted(bad_warnings)}")
    if failed_branches.num_rows and "parent_field_ids" in failed_branches.column_names:
        for idx, raw in enumerate(failed_branches.column("parent_field_ids").to_pylist()):
            parents = _load_json_cell(raw, errors=errors, context=f"FailedBranches.parent_field_ids[{idx}]")
            if parents is None:
                continue
            if not isinstance(parents, list):
                errors.append(f"FailedBranches.parent_field_ids[{idx}] is not a JSON list")
                continue
            bad = {x for x in parents if x not in v_ids}
            if bad:
                errors.append(f"FailedBranches parent IDs absent from V_fields at row {idx}: {sorted(bad)}")
    illegal_ids = set()
    if "state" in v.column_names and "field_id" in v.column_names:
        field_ids = v.column("field_id").to_pylist()
        states = v.column("state").to_pylist()
        illegal_ids = {fid for fid, state in zip(field_ids, states, strict=False) if state == "illegal_excluded"}
    for table_name, table in [("ModelAssociations", model_assoc), ("ResidualAssociations", residual_assoc), ("Hypotheses", hypotheses)]:
        if not illegal_ids:
            break
        for col in FIELD_REFERENCE_COLUMNS & set(table.column_names):
            bad = _nonnull(_column_values(table, col)) & illegal_ids
            if bad:
                errors.append(f"{table_name}.{col} references illegal_excluded fields: {sorted(bad)}")
    for table_name, table in [("QuarantinedFields", quarantined), ("ForcedFields", forced)]:
        for col in FIELD_REFERENCE_COLUMNS & set(table.column_names):
            bad = _nonnull(_column_values(table, col)) - v_ids
            if bad:
                errors.append(f"{table_name}.{col} contains IDs absent from V_fields: {sorted(bad)}")
    _validate_cnes_sih_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)
    _validate_population_tensor_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)
    _validate_race_bridge_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)


def validate_output_bundle(*, run_dir: str, schema_registry: OutputSchemaRegistry | None = None) -> OutputValidationResult:
    """Validate exact 17-key output bundle and mandatory cross-references."""
    schema_registry = schema_registry or OutputSchemaRegistry()
    root = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if not root.exists():
        return OutputValidationResult(ok=False, errors=[f"run_dir does not exist: {root}"], warnings=[])
    if not root.is_dir():
        return OutputValidationResult(ok=False, errors=[f"run_dir is not a directory: {root}"], warnings=[])
    _validate_first_class_keys(root, schema_registry, errors)
    if errors:
        return OutputValidationResult(ok=False, errors=errors, warnings=warnings)
    _user_intent, run_config, manifest = _validate_manifest_and_config(root=root, errors=errors, warnings=warnings)
    _validate_parquet_contracts(root=root, run_config=run_config, manifest=manifest, errors=errors)
    return OutputValidationResult(ok=not errors, errors=errors, warnings=warnings)
