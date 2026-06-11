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

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.registries.cnes_capacity import CAPACITY_COMPONENTS
from pegasus.registries.sih_cost import COST_COMPONENTS
from pegasus.she.cnes_capacity import summarize_cnes_capacity
from pegasus.she.sih_costs import summarize_sih_costs


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


def _support(years, cod6, cod7, n_events, n_denom=None, extra=None):
    out = {"years": years, "municipalities": cod6, "municipalities_ibge_cod7": cod7, "n_events": n_events, "n_eff": n_events, "missingness": 0.0, "denom_fragility": 1.0 if n_denom is None else 0.0}
    if n_denom is not None:
        out["n_denom"] = n_denom
    if extra:
        out.update(extra)
    return out


def _field(field_id, name, kind, carrier, unit, aggregation, role, source, support, axes, operator, provenance, warnings, state="fragile", dashboard_safe="warning"):
    return {"field_id": field_id, "name": name, "kind": kind, "carrier": carrier, "unit": unit, "support_json": _compact(support), "axes_json": _compact(axes), "aggregation": aggregation, "role_json": _compact(role), "source_json": _compact(source), "operator": operator, "provenance_json": _compact(provenance), "state": state, "warnings_json": _compact(warnings), "lineage_json": _compact({"parent_ids": [], "operator": operator, "created_at": _now()}), "materialization_state": "materialized", "path": None, "dashboard_safe": dashboard_safe}


def _q(field, n_events, n_denom=None, warnings=None):
    return {"field_id": field["field_id"], "n_events": n_events, "n_denom": n_denom, "n_eff": n_events, "cov_S": 1.0, "cov_T": 1.0, "missingness": 0.0, "zero_inflation": 0.0, "denom_fragility": 1.0 if n_denom is None else 0.0, "cv": None, "moran_i": None, "temporal_roughness": None, "spatial_entropy": None, "provenance_risk": 0.2, "race_axis_source": None, "race_axis_target": None, "missing_race_share": None, "emission_prior_strength": None, "race_bridge_cv": None, "sensitivity_width": None, "bridge_mode": None, "state": field["state"], "dashboard_safe": field["dashboard_safe"], "warnings_json": _compact(warnings or []), "computed_at": _now(), "q_schema_version": "1.0"}


def _vd(field, definition, estimand, warning):
    return {"field_id": field["field_id"], "name": field["name"], "definition": definition, "estimand": estimand, "interpretation_warning": warning, "unit": field["unit"], "source_system": ",".join(json.loads(field["source_json"])), "carrier": field["carrier"], "role_json": field["role_json"], "state": field["state"], "dashboard_safe": field["dashboard_safe"]}


def _edge(edge_id, parent, child, operator, params):
    return {"edge_id": edge_id, "parent_field_id": parent, "child_field_id": child, "operator": operator, "params_json": _compact(params), "created_at": _now()}


def _warning(warning_id, field_id, source, severity, code, message):
    return {"warning_id": warning_id, "field_id": field_id, "source": source, "severity": severity, "code": code, "message": message, "inherited_from_json": "[]", "created_at": _now()}


def _failed(branch_id, operator, parents, stage, terms, reason, warnings):
    return {"failed_branch_id": branch_id, "attempted_operator": operator, "parent_field_ids": _compact(parents), "failure_stage": stage, "failed_terms": _compact(terms), "reason": reason, "warnings_json": _compact(warnings), "created_at": _now()}


def write_cnes_sih_fixture_efg_bundle(*, cnes_events_path: str | Path, sih_events_path: str | Path, run_dir: str | Path, municipality_cod6: str | None = None) -> Path:
    cnes_events_path = Path(cnes_events_path); sih_events_path = Path(sih_events_path); run_dir = Path(run_dir)
    create_empty_output_bundle(run_dir)
    cnes = summarize_cnes_capacity(cnes_events_path, municipality_cod6=municipality_cod6)
    sih = summarize_sih_costs(sih_events_path, municipality_cod6=municipality_cod6)
    fields=[]; q_rows=[]; vd_rows=[]; edges=[]
    facilities = _field("cnes_facilities_all", "CNESFacilitiesAll", "extensive_measure", "Facilities", "facilities", "additive", ["facility_stock"], ["CNES-ST"], cnes.support(), {"geography_axis":"DATASUS_COD6", "facility_axis":"CNES"}, "count_facilities", ["CNES-ST", "facility_stock_state"], [])
    fields.append(facilities); q_rows.append(_q(facilities, cnes.facilities_total)); vd_rows.append(_vd(facilities, "Valid CNES-ST facility stock records.", "facility_count", "Facility stock, not utilization."))
    for raw, total in cnes.capacity_sums.items():
        comp = CAPACITY_COMPONENTS[raw]
        f = _field(f"cnes_capacity_{raw.lower()}", f"CNESCapacity_{raw}", "extensive_measure", comp.carrier, comp.unit, "additive", ["capacity_vector_component"], ["CNES-ST"], _support(cnes.years, cnes.municipalities_cod6, cnes.municipalities_cod7, total, extra={"capacity_vector_index": raw}), {"capacity_vector_index": raw, "capacity_family": comp.family}, "sum_capacity_component", ["CNES-ST", "capacity_vector_indexed"], ["generic_beds_forbidden_without_capacity_vector_index"])
        fields.append(f); q_rows.append(_q(f, total)); vd_rows.append(_vd(f, f"CNES vector component {raw}: {comp.label}.", "capacity_vector_component_count", "Vector-indexed capacity, not generic beds.")); edges.append(_edge(f"edge_cnes_facilities_to_{raw.lower()}", facilities["field_id"], f["field_id"], "capacity_component_sum", {"capacity_vector_index": raw}))
    invalid_flag = _field("cnes_invalid_boolean_flag_share", "CNESInvalidBooleanFlagShare", "marked_functional", "Facilities", "proportion", "statistical_functional", ["quality_observer"], ["CNES-ST"], _support(cnes.years, cnes.municipalities_cod6, cnes.municipalities_cod7, cnes.invalid_flag_count, cnes.flag_count, {"invalid_flag_share": cnes.invalid_flag_share}), {"observer_mark":"invalid_boolean_flag_share"}, "flag_state_share", ["CNES-ST", "Clamp_bool"], ["boolean_outliers_invalid_not_true"])
    fields.append(invalid_flag); q_rows.append(_q(invalid_flag, cnes.invalid_flag_count, cnes.flag_count)); vd_rows.append(_vd(invalid_flag, "Share of CNES boolean flags decoded invalid/unparseable.", "invalid_flag_share", "Outlier booleans are not clamped to true."))
    zero_cnpj = _field("cnes_zero_facility_cnpj_share", "CNESZeroFacilityCNPJShare", "marked_functional", "Facilities", "proportion", "statistical_functional", ["quality_observer", "facility_linkage_gate"], ["CNES-ST"], _support(cnes.years, cnes.municipalities_cod6, cnes.municipalities_cod7, cnes.zero_facility_cnpj_count, cnes.facilities_total, {"zero_facility_cnpj_share": cnes.zero_facility_cnpj_share}), {"observer_mark":"nullified_zero_cnpj"}, "cnpj_nullification_share", ["CNES-ST", "Filter_CNPJ"], ["all_zero_cnpj_nullified_before_facility_flow_linkage"])
    fields.append(zero_cnpj); q_rows.append(_q(zero_cnpj, cnes.zero_facility_cnpj_count, cnes.facilities_total)); vd_rows.append(_vd(zero_cnpj, "Share of all-zero CNPJ-like facility identifiers nullified.", "zero_cnpj_share", "All-zero CNPJ cannot enter facility-flow linkage."))
    admissions = _field("sih_admissions_all", "SIHHospitalAdmissionsAll", "extensive_measure", "HospitalAdmissions", "admissions", "additive", ["hospitalization_count"], ["SIH-RD"], sih.support(), {"diagnostic_topology":"SIH_principal_diagnosis_preserved"}, "count_admissions", ["SIH-RD", "hospital_admission_event"], [])
    fields.append(admissions); q_rows.append(_q(admissions, sih.admissions_total)); vd_rows.append(_vd(admissions, "Valid SIH-RD hospitalization admissions.", "hospital_admission_count", "Billing/hospitalization records, not incident cases."))
    deaths = _field("sih_inpatient_deaths", "SIHInpatientDeaths", "extensive_measure", "HospitalAdmissions", "deaths", "additive", ["inpatient_death_count"], ["SIH-RD"], _support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, sih.inpatient_deaths), {"event_mark":"inpatient_death"}, "count_inpatient_deaths", ["SIH-RD", "death_flag"], [])
    fields.append(deaths); q_rows.append(_q(deaths, sih.inpatient_deaths)); vd_rows.append(_vd(deaths, "SIH admissions marked as inpatient death.", "inpatient_death_count", "Hospital outcome mark, not mortality surveillance.")); edges.append(_edge("edge_sih_admissions_to_deaths", admissions["field_id"], deaths["field_id"], "death_flag_filter", {}))
    fatality = _field("sih_inpatient_fatality", "SIHInpatientFatality", "intensive_density", "HospitalAdmissions", "proportion", "weighted_mean", ["hospital_outcome"], ["SIH-RD"], _support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, sih.inpatient_deaths, sih.admissions_total), {"numerator":"SIH_inpatient_deaths", "denominator":"SIH_admissions"}, "ratio", ["SIH-RD", "hospital_outcome_bridge"], ["not_case_fatality_without_case_incidence_denominator"])
    fields.append(fatality); q_rows.append(_q(fatality, sih.inpatient_deaths, sih.admissions_total)); vd_rows.append(_vd(fatality, "Inpatient deaths divided by SIH admissions.", "inpatient_fatality_proportion", "Not population mortality or disease case fatality.")); edges.append(_edge("edge_sih_deaths_to_fatality", deaths["field_id"], fatality["field_id"], "ratio_numerator", {})); edges.append(_edge("edge_sih_admissions_to_fatality", admissions["field_id"], fatality["field_id"], "ratio_denominator", {}))
    los = _field("sih_mean_los", "SIHMeanLengthOfStay", "marked_functional", "HospitalAdmissions", "mean_duration", "weighted_mean", ["hospitalization_duration"], ["SIH-RD"], _support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, sih.stay_days_total, sih.stay_days_observed), {"duration_mark":"DIAS_PERM_or_QT_DIARIAS"}, "mean_duration", ["SIH-RD", "stay_length_days"], ["invalid_los_not_clamped_to_zero"])
    fields.append(los); q_rows.append(_q(los, sih.stay_days_total, sih.stay_days_observed)); vd_rows.append(_vd(los, "Mean SIH length of stay.", "mean_length_of_stay", "Invalid LOS is not clamped to zero.")); edges.append(_edge("edge_sih_admissions_to_los", admissions["field_id"], los["field_id"], "mean_duration_denominator", {}))
    for raw, comp in COST_COMPONENTS.items():
        total = sih.cost_sums[raw]; denom = sih.cost_observed[raw]
        f = _field(f"sih_cost_{raw.lower()}", f"SIHCost_{raw}", "marked_functional", comp.carrier, comp.unit, "additive", ["economic_burden_component"], ["SIH-RD"], _support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, total, denom, {"cost_component": raw}), {"cost_component": raw, "economic_component_id": comp.component_id}, "sum_cost_component", ["SIH-RD", "component_specific_cost"], ["sih_cost_components_not_interchangeable"])
        fields.append(f); q_rows.append(_q(f, total, denom)); vd_rows.append(_vd(f, f"SIH cost component {raw}: {comp.label}.", "component_specific_cost_sum", "VAL_SH, VAL_SP, VAL_UTI, and VAL_TOT remain separate.")); edges.append(_edge(f"edge_sih_admissions_to_cost_{raw.lower()}", admissions["field_id"], f["field_id"], "cost_component_support", {"component": raw}))
    warnings = [
        _warning("slice5a_generic_beds_forbidden", "run", "CNES-ST", "abort", "generic_beds_forbidden_without_capacity_index", "Generic CNES beds/capacity is illegal without an explicit vector index."),
        _warning("slice5a_cnpj_nullified", zero_cnpj["field_id"], "CNES-ST", "warning", "all_zero_cnpj_nullified", "All-zero facility CNPJ-like identifiers were nullified before linkage."),
        _warning("slice5a_sih_cost_components", "run", "SIH-RD", "abort", "generic_sih_cost_forbidden", "Generic SIH cost is illegal when component-specific semantics are required."),
        _warning("slice5a_sih_topology", admissions["field_id"], "SIH-RD", "warning", "sih_principal_secondary_diagnosis_topology_preserved", "SIH principal diagnosis, secondary diagnoses, procedures, and SIM underlying cause are distinct topologies."),
    ]
    failed = [
        _failed("failed_cnes_generic_beds_without_capacity_index", "RN", [], "carrier", ["carrier"], "Generic CNES Beds carrier requested without capacity-vector index.", ["generic_beds_forbidden_without_capacity_vector_index"]),
        _failed("failed_facility_flow_unsanitized_cnpj", "FacilityFlow", [zero_cnpj["field_id"]], "provenance", ["provenance", "quality"], "Unsanitized/all-zero CNPJ-like identifiers cannot enter facility-flow linkage.", ["all_zero_cnpj_nullified_before_facility_flow_linkage"]),
        _failed("failed_sih_generic_cost_pooling", "EconomicBurden", [f"sih_cost_{raw.lower()}" for raw in COST_COMPONENTS], "unit", ["unit", "carrier"], "Generic SIH cost pooling rejected; components remain separate.", ["sih_cost_components_not_interchangeable"]),
        _failed("failed_sih_diagnostic_topology_collapse", "DiagnosticProjection", [admissions["field_id"]], "axes", ["axes"], "SIH principal/secondary topology cannot be erased without explicit projection.", ["diagnostic_topology_preserved"]),
    ]
    _write_rows_like(run_dir / "V_fields.parquet", fields); _write_rows_like(run_dir / "Q_tensor.parquet", q_rows); _write_rows_like(run_dir / "VariableDictionary.parquet", vd_rows); _write_rows_like(run_dir / "E_DAG.parquet", edges); _write_rows_like(run_dir / "Warnings.parquet", warnings); _write_rows_like(run_dir / "FailedBranches.parquet", failed)
    for name in [
        "ModelAssociations.parquet",
        "ResidualAssociations.parquet",
        "Hypotheses.parquet",
        "QuarantinedFields.parquet",
        "ForcedFields.parquet",
    ]:
        _write_rows_like(run_dir / name, [])
    (run_dir / "Tables").mkdir(exist_ok=True)
    pl.DataFrame([{**{"facilities_total": cnes.facilities_total, "zero_facility_cnpj_count": cnes.zero_facility_cnpj_count, "invalid_flag_count": cnes.invalid_flag_count}, **{f"capacity_{k}": v for k, v in cnes.capacity_sums.items()}}]).write_parquet(run_dir / "Tables" / "slice5a_cnes_capacity_summary.parquet")
    pl.DataFrame([{**{"admissions_total": sih.admissions_total, "inpatient_deaths": sih.inpatient_deaths, "mean_los": sih.mean_los}, **{f"cost_{k}": v for k, v in sih.cost_sums.items()}}]).write_parquet(run_dir / "Tables" / "slice5a_sih_cost_summary.parquet")
    user_intent = {"workflow": "slice5a_cnes_sih_fixture", "municipality_cod6": municipality_cod6, "source_systems": ["CNES-ST", "SIH-RD"]}
    run_config = {"schema_version": "1.0", "workflow": "slice5a_cnes_sih_fixture", "source_systems": ["CNES-ST", "SIH-RD"], "source_hashes": {"cnes_events": str(cnes_events_path), "sih_events": str(sih_events_path)}, "registry_hashes": {"cnes_capacity": "fixture_v1", "sih_cost": "fixture_v1"}, "slice5a": {"generic_beds_blocked": True, "generic_sih_cost_blocked": True, "facility_flow_requires_sanitized_cnpj": True}}
    stage_status = {stage: "skipped" for stage in COMPILE_TELEMETRY_STAGES}
    stage_wall_seconds = {stage: 0.0 for stage in COMPILE_TELEMETRY_STAGES}
    for stage in ["datasus_decode", "she_build", "efg_build", "q_tensor", "output_serialization", "output_validation"]:
        if stage in stage_status:
            stage_status[stage] = "success"
    for stage in ["population_solver", "stdfm", "pirs_model", "pirs_hsic"]:
        if stage in stage_status:
            stage_status[stage] = "blocked"
    telemetry = {
        "total_wall_seconds": 0.0,
        "stage_status": stage_status,
        "stage_wall_seconds": stage_wall_seconds,
        "stage_errors": {
            "population_solver": "No population denominator tensor invoked in Slice 5A CNES/SIH fixture path.",
            "stdfm": "ST-DFM scaffold remains blocked for Slice 5A CNES/SIH fixture path.",
            "pirs_model": "PIRS model fitting is outside Slice 5A.",
            "pirs_hsic": "HSIC scanning is outside Slice 5A.",
        },
        "resource_summary": {
            "peak_rss_mb": None,
            "peak_vram_mb": None,
            "duckdb_temp_bytes": None,
            "rows_read": {"cnes_events": cnes.facilities_total, "sih_events": sih.admissions_total},
            "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows), "FailedBranches": len(failed)},
            "parquet_bytes_written": 0,
        },
    }
    manifest = {"schema_version": "1.0", "run_id": run_dir.name, "generated_at": _now(), "workflow": "slice5a_cnes_sih_fixture", "source_hashes": run_config["source_hashes"], "registry_hashes": run_config["registry_hashes"], "telemetry": telemetry, "environment": {"python": sys.version.split()[0], "platform": platform.platform()}}
    p_vector = {"source_systems": ["CNES-ST", "SIH-RD"], "field_count": len(fields), "blocked_outputs": [f["failed_branch_id"] for f in failed], "provenance": {row["field_id"]: json.loads(row["provenance_json"]) for row in fields}}
    (run_dir / "UserIntent.json").write_text(_json(user_intent), encoding="utf-8"); (run_dir / "RunConfig.json").write_text(_json(run_config), encoding="utf-8"); (run_dir / "ReproducibilityManifest.json").write_text(_json(manifest), encoding="utf-8"); (run_dir / "P_vector.json").write_text(_json(p_vector), encoding="utf-8")
    expected = set(OUTPUT_BUNDLE_FILES.values()); found = {p.name for p in run_dir.iterdir()}
    if found - expected or expected - found:
        raise RuntimeError(f"Invalid first-class bundle keys. extra={sorted(found - expected)} missing={sorted(expected - found)}")
    return run_dir
