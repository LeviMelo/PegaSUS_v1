from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pegasus.geo.state_panel import clean_datasus_municipalities

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.registries.cnes_capacity import CAPACITY_COMPONENTS
from pegasus.registries.sih_cost import COST_COMPONENTS
from pegasus.she.cnes_capacity import CNESCapacitySummary, summarize_cnes_capacity
from pegasus.she.sih_costs import SIHCostSummary, summarize_sih_costs


@dataclass(frozen=True)
class CNESSIHBuiltRows:
    fields: list[dict[str, Any]]
    q_rows: list[dict[str, Any]]
    vd_rows: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    failed_branches: list[dict[str, Any]]
    cnes_summary: CNESCapacitySummary
    sih_summary: SIHCostSummary
    metadata: dict[str, Any]


CNES_SIH_FIELD_PREFIXES = (
    "cnes_",
    "sih_",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _read_schema(path: Path) -> pa.Schema:
    return pq.read_table(path).schema


def write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = _read_schema(path)
    shaped = [{name: row.get(name) for name in schema.names} for row in rows]
    if shaped:
        table = pa.Table.from_pylist(shaped, schema=schema)
    else:
        table = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    pq.write_table(table, path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def append_rows_like(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    remove_field_prefixes: tuple[str, ...] = (),
    id_column: str | None = None,
) -> None:
    existing = read_rows(path)
    if remove_field_prefixes:
        def keep(row: dict[str, Any]) -> bool:
            value = str(row.get(id_column or "field_id") or "")
            return not any(value.startswith(prefix) for prefix in remove_field_prefixes)

        existing = [row for row in existing if keep(row)]
    write_rows_like(path, existing + rows)


def _support(
    years: list[int],
    cod6: list[str],
    cod7: list[str],
    n_events: float | int | None,
    n_denom: float | int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "years": years,
        "municipalities": cod6,
        "municipalities_ibge_cod7": cod7,
        "n_events": n_events,
        "n_eff": n_denom if n_denom is not None else n_events,
        "missingness": 0.0,
        "denom_fragility": 1.0 if n_denom is None else 0.0,
    }
    if n_denom is not None:
        out["n_denom"] = n_denom
    if extra:
        out.update(extra)
    return out


def _lineage_hash(field_id: str, operator: str, support: dict[str, Any], axes: dict[str, Any]) -> str:
    return content_hash(
        {
            "field_id": field_id,
            "operator": operator,
            "support": support,
            "axes": axes,
            "slice": "5b_cnes_sih",
        }
    )


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
    operator: str,
    provenance: list[str],
    warnings: list[str],
    state: str = "fragile",
    dashboard_safe: str = "warning",
    registry_hash: str = "slice5_cnes_sih_registry_v1",
) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "name": name,
        "kind": kind,
        "carrier": carrier,
        "unit": unit,
        "aggregation": aggregation,
        "role": _compact(role),
        "source": _compact(source),
        "support_json": _compact(support),
        "axes_json": _compact(axes),
        "operator": operator,
        "provenance": _compact(provenance),
        "state": state,
        "dashboard_safe": dashboard_safe,
        "warnings": _compact(warnings),
        "lineage_hash": _lineage_hash(field_id, operator, support, axes),
        "registry_hash": registry_hash,
        "materialization_state": "materialized",
        "path": None,
    }


def _q(field: dict[str, Any], n_events: float | int | None, n_denom: float | int | None = None, warnings: list[str] | None = None) -> dict[str, Any]:
    n_eff = n_denom if n_denom is not None else n_events
    return {
        "field_id": field["field_id"],
        "n_events": float(n_events) if n_events is not None else None,
        "n_denom": float(n_denom) if n_denom is not None else None,
        "n_eff": float(n_eff) if n_eff is not None else None,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 1.0 if n_denom is None else 0.0,
        "cv": None,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.2,
        "race_axis_source": None,
        "race_axis_target": None,
        "missing_race_share": None,
        "emission_prior_strength": None,
        "race_bridge_cv": None,
        "sensitivity_width": None,
        "bridge_mode": None,
        "state": field["state"],
        "dashboard_safe": field["dashboard_safe"],
        "warnings": _compact(warnings or []),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def _vd(field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "display_name": field["name"],
        "technical_name": field["field_id"],
        "definition": definition,
        "estimand_label": estimand,
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


def _edge(edge_id: str, parent: str, child: str, operator: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": edge_id,
        "parent_field_id": parent,
        "child_field_id": child,
        "operator": operator,
        "operator_params_json": _compact(params),
        "registry_versions_json": _compact({"slice5_cnes_sih": "v1"}),
        "created_at": _now(),
    }


def _warning(warning_id: str, field_id: str, source: str, severity: str, code: str, message: str) -> dict[str, Any]:
    return {
        "warning_id": warning_id,
        "field_id": field_id,
        "source": source,
        "severity": severity,
        "code": code,
        "message": message,
        "inherited_from": "[]",
        "created_at": _now(),
    }


def _failed(branch_id: str, operator: str, parents: list[str], stage: str, terms: list[str], reason: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "failed_branch_id": branch_id,
        "attempted_operator": operator,
        "parent_field_ids": _compact(parents),
        "failure_stage": stage,
        "failed_terms": _compact(terms),
        "reason": reason,
        "warnings": _compact(warnings),
        "created_at": _now(),
    }


def _cnes_sih_metadata(cnes: CNESCapacitySummary, sih: SIHCostSummary) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source_systems": ["CNES-ST", "SIH-RD"],
        "attach_stage": "she_build",
        "cnes": {
            "capacity_vector_indices": sorted(CAPACITY_COMPONENTS),
            "capacity_family": sorted({component.family for component in CAPACITY_COMPONENTS.values()}),
            "generic_beds_blocked": True,
            "facility_flow_requires_sanitized_cnpj": True,
            "zero_facility_cnpj_count": cnes.zero_facility_cnpj_count,
            "invalid_flag_count": cnes.invalid_flag_count,
            "invalid_flag_share": cnes.invalid_flag_share,
            "zero_facility_cnpj_share": cnes.zero_facility_cnpj_share,
        },
        "sih": {
            "cost_components": sorted(COST_COMPONENTS),
            "generic_sih_cost_blocked": True,
            "diagnostic_topology_preserved": True,
            "principal_diagnosis_preserved": True,
            "inpatient_fatality_is_hospital_outcome_not_population_mortality": True,
            "admissions_total": sih.admissions_total,
            "inpatient_deaths": sih.inpatient_deaths,
        },
    }


def build_cnes_sih_rows(
    *,
    cnes_events_path: str | Path,
    sih_events_path: str | Path,
    municipality_cod6: str | None = None,
) -> CNESSIHBuiltRows:
    cnes_events_path = Path(cnes_events_path)
    sih_events_path = Path(sih_events_path)
    cnes = summarize_cnes_capacity(cnes_events_path, municipality_cod6=municipality_cod6)
    sih = summarize_sih_costs(sih_events_path, municipality_cod6=municipality_cod6)

    fields: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    vd_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    facilities = _field(
        field_id="cnes_facilities_all",
        name="CNESFacilitiesAll",
        kind="extensive_measure",
        carrier="Facilities",
        unit="facilities",
        aggregation="additive",
        role=["facility_stock"],
        source=["CNES-ST"],
        support=cnes.support(),
        axes={"geography_axis": "DATASUS_COD6", "facility_axis": "CNES", "source_axis": "CNES_ST_FACILITY_STOCK"},
        operator="count_facilities",
        provenance=["CNES-ST", "facility_stock_state"],
        warnings=[],
    )
    fields.append(facilities)
    q_rows.append(_q(facilities, cnes.facilities_total))
    vd_rows.append(_vd(facilities, "Valid CNES-ST facility stock records.", "facility_count", "Facility stock, not utilization."))

    for raw, total in sorted(cnes.capacity_sums.items()):
        comp = CAPACITY_COMPONENTS[raw]
        support = _support(
            cnes.years,
            cnes.municipalities_cod6,
            cnes.municipalities_cod7,
            total,
            extra={"capacity_vector_index": raw, "capacity_family": comp.family},
        )
        axes = {
            "capacity_vector_index": raw,
            "capacity_family": comp.family,
            "capacity_label": comp.label,
            "raw_field": raw,
            "generic_beds_forbidden": True,
        }
        field = _field(
            field_id=f"cnes_capacity_{raw.lower()}",
            name=f"CNESCapacity_{raw}",
            kind="extensive_measure",
            carrier=comp.carrier,
            unit=comp.unit,
            aggregation="additive",
            role=["capacity_vector_component"],
            source=["CNES-ST"],
            support=support,
            axes=axes,
            operator="sum_capacity_component",
            provenance=["CNES-ST", "capacity_vector_indexed", raw],
            warnings=["generic_beds_forbidden_without_capacity_vector_index"],
            registry_hash="cnes_capacity_registry_v1",
        )
        fields.append(field)
        q_rows.append(_q(field, total, warnings=["generic_beds_forbidden_without_capacity_vector_index"]))
        vd_rows.append(
            _vd(
                field,
                f"CNES vector component {raw}: {comp.label}.",
                "capacity_vector_component_count",
                "Vector-indexed capacity component; never a generic bed count.",
            )
        )
        edges.append(
            _edge(
                f"edge_cnes_facilities_to_{raw.lower()}",
                facilities["field_id"],
                field["field_id"],
                "capacity_component_sum",
                {"capacity_vector_index": raw, "capacity_family": comp.family},
            )
        )

    invalid_flag = _field(
        field_id="cnes_invalid_boolean_flag_share",
        name="CNESInvalidBooleanFlagShare",
        kind="marked_functional",
        carrier="Facilities",
        unit="proportion",
        aggregation="statistical_functional",
        role=["quality_observer"],
        source=["CNES-ST"],
        support=_support(
            cnes.years,
            cnes.municipalities_cod6,
            cnes.municipalities_cod7,
            cnes.invalid_flag_count,
            cnes.flag_count,
            {"invalid_flag_share": cnes.invalid_flag_share},
        ),
        axes={"observer_mark": "invalid_boolean_flag_share", "decoder": "clamp_bool", "invalid_not_true": True},
        operator="flag_state_share",
        provenance=["CNES-ST", "Clamp_bool"],
        warnings=["boolean_outliers_invalid_not_true"],
    )
    fields.append(invalid_flag)
    q_rows.append(_q(invalid_flag, cnes.invalid_flag_count, cnes.flag_count, ["boolean_outliers_invalid_not_true"]))
    vd_rows.append(_vd(invalid_flag, "Share of CNES boolean flags decoded invalid/unparseable.", "invalid_flag_share", "Outlier booleans are not clamped to true."))

    zero_cnpj = _field(
        field_id="cnes_zero_facility_cnpj_share",
        name="CNESZeroFacilityCNPJShare",
        kind="marked_functional",
        carrier="Facilities",
        unit="proportion",
        aggregation="statistical_functional",
        role=["quality_observer", "facility_linkage_gate"],
        source=["CNES-ST"],
        support=_support(
            cnes.years,
            cnes.municipalities_cod6,
            cnes.municipalities_cod7,
            cnes.zero_facility_cnpj_count,
            cnes.facilities_total,
            {"zero_facility_cnpj_share": cnes.zero_facility_cnpj_share},
        ),
        axes={"observer_mark": "nullified_zero_cnpj", "facility_flow_linkage_gate": "sanitized_cnpj_required"},
        operator="cnpj_nullification_share",
        provenance=["CNES-ST", "Filter_CNPJ"],
        warnings=["all_zero_cnpj_nullified_before_facility_flow_linkage"],
    )
    fields.append(zero_cnpj)
    q_rows.append(_q(zero_cnpj, cnes.zero_facility_cnpj_count, cnes.facilities_total, ["all_zero_cnpj_nullified_before_facility_flow_linkage"]))
    vd_rows.append(_vd(zero_cnpj, "Share of all-zero CNPJ-like facility identifiers nullified.", "zero_cnpj_share", "All-zero CNPJ cannot enter facility-flow linkage."))

    admissions = _field(
        field_id="sih_admissions_all",
        name="SIHHospitalAdmissionsAll",
        kind="extensive_measure",
        carrier="HospitalAdmissions",
        unit="admissions",
        aggregation="additive",
        role=["hospitalization_count"],
        source=["SIH-RD"],
        support=sih.support(),
        axes={
            "diagnostic_topology": "SIH_principal_diagnosis_preserved",
            "principal_diagnosis_axis": "DIAG_PRINC",
            "secondary_diagnosis_axis": "DIAG_SECUN",
            "source_event": "AIH_hospitalization_record",
        },
        operator="count_admissions",
        provenance=["SIH-RD", "hospital_admission_event"],
        warnings=[],
    )
    fields.append(admissions)
    q_rows.append(_q(admissions, sih.admissions_total))
    vd_rows.append(_vd(admissions, "Valid SIH-RD hospitalization admissions.", "hospital_admission_count", "Billing/hospitalization records, not incident cases."))

    deaths = _field(
        field_id="sih_inpatient_deaths",
        name="SIHInpatientDeaths",
        kind="extensive_measure",
        carrier="HospitalAdmissions",
        unit="deaths",
        aggregation="additive",
        role=["inpatient_death_count"],
        source=["SIH-RD"],
        support=_support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, sih.inpatient_deaths),
        axes={"event_mark": "inpatient_death", "hospital_outcome_axis": "MORTE"},
        operator="count_inpatient_deaths",
        provenance=["SIH-RD", "death_flag"],
        warnings=[],
    )
    fields.append(deaths)
    q_rows.append(_q(deaths, sih.inpatient_deaths))
    vd_rows.append(_vd(deaths, "SIH admissions marked as inpatient death.", "inpatient_death_count", "Hospital outcome mark, not mortality surveillance."))
    edges.append(_edge("edge_sih_admissions_to_deaths", admissions["field_id"], deaths["field_id"], "death_flag_filter", {}))

    fatality = _field(
        field_id="sih_inpatient_fatality",
        name="SIHInpatientFatality",
        kind="intensive_density",
        carrier="HospitalAdmissions",
        unit="proportion",
        aggregation="weighted_mean",
        role=["hospital_outcome"],
        source=["SIH-RD"],
        support=_support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, sih.inpatient_deaths, sih.admissions_total),
        axes={"numerator": "SIH_inpatient_deaths", "denominator": "SIH_admissions", "not_population_mortality": True},
        operator="ratio",
        provenance=["SIH-RD", "hospital_outcome_bridge"],
        warnings=["not_case_fatality_without_case_incidence_denominator"],
    )
    fields.append(fatality)
    q_rows.append(_q(fatality, sih.inpatient_deaths, sih.admissions_total, ["not_case_fatality_without_case_incidence_denominator"]))
    vd_rows.append(_vd(fatality, "Inpatient deaths divided by SIH admissions.", "inpatient_fatality_proportion", "Not population mortality or disease case fatality."))
    edges.append(_edge("edge_sih_deaths_to_fatality", deaths["field_id"], fatality["field_id"], "ratio_numerator", {}))
    edges.append(_edge("edge_sih_admissions_to_fatality", admissions["field_id"], fatality["field_id"], "ratio_denominator", {}))

    los = _field(
        field_id="sih_mean_los",
        name="SIHMeanLengthOfStay",
        kind="marked_functional",
        carrier="HospitalAdmissions",
        unit="mean_duration",
        aggregation="weighted_mean",
        role=["hospitalization_duration"],
        source=["SIH-RD"],
        support=_support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, sih.stay_days_total, sih.stay_days_observed),
        axes={"duration_mark": "DIAS_PERM_or_QT_DIARIAS", "invalid_los_not_zero": True},
        operator="mean_duration",
        provenance=["SIH-RD", "stay_length_days"],
        warnings=["invalid_los_not_clamped_to_zero"],
    )
    fields.append(los)
    q_rows.append(_q(los, sih.stay_days_total, sih.stay_days_observed, ["invalid_los_not_clamped_to_zero"]))
    vd_rows.append(_vd(los, "Mean SIH length of stay.", "mean_length_of_stay", "Invalid LOS is not clamped to zero."))
    edges.append(_edge("edge_sih_admissions_to_los", admissions["field_id"], los["field_id"], "mean_duration_denominator", {}))

    for raw, comp in sorted(COST_COMPONENTS.items()):
        total = sih.cost_sums[raw]
        denom = sih.cost_observed[raw]
        support = _support(sih.years, sih.municipalities_cod6, sih.municipalities_cod7, total, denom, {"cost_component": raw})
        axes = {
            "cost_component": raw,
            "economic_component_id": comp.component_id,
            "component_label": comp.label,
            "generic_sih_cost_forbidden": True,
        }
        field = _field(
            field_id=f"sih_cost_{raw.lower()}",
            name=f"SIHCost_{raw}",
            kind="marked_functional",
            carrier=comp.carrier,
            unit=comp.unit,
            aggregation="additive",
            role=["economic_burden_component"],
            source=["SIH-RD"],
            support=support,
            axes=axes,
            operator="sum_cost_component",
            provenance=["SIH-RD", "component_specific_cost", raw],
            warnings=["sih_cost_components_not_interchangeable"],
            registry_hash="sih_cost_registry_v1",
        )
        fields.append(field)
        q_rows.append(_q(field, total, denom, ["sih_cost_components_not_interchangeable"]))
        vd_rows.append(
            _vd(
                field,
                f"SIH cost component {raw}: {comp.label}.",
                "component_specific_cost_sum",
                "VAL_SH, VAL_SP, VAL_UTI, and VAL_TOT remain separate.",
            )
        )
        edges.append(_edge(f"edge_sih_admissions_to_cost_{raw.lower()}", admissions["field_id"], field["field_id"], "cost_component_support", {"component": raw}))

    warnings = [
        _warning("slice5a_generic_beds_forbidden", "run", "CNES-ST", "abort", "generic_beds_forbidden_without_capacity_index", "Generic CNES beds/capacity is illegal without an explicit vector index."),
        _warning("slice5a_cnpj_nullified", zero_cnpj["field_id"], "CNES-ST", "warning", "all_zero_cnpj_nullified", "All-zero facility CNPJ-like identifiers were nullified before linkage."),
        _warning("slice5a_sih_cost_components", "run", "SIH-RD", "abort", "generic_sih_cost_forbidden", "Generic SIH cost is illegal when component-specific semantics are required."),
        _warning("slice5a_sih_topology", admissions["field_id"], "SIH-RD", "warning", "sih_principal_secondary_diagnosis_topology_preserved", "SIH principal diagnosis, secondary diagnoses, procedures, and SIM underlying cause are distinct topologies."),
    ]
    failed = [
        _failed("failed_cnes_generic_beds_without_capacity_index", "RN", [], "carrier", ["carrier"], "Generic CNES Beds carrier requested without capacity-vector index.", ["generic_beds_forbidden_without_capacity_vector_index"]),
        _failed("failed_facility_flow_unsanitized_cnpj", "FacilityFlow", [zero_cnpj["field_id"]], "provenance", ["provenance", "quality"], "Unsanitized/all-zero CNPJ-like identifiers cannot enter facility-flow linkage.", ["all_zero_cnpj_nullified_before_facility_flow_linkage"]),
        _failed("failed_sih_generic_cost_pooling", "EconomicBurden", [f"sih_cost_{raw.lower()}" for raw in sorted(COST_COMPONENTS)], "unit", ["unit", "carrier"], "Generic SIH cost pooling rejected; components remain separate.", ["sih_cost_components_not_interchangeable"]),
        _failed("failed_sih_diagnostic_topology_collapse", "DiagnosticProjection", [admissions["field_id"]], "axes", ["axes"], "SIH principal/secondary topology cannot be erased without explicit projection.", ["diagnostic_topology_preserved"]),
    ]

    return CNESSIHBuiltRows(
        fields=fields,
        q_rows=q_rows,
        vd_rows=vd_rows,
        edges=edges,
        warnings=warnings,
        failed_branches=failed,
        cnes_summary=cnes,
        sih_summary=sih,
        metadata=_cnes_sih_metadata(cnes, sih),
    )


def write_cnes_sih_summary_tables(run_dir: str | Path, built: CNESSIHBuiltRows) -> dict[str, Path]:
    run_dir = Path(run_dir)
    tables_dir = run_dir / "Tables"
    tables_dir.mkdir(exist_ok=True)
    cnes_path = tables_dir / "slice5a_cnes_capacity_summary.parquet"
    sih_path = tables_dir / "slice5a_sih_cost_summary.parquet"
    cnes = built.cnes_summary
    sih = built.sih_summary
    pl.DataFrame(
        [
            {
                "facilities_total": cnes.facilities_total,
                "zero_facility_cnpj_count": cnes.zero_facility_cnpj_count,
                "invalid_flag_count": cnes.invalid_flag_count,
                **{f"capacity_{key}": value for key, value in cnes.capacity_sums.items()},
            }
        ]
    ).write_parquet(cnes_path)
    pl.DataFrame(
        [
            {
                "admissions_total": sih.admissions_total,
                "inpatient_deaths": sih.inpatient_deaths,
                "mean_los": sih.mean_los,
                **{f"cost_{key}": value for key, value in sih.cost_sums.items()},
            }
        ]
    ).write_parquet(sih_path)
    return {"cnes_capacity_summary": cnes_path, "sih_cost_summary": sih_path}


def write_cnes_sih_fixture_efg_bundle(
    *,
    cnes_events_path: str | Path,
    sih_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
) -> Path:
    cnes_events_path = Path(cnes_events_path)
    sih_events_path = Path(sih_events_path)
    run_dir = Path(run_dir)
    create_empty_output_bundle(run_dir)
    built = build_cnes_sih_rows(cnes_events_path=cnes_events_path, sih_events_path=sih_events_path, municipality_cod6=municipality_cod6)

    write_rows_like(run_dir / "V_fields.parquet", built.fields)
    write_rows_like(run_dir / "Q_tensor.parquet", built.q_rows)
    write_rows_like(run_dir / "VariableDictionary.parquet", built.vd_rows)
    write_rows_like(run_dir / "E_DAG.parquet", built.edges)
    write_rows_like(run_dir / "Warnings.parquet", built.warnings)
    write_rows_like(run_dir / "FailedBranches.parquet", built.failed_branches)
    for name in ["ModelAssociations.parquet", "ResidualAssociations.parquet", "Hypotheses.parquet", "QuarantinedFields.parquet", "ForcedFields.parquet"]:
        write_rows_like(run_dir / name, [])

    summary_paths = write_cnes_sih_summary_tables(run_dir, built)
    source_hashes = {
        "cnes_events": sha256_file(cnes_events_path),
        "sih_events": sha256_file(sih_events_path),
        "cnes_capacity_summary": sha256_file(summary_paths["cnes_capacity_summary"]),
        "sih_cost_summary": sha256_file(summary_paths["sih_cost_summary"]),
    }
    registry_hashes = {
        "cnes_capacity": content_hash({"capacity_components": sorted(CAPACITY_COMPONENTS)}),
        "sih_cost": content_hash({"cost_components": sorted(COST_COMPONENTS)}),
    }
    user_intent = {"workflow": "slice5a_cnes_sih_fixture", "municipality_cod6": municipality_cod6, "source_systems": ["CNES-ST", "SIH-RD"]}
    run_config = {
        "schema_version": "1.0",
        "workflow": "slice5a_cnes_sih_fixture",
        "source_systems": ["CNES-ST", "SIH-RD"],
        "source_hashes": source_hashes,
        "registry_hashes": registry_hashes,
        "cnes_sih": built.metadata,
    }
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
            "rows_read": {"cnes_events": built.cnes_summary.facilities_total, "sih_events": built.sih_summary.admissions_total},
            "rows_written": {
                "V_fields": len(built.fields),
                "Q_tensor": len(built.q_rows),
                "VariableDictionary": len(built.vd_rows),
                "FailedBranches": len(built.failed_branches),
            },
            "parquet_bytes_written": 0,
        },
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "generated_at": _now(),
        "workflow": "slice5a_cnes_sih_fixture",
        "source_hashes": source_hashes,
        "registry_hashes": registry_hashes,
        "telemetry": telemetry,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "cnes_sih": built.metadata,
    }
    p_vector = {
        "source_systems": ["CNES-ST", "SIH-RD"],
        "field_count": len(built.fields),
        "blocked_outputs": [row["failed_branch_id"] for row in built.failed_branches],
        "provenance": {row["field_id"]: json.loads(row["provenance"]) for row in built.fields},
    }
    (run_dir / "UserIntent.json").write_text(_json(user_intent), encoding="utf-8")
    (run_dir / "RunConfig.json").write_text(_json(run_config), encoding="utf-8")
    (run_dir / "ReproducibilityManifest.json").write_text(_json(manifest), encoding="utf-8")
    (run_dir / "P_vector.json").write_text(_json(p_vector), encoding="utf-8")

    expected = set(OUTPUT_BUNDLE_FILES.values())
    found = {p.name for p in run_dir.iterdir()}
    if found - expected or expected - found:
        raise RuntimeError(f"Invalid first-class bundle keys. extra={sorted(found - expected)} missing={sorted(expected - found)}")
    return run_dir
