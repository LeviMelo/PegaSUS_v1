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

from pegasus.core.hashing import sha256_file, sha256_text
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.she.maternal_child import MaternalChildSummary, summarize_maternal_child_events


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _empty_like(path: Path) -> None:
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 0), path)


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pq.read_table(path).schema
    fixed = [{field.name: row.get(field.name) for field in schema} for row in rows]
    table = pa.Table.from_pylist(fixed, schema=schema) if fixed else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, path)


def _field_row(
    *,
    field_id: str,
    name: str,
    kind: str,
    unit: str,
    aggregation: str,
    role: list[str],
    support: dict[str, Any],
    axes: dict[str, Any],
    operator: str | None,
    provenance: list[str],
    warnings: list[str],
    state: str = "verified",
    path: str | None = None,
) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "name": name,
        "kind": kind,
        "carrier": "LiveBirths",
        "unit": unit,
        "aggregation": aggregation,
        "role": _json(role),
        "source": _json(["SINASC"]),
        "support_json": _json(support),
        "axes_json": _json(axes),
        "operator": operator,
        "provenance": _json(provenance),
        "state": state,
        "dashboard_safe": "True" if state == "verified" else "False",
        "warnings": _json(warnings),
        "lineage_hash": sha256_text(_json({"field_id": field_id, "operator": operator, "support": support})),
        "registry_hash": "sinasc_maternal_child_registry_v1",
        "materialization_state": "materialized",
        "path": path,
    }


def _q_row(*, field: dict[str, Any], n_events: int | float | None, n_denom: int | float | None, warnings: list[str]) -> dict[str, Any]:
    n_eff = n_denom if n_denom is not None else n_events
    missingness = 0.0
    if "decoder_missingness" in field:
        missingness = float(field["decoder_missingness"])
    return {
        "field_id": field["field_id"],
        "n_events": float(n_events) if n_events is not None else None,
        "n_denom": float(n_denom) if n_denom is not None else None,
        "n_eff": float(n_eff) if n_eff is not None else None,
        "cov_S": None,
        "cov_T": None,
        "missingness": missingness,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0 if n_denom else None,
        "provenance_risk": 0.15,
        "state": field["state"],
        "dashboard_safe": field["dashboard_safe"],
        "warnings": _json(warnings),
        "computed_at": _now(),
        "q_schema_version": "v1",
    }


def _vd_row(*, field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "display_name": field["name"],
        "technical_name": field["field_id"],
        "definition": definition,
        "estimand_label": estimand,
        "source_systems": _json(["SINASC"]),
        "carrier": field["carrier"],
        "unit": field["unit"],
        "support_description": "SINASC live-birth event support by birth year and municipality of residence.",
        "axis_description": "Maternal and newborn administrative race axes are preserved separately; no race bridge or IBGE denominator replacement is applied.",
        "provenance_description": "Derived from normalized SINASC DN records with explicit decoder states.",
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
        "operator_params_json": _json(params),
        "registry_versions_json": _json({"maternal_child": "v1"}),
        "created_at": _now(),
    }


def _failed_branch_rows(summary: MaternalChildSummary) -> list[dict[str, Any]]:
    parent_ids = _json(["sinasc_births_all"])
    return [
        {
            "branch_id": "sinasc_crude_birth_rate_requires_population_denominator",
            "parent_field_ids": parent_ids,
            "operator": "birth_rate",
            "reason": "blocked_missing_population_denominator_anchor",
            "severity": "blocked",
            "created_at": _now(),
            "details_json": _json({"births_total": summary.births_total, "required_source": "SIDRA population denominator"}),
        },
        {
            "branch_id": "sinasc_infant_mortality_requires_sim_linkage",
            "parent_field_ids": parent_ids,
            "operator": "infant_mortality_rate",
            "reason": "blocked_missing_sim_death_numerator_linkage",
            "severity": "blocked",
            "created_at": _now(),
            "details_json": _json({"births_total": summary.births_total, "required_source": "SIM-DO"}),
        },
        {
            "branch_id": "sinasc_neonatal_mortality_requires_sim_linkage",
            "parent_field_ids": parent_ids,
            "operator": "neonatal_mortality_rate",
            "reason": "blocked_missing_sim_neonatal_death_numerator_linkage",
            "severity": "blocked",
            "created_at": _now(),
            "details_json": _json({"births_total": summary.births_total, "required_source": "SIM-DO"}),
        },
        {
            "branch_id": "sinasc_postneonatal_mortality_requires_sim_linkage",
            "parent_field_ids": parent_ids,
            "operator": "postneonatal_mortality_rate",
            "reason": "blocked_missing_sim_postneonatal_death_numerator_linkage",
            "severity": "blocked",
            "created_at": _now(),
            "details_json": _json({"births_total": summary.births_total, "required_source": "SIM-DO"}),
        },
    ]


def _warning_rows(summary: MaternalChildSummary) -> list[dict[str, Any]]:
    return [
        {
            "warning_id": "sinasc_race_axes_preserved_separately",
            "field_id": "run",
            "severity": "info",
            "message": "Maternal and newborn administrative race axes remain separate; no IBGE self-declared denominator replacement or Bridge_R has been applied.",
            "created_at": _now(),
            "inherited_from": None,
        },
        {
            "warning_id": "sinasc_mortality_fields_blocked_without_sim_linkage",
            "field_id": "run",
            "severity": "blocked",
            "message": "Infant, neonatal, and postneonatal mortality require SIM numerator linkage and are emitted as FailedBranches in this slice.",
            "created_at": _now(),
            "inherited_from": None,
        },
        {
            "warning_id": "sinasc_birth_rate_blocked_without_population_denominator",
            "field_id": "run",
            "severity": "blocked",
            "message": "Crude birth rate requires a legal population denominator anchor and is emitted as a FailedBranch in this foundation slice.",
            "created_at": _now(),
            "inherited_from": None,
        },
        {
            "warning_id": "sinasc_decoder_state_counts",
            "field_id": "run",
            "severity": "info",
            "message": _json({
                "birth_weight_missing_or_invalid": summary.birth_weight_missing_or_invalid,
                "gestational_age_missing_or_invalid": summary.gestational_age_missing_or_invalid,
                "apgar_missing_or_invalid": summary.apgar_missing_or_invalid,
                "race_missing_or_ignored": summary.race_missing_or_ignored,
            }),
            "created_at": _now(),
            "inherited_from": None,
        },
    ]


def _count_specs(summary: MaternalChildSummary) -> list[tuple[str, str, int, str, str]]:
    return [
        ("sinasc_births_all", "SINASC Live Births", summary.births_total, "Live-birth count from normalized SINASC records.", "live_birth_count"),
        ("sinasc_low_birth_weight_births", "Low Birth Weight Births", summary.low_birth_weight_births, "Births with valid birth weight below 2500 g.", "low_birth_weight_count"),
        ("sinasc_prematurity_births", "Prematurity Births", summary.prematurity_births, "Births with valid gestational age below 37 completed weeks.", "prematurity_count"),
        ("sinasc_cesarean_births", "Cesarean Births", summary.cesarean_births, "Births with SINASC delivery mode decoded as cesarean.", "cesarean_count"),
        ("sinasc_congenital_anomaly_births", "Congenital Anomaly Births", summary.congenital_anomaly_births, "Births with a valid congenital anomaly flag or Q-prefix anomaly code.", "congenital_anomaly_count"),
        ("sinasc_low_apgar5_births", "Low 5-Minute APGAR Births", summary.low_apgar5_births, "Births with valid 5-minute APGAR below 7.", "low_apgar5_count"),
        ("sinasc_adolescent_mother_births", "Adolescent Mother Births", summary.adolescent_mother_births, "Births where maternal age is valid and below 20 years.", "adolescent_mother_count"),
        ("sinasc_advanced_maternal_age_births", "Advanced Maternal Age Births", summary.advanced_maternal_age_births, "Births where maternal age is valid and at least 35 years.", "advanced_maternal_age_count"),
        ("sinasc_insufficient_prenatal_births", "Insufficient Prenatal Consultation Births", summary.insufficient_prenatal_births, "Births with a valid prenatal consultation count below 7.", "insufficient_prenatal_count"),
    ]


def _rate_specs(summary: MaternalChildSummary) -> list[tuple[str, str, int, str, str, str]]:
    return [
        ("sinasc_low_birth_weight_prevalence", "Low Birth Weight Prevalence", summary.low_birth_weight_births, "sinasc_low_birth_weight_births", "Low birth weight births divided by all live births.", "low_birth_weight_prevalence"),
        ("sinasc_prematurity_prevalence", "Prematurity Prevalence", summary.prematurity_births, "sinasc_prematurity_births", "Prematurity births divided by all live births.", "prematurity_prevalence"),
        ("sinasc_cesarean_prevalence", "Cesarean Birth Prevalence", summary.cesarean_births, "sinasc_cesarean_births", "Cesarean births divided by all live births.", "cesarean_prevalence"),
        ("sinasc_congenital_anomaly_prevalence", "Congenital Anomaly Prevalence", summary.congenital_anomaly_births, "sinasc_congenital_anomaly_births", "Congenital anomaly births divided by all live births.", "congenital_anomaly_prevalence"),
        ("sinasc_low_apgar5_prevalence", "Low 5-Minute APGAR Prevalence", summary.low_apgar5_births, "sinasc_low_apgar5_births", "Low 5-minute APGAR births divided by all live births.", "low_apgar5_prevalence"),
        ("sinasc_adolescent_mother_share", "Adolescent Mother Share", summary.adolescent_mother_births, "sinasc_adolescent_mother_births", "Births to mothers under 20 divided by all live births.", "adolescent_mother_share"),
        ("sinasc_advanced_maternal_age_share", "Advanced Maternal Age Share", summary.advanced_maternal_age_births, "sinasc_advanced_maternal_age_births", "Births to mothers at least 35 divided by all live births.", "advanced_maternal_age_share"),
        ("sinasc_insufficient_prenatal_share", "Insufficient Prenatal Consultation Share", summary.insufficient_prenatal_births, "sinasc_insufficient_prenatal_births", "Births with fewer than 7 prenatal consultations divided by all live births.", "insufficient_prenatal_share"),
    ]


def build_sinasc_fixture_rows(summary: MaternalChildSummary, *, events_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    support = summary.support()
    axes = {
        "time_axis": "birth_year",
        "geo_axis": "mun_residence_cod6",
        "event_axis": "SINASC_DN",
        "maternal_race_axis": "RACACORMAE_admin",
        "newborn_race_axis": "RACACOR_admin",
        "anomaly_axis": "IDANOMAL_plus_CODANOMAL",
    }
    path = "Tables/sinasc_maternal_child_summary.parquet"
    shared_warning = ["sinasc_fixture_support", "race_axes_not_bridged", "mortality_linkage_blocked"]
    fields: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    vd_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for field_id, name, value, definition, estimand in _count_specs(summary):
        field = _field_row(
            field_id=field_id,
            name=name,
            kind="extensive_measure",
            unit="births",
            aggregation="additive_count",
            role=["exposure", "outcome_candidate", "maternal_child"],
            support={**support, "n_events": value},
            axes=axes,
            operator="sinasc_event_count" if field_id != "sinasc_births_all" else "identity_count",
            provenance=["SINASC", str(events_path)],
            warnings=shared_warning,
            path=path,
        )
        fields.append(field)
        q_rows.append(_q_row(field=field, n_events=value, n_denom=None, warnings=shared_warning))
        vd_rows.append(_vd_row(field=field, definition=definition, estimand=estimand, warning="Fixture-level maternal-child field; mortality outputs require SIM linkage."))

    for field_id, name, numerator_value, numerator_id, definition, estimand in _rate_specs(summary):
        field = _field_row(
            field_id=field_id,
            name=name,
            kind="intensive_density",
            unit="proportion",
            aggregation="ratio_recomputed_from_counts",
            role=["outcome_candidate", "descriptive_rate", "maternal_child"],
            support={**support, "n_events": numerator_value, "n_denom": summary.births_total},
            axes={**axes, "denominator_field_id": "sinasc_births_all", "numerator_field_id": numerator_id},
            operator="ratio_from_aligned_sinasc_counts",
            provenance=["SINASC", str(events_path)],
            warnings=shared_warning,
            path=path,
        )
        fields.append(field)
        q_rows.append(_q_row(field=field, n_events=numerator_value, n_denom=summary.births_total, warnings=shared_warning))
        vd_rows.append(_vd_row(field=field, definition=definition, estimand=estimand, warning="Ratio is recomputed from aligned SINASC count numerator and live-birth denominator."))
        edges.append(_edge(f"edge_{numerator_id}_to_{field_id}", numerator_id, field_id, "ratio_numerator", {"law": "count_over_births"}))
        edges.append(_edge(f"edge_sinasc_births_all_to_{field_id}", "sinasc_births_all", field_id, "ratio_denominator", {"law": "count_over_births"}))

    return fields, q_rows, vd_rows, edges


def _summary_table(summary: MaternalChildSummary) -> list[dict[str, Any]]:
    return [{
        "births_total": summary.births_total,
        "low_birth_weight_births": summary.low_birth_weight_births,
        "prematurity_births": summary.prematurity_births,
        "cesarean_births": summary.cesarean_births,
        "congenital_anomaly_births": summary.congenital_anomaly_births,
        "low_apgar5_births": summary.low_apgar5_births,
        "adolescent_mother_births": summary.adolescent_mother_births,
        "advanced_maternal_age_births": summary.advanced_maternal_age_births,
        "insufficient_prenatal_births": summary.insufficient_prenatal_births,
        **summary.rates(),
    }]


def write_sinasc_fixture_efg_bundle(
    *,
    sinasc_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
    datasus_uf_prefix: str = "27",
    uf: str | None = None,
) -> Path:
    events_path = Path(sinasc_events_path)
    run_dir = create_empty_output_bundle(run_dir)
    summary = summarize_maternal_child_events(
        events_path,
        municipality_cod6=municipality_cod6,
        datasus_uf_prefix=datasus_uf_prefix,
    )
    fields, q_rows, vd_rows, edges = build_sinasc_fixture_rows(summary, events_path=events_path)

    pl.DataFrame(_summary_table(summary)).write_parquet(run_dir / "Tables" / "sinasc_maternal_child_summary.parquet")

    _write_rows_like(run_dir / "V_fields.parquet", fields)
    _write_rows_like(run_dir / "Q_tensor.parquet", q_rows)
    _write_rows_like(run_dir / "VariableDictionary.parquet", vd_rows)
    _write_rows_like(run_dir / "E_DAG.parquet", edges)
    _write_rows_like(run_dir / "Warnings.parquet", _warning_rows(summary))
    _write_rows_like(run_dir / "FailedBranches.parquet", _failed_branch_rows(summary))

    for name in ["ModelAssociations.parquet", "ResidualAssociations.parquet", "Hypotheses.parquet", "QuarantinedFields.parquet", "ForcedFields.parquet"]:
        _empty_like(run_dir / name)

    user_intent = {
        "geography": {"level": "municipality", "codes": summary.municipalities_cod6, "uf": [uf] if uf else []},
        "time": {"start_year": min(summary.years) if summary.years else 2022, "end_year": max(summary.years) if summary.years else 2022},
        "health_seeds": ["maternal_child"],
        "mandatory_fields": [row["field_id"] for row in fields],
        "system_weights": {"SINASC": 1.0},
        "context_policy": [],
        "budget": "fast",
        "geo_mode": "native",
        "force_selectors": [],
        "exclude_systems": [],
        "execution_scale": "smoke",
        "race_tensor_mode": "decoupled",
        "population_mode": "blocked_missing",
    }
    run_config = {
        "run_id": Path(run_dir).name,
        "workflow": "sinasc_fixture_maternal_child_efg",
        "source_hashes": {"sinasc_events": sha256_file(events_path)},
        "registry_hashes": {"maternal_child": "sinasc_maternal_child_registry_v1"},
        "municipality_filter_cod6": municipality_cod6,
        "created_at": _now(),
    }
    stage_status = {stage: "skipped" for stage in COMPILE_TELEMETRY_STAGES}
    stage_wall_seconds = {stage: 0.0 for stage in COMPILE_TELEMETRY_STAGES}
    for stage in ["datasus_normalize", "she_build", "efg_build", "q_tensor", "output_serialization", "output_validation"]:
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
            "population_solver": "No population denominator tensor invoked in Slice 3A SINASC fixture path.",
            "stdfm": "ST-DFM scaffold remains blocked for Slice 3A SINASC fixture path.",
            "pirs_model": "PIRS model fitting is outside Slice 3A.",
            "pirs_hsic": "HSIC scanning is outside Slice 3A.",
        },
        "resource_summary": {
            "peak_rss_mb": None,
            "peak_vram_mb": None,
            "duckdb_temp_bytes": None,
            "rows_read": {"sinasc_events": summary.births_total},
            "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows), "FailedBranches": 4},
            "parquet_bytes_written": 0,
        },
    }
    manifest = {
        "run_id": Path(run_dir).name,
        "created_at": _now(),
        "workflow": "sinasc_fixture_maternal_child_efg",
        "source_hashes": run_config["source_hashes"],
        "registry_hashes": run_config["registry_hashes"],
        "telemetry": telemetry,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows), "FailedBranches": 4},
    }
    p_vector = {
        "source_systems": ["SINASC"],
        "source_hashes": run_config["source_hashes"],
        "registry_hashes": run_config["registry_hashes"],
        "field_count": len(fields),
        "blocked_outputs": ["crude_birth_rate", "infant_mortality", "neonatal_mortality", "postneonatal_mortality"],
        "provenance": {row["field_id"]: ["SINASC", "normalized_dn_fixture", "maternal_child_field"] for row in fields},
    }
    (run_dir / "UserIntent.json").write_text(_json(user_intent), encoding="utf-8")
    (run_dir / "RunConfig.json").write_text(_json(run_config), encoding="utf-8")
    (run_dir / "ReproducibilityManifest.json").write_text(_json(manifest), encoding="utf-8")
    (run_dir / "P_vector.json").write_text(_json(p_vector), encoding="utf-8")

    expected = set(OUTPUT_BUNDLE_FILES.values())
    found = {p.name for p in run_dir.iterdir()}
    extra = found - expected
    missing = expected - found
    if extra or missing:
        raise RuntimeError(f"Invalid first-class bundle keys. extra={sorted(extra)} missing={sorted(missing)}")
    return run_dir
