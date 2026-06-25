from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pegasus.output.table_io import append_rows, read_rows, write_rows, write_rows_like

import pyarrow as pa
import polars as pl

from pegasus.core.hashing import sha256_file, sha256_text
from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7
from pegasus.she.maternal_child_linkage import MaternalChildLinkedSummary, summarize_maternal_child_linkage


FIELD_IDS = {
    "SINASCLiveBirthsAll",
    "SINASCLowBirthWeightBirths",
    "SINASCPrematurityBirths",
    "SINASCCongenitalAnomalyBirths",
    "SIMInfantDeathsForSINASCBirths",
    "SIMNeonatalDeathsForSINASCBirths",
    "SIMPostNeonatalDeathsForSINASCBirths",
    "SINASCCrudeBirthRateSIDRAOfficial",
    "SINASCLowBirthWeightPrevalence",
    "SINASCPrematurityPrevalence",
    "SINASCCongenitalAnomalyPrevalence",
    "SIMInfantMortalitySINASCBirths",
    "SIMNeonatalMortalitySINASCBirths",
    "SIMPostNeonatalMortalitySINASCBirths",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _load_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}



def _read_rows(path: Path) -> list[dict[str, Any]]:
    return read_rows(path)

def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    write_rows_like(path, rows)

def _append_rows(path: Path, rows: list[dict[str, Any]], *, remove_column: str, remove_values: set[str]) -> None:
    existing = [row for row in read_rows(path) if str(row.get(remove_column)) not in remove_values]
    write_rows_like(path, existing + list(rows))
def _append_edges(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = _read_rows(path)
    edge_ids = {str(row["edge_id"]) for row in rows}
    child_ids = FIELD_IDS
    kept = [
        row
        for row in existing
        if str(row.get("edge_id")) not in edge_ids and str(row.get("child_field_id")) not in child_ids
    ]
    _write_rows_like(path, kept + rows)


def _field_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("name") == name or row.get("field_id") == name:
            return row
    raise ValueError(f"Field not found in run bundle: {name}")


def _population_value(population_row: dict[str, Any]) -> float:
    support = _load_json(population_row.get("support_json"))
    for key in ["n_denom", "n_eff", "n_events"]:
        value = support.get(key)
        if value is not None:
            return float(value)
    raise ValueError("SIDRA population anchor support does not expose n_denom/n_eff/n_events.")


def _support_alignment(summary: MaternalChildLinkedSummary) -> dict[str, Any]:
    if summary.municipalities_ibge_cod7:
        crosswalk = "datasus_cod6_to_ibge_cod7_explicit_crosswalk"
        denominator_cod7 = summary.municipalities_ibge_cod7
        reason = "municipality_year_exact_after_datasus_cod6_to_ibge_cod7_crosswalk"
    else:
        crosswalk = "native_datasus_cod6_state_panel"
        denominator_cod7 = []
        reason = "state_panel_native_datasus_cod6_support_without_single_municipality_ibge7_filter"

    return {
        "aligned": True,
        "reason": reason,
        "numerator_municipalities_cod6": summary.municipalities_cod6,
        "numerator_municipalities_ibge_cod7": summary.municipalities_ibge_cod7,
        "denominator_municipalities_ibge_cod7": denominator_cod7,
        "numerator_years": summary.years,
        "denominator_years": summary.years,
        "crosswalk": crosswalk,
        "support_kind": "state_panel_cod6_year" if not summary.municipalities_ibge_cod7 else "municipality_year_cod6_cod7",
    }


def _lineage(field_id: str, parents: list[str], operator: str, support: dict[str, Any]) -> str:
    return sha256_text(_json({"field_id": field_id, "parents": parents, "operator": operator, "support": support}))


def _field_row(
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
    parents: list[str],
    warnings: list[str],
    state: str = "quarantined_descriptive",
    dashboard_safe: str = "False",
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
        "provenance": _json(source + ["maternal_child_state_panel_linkage"]),
        "state": state,
        "dashboard_safe": dashboard_safe,
        "warnings": _json(warnings),
        "lineage_hash": _lineage(field_id, parents, operator, support),
        "registry_hash": "maternal_child_linkage_registry_v1",
        "materialization_state": "materialized",
        "path": "Tables/maternal_child_linkage_summary.parquet",
    }


def _support_payload(field: dict[str, Any]) -> dict[str, Any]:
    raw = field.get("support_json")
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _support_cov_s(field: dict[str, Any]) -> float:
    payload = _support_payload(field)
    geography = payload.get("geography")
    if isinstance(geography, dict):
        for key in ("municipality_cod6", "municipality_ibge_cod7"):
            values = geography.get(key)
            if isinstance(values, list):
                return float(len(values))
    values = payload.get("municipalities")
    if isinstance(values, list):
        return float(len(values))
    return 1.0


def _support_cov_t(field: dict[str, Any]) -> float:
    payload = _support_payload(field)
    time = payload.get("time")
    if isinstance(time, dict):
        years = time.get("years")
        if isinstance(years, list):
            return float(len(years))
    years = payload.get("years")
    if isinstance(years, list):
        return float(len(years))
    return 1.0


def _q_row(field: dict[str, Any], *, n_events: float | int | None, n_denom: float | int | None, warnings: list[str]) -> dict[str, Any]:
    n_eff = n_denom if n_denom is not None else n_events
    return {
        "field_id": field["field_id"],
        "n_events": float(n_events) if n_events is not None else None,
        "n_denom": float(n_denom) if n_denom is not None else None,
        "n_eff": float(n_eff) if n_eff is not None else None,
        "cov_S": _support_cov_s(field),
        "cov_T": _support_cov_t(field),
        "missingness": 0.0,
        "zero_inflation": 1.0 if n_events == 0 else 0.0,
        "denom_fragility": 0.0 if n_denom else None,
        "cv": None,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.25,
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
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def _vd_row(field: dict[str, Any], *, definition: str, estimand: str, interpretation_warning: str) -> dict[str, Any]:
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
        "interpretation_warning": interpretation_warning,
        "diagnostic_role": None,
        "topology": None,
        "position": None,
        "icd_group_kind": None,
        "icd_group_id": None,
    }


def _edge(edge_id: str, parent: str, child: str, operator: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": edge_id,
        "parent_field_id": parent,
        "child_field_id": child,
        "operator": operator,
        "operator_params_json": _json(params),
        "registry_versions_json": _json({"maternal_child_linkage": "v1"}),
        "created_at": _now(),
    }


def _warning_rows(summary: MaternalChildLinkedSummary) -> list[dict[str, Any]]:
    return [
        {
            "warning_id": "maternal_child_state_panel_linkage",
            "field_id": "run",
            "severity": "info",
            "message": _json({
                "births_total": summary.births_total,
                "infant_deaths": summary.infant_deaths,
                "neonatal_deaths": summary.neonatal_deaths,
                "postneonatal_deaths": summary.postneonatal_deaths,
                "support_alignment": _support_alignment(summary),
                "source_reality": "materialized_external",
            }),
            "created_at": _now(),
            "inherited_from": None,
        }
    ]


def _build_fields(summary: MaternalChildLinkedSummary, population_row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alignment = _support_alignment(summary)
    support_births = {
        **summary.support_cod6(),
        "municipality_ibge_cod7": summary.municipalities_ibge_cod7,
        "support_alignment": alignment,
    }
    support_pop = {
        "time": {"years": summary.years},
        "geography": {"municipality_ibge_cod7": summary.municipalities_ibge_cod7},
        "n_events": summary.births_total,
        "n_denom": summary.denominator_population,
        "n_eff": summary.denominator_population,
        "support_alignment": alignment,
    }
    axes_common = {
        "time": "year",
        "geography": "mun_residence_cod6",
        "maternal_race_axis": "RACACORMAE_admin",
        "newborn_race_axis": "RACACOR_admin",
        "race_bridge": None,
    }
    axes_rate = {**axes_common, "support_alignment": alignment}
    warnings = ["state_panel_support_aligned_maternal_child", "race_axes_not_bridged"]

    specs_count = [
        ("SINASCLiveBirthsAll", "SINASC Live Births All", summary.births_total, "SINASC", "LiveBirths", "live_birth_count", "identity_count"),
        ("SINASCLowBirthWeightBirths", "SINASC Low Birth Weight Births", summary.low_birth_weight_births, "SINASC", "LiveBirths", "low_birth_weight_count", "indicator_count"),
        ("SINASCPrematurityBirths", "SINASC Prematurity Births", summary.prematurity_births, "SINASC", "LiveBirths", "prematurity_count", "indicator_count"),
        ("SINASCCongenitalAnomalyBirths", "SINASC Congenital Anomaly Births", summary.congenital_anomaly_births, "SINASC", "LiveBirths", "congenital_anomaly_count", "indicator_count"),
        ("SIMInfantDeathsForSINASCBirths", "SIM Infant Deaths for SINASC Birth Denominator", summary.infant_deaths, "SIM-DO", "Deaths", "infant_death_count", "age_days_indicator_count"),
        ("SIMNeonatalDeathsForSINASCBirths", "SIM Neonatal Deaths for SINASC Birth Denominator", summary.neonatal_deaths, "SIM-DO", "Deaths", "neonatal_death_count", "age_days_indicator_count"),
        ("SIMPostNeonatalDeathsForSINASCBirths", "SIM Postneonatal Deaths for SINASC Birth Denominator", summary.postneonatal_deaths, "SIM-DO", "Deaths", "postneonatal_death_count", "age_days_indicator_count"),
    ]
    fields: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    vd_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for field_id, name, count, source, carrier, estimand, operator in specs_count:
        field = _field_row(
            field_id=field_id,
            name=name,
            kind="extensive_measure",
            carrier=carrier,
            unit="count",
            aggregation="additive_count",
            role=["outcome", "demographic"] if source == "SINASC" else ["outcome"],
            source=[source],
            support={**support_births, "n_events": count},
            axes=axes_common,
            operator=operator,
            parents=[],
            warnings=warnings,
        )
        fields.append(field)
        q_rows.append(_q_row(field, n_events=count, n_denom=None, warnings=warnings))
        vd_rows.append(_vd_row(field, definition=f"{name} over the aligned SINASC/SIM smoke support.", estimand=estimand, interpretation_warning="Fixture-derived count; no population or birth denominator applied unless used by a derived rate field."))

    rate_specs = [
        ("SINASCCrudeBirthRateSIDRAOfficial", "SINASC Crude Birth Rate with SIDRA Official Population", summary.births_total, summary.denominator_population, "SINASCLiveBirthsAll", population_row["field_id"], ["SINASC", "SIDRA"], "LiveBirths/Population", "births per person", "crude_birth_rate", support_pop, "ratio_births_population"),
        ("SINASCLowBirthWeightPrevalence", "SINASC Low Birth Weight Prevalence", summary.low_birth_weight_births, summary.births_total, "SINASCLowBirthWeightBirths", "SINASCLiveBirthsAll", ["SINASC"], "LiveBirths/LiveBirths", "proportion", "low_birth_weight_prevalence", support_births, "ratio_indicator_births"),
        ("SINASCPrematurityPrevalence", "SINASC Prematurity Prevalence", summary.prematurity_births, summary.births_total, "SINASCPrematurityBirths", "SINASCLiveBirthsAll", ["SINASC"], "LiveBirths/LiveBirths", "proportion", "prematurity_prevalence", support_births, "ratio_indicator_births"),
        ("SINASCCongenitalAnomalyPrevalence", "SINASC Congenital Anomaly Prevalence", summary.congenital_anomaly_births, summary.births_total, "SINASCCongenitalAnomalyBirths", "SINASCLiveBirthsAll", ["SINASC"], "LiveBirths/LiveBirths", "proportion", "congenital_anomaly_prevalence", support_births, "ratio_indicator_births"),
        ("SIMInfantMortalitySINASCBirths", "SIM Infant Mortality with SINASC Birth Denominator", summary.infant_deaths, summary.births_total, "SIMInfantDeathsForSINASCBirths", "SINASCLiveBirthsAll", ["SIM-DO", "SINASC"], "Deaths/LiveBirths", "deaths per live birth", "infant_mortality", support_births, "ratio_sim_deaths_sinasc_births"),
        ("SIMNeonatalMortalitySINASCBirths", "SIM Neonatal Mortality with SINASC Birth Denominator", summary.neonatal_deaths, summary.births_total, "SIMNeonatalDeathsForSINASCBirths", "SINASCLiveBirthsAll", ["SIM-DO", "SINASC"], "Deaths/LiveBirths", "deaths per live birth", "neonatal_mortality", support_births, "ratio_sim_deaths_sinasc_births"),
        ("SIMPostNeonatalMortalitySINASCBirths", "SIM Postneonatal Mortality with SINASC Birth Denominator", summary.postneonatal_deaths, summary.births_total, "SIMPostNeonatalDeathsForSINASCBirths", "SINASCLiveBirthsAll", ["SIM-DO", "SINASC"], "Deaths/LiveBirths", "deaths per live birth", "postneonatal_mortality", support_births, "ratio_sim_deaths_sinasc_births"),
    ]
    for field_id, name, numerator, denominator, numerator_id, denominator_id, source, carrier, unit, estimand, support, operator in rate_specs:
        field = _field_row(
            field_id=field_id,
            name=name,
            kind="intensive_density",
            carrier=carrier,
            unit=unit,
            aggregation="ratio_recomputed_from_aligned_counts",
            role=["outcome", "demographic"],
            source=source,
            support={**support, "n_events": numerator, "n_denom": denominator, "n_eff": denominator},
            axes={**axes_rate, "numerator_field_id": numerator_id, "denominator_field_id": denominator_id},
            operator=operator,
            parents=[numerator_id, denominator_id],
            warnings=warnings,
        )
        fields.append(field)
        q_rows.append(_q_row(field, n_events=numerator, n_denom=denominator, warnings=warnings))
        vd_rows.append(_vd_row(field, definition=f"{name}: numerator {numerator_id} divided by denominator {denominator_id} after aligned support checks.", estimand=estimand, interpretation_warning="Rate is recomputed from aligned numerator and denominator counts; fixture scale is not inferential."))
        edges.append(_edge(f"edge_{numerator_id}_to_{field_id}", numerator_id, field_id, "ratio_numerator", {"operator": operator}))
        edges.append(_edge(f"edge_{denominator_id}_to_{field_id}", denominator_id, field_id, "ratio_denominator", {"operator": operator}))

    return fields, q_rows, vd_rows, edges


def _write_summary_table(run_dir: Path, summary: MaternalChildLinkedSummary) -> None:
    rates = summary.rates()
    row = {
        "births_total": summary.births_total,
        "low_birth_weight_births": summary.low_birth_weight_births,
        "prematurity_births": summary.prematurity_births,
        "congenital_anomaly_births": summary.congenital_anomaly_births,
        "infant_deaths": summary.infant_deaths,
        "neonatal_deaths": summary.neonatal_deaths,
        "postneonatal_deaths": summary.postneonatal_deaths,
        "denominator_population": summary.denominator_population,
        **rates,
    }
    table_path = run_dir / "Tables" / "maternal_child_linkage_summary.parquet"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(table_path, [row])


def _update_json_outputs(run_dir: Path, *, sinasc_events_path: Path, sim_events_path: Path, summary: MaternalChildLinkedSummary) -> None:
    for name in ["RunConfig.json", "P_vector.json", "ReproducibilityManifest.json"]:
        path = run_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_hashes = payload.setdefault("source_hashes", {})
        source_hashes.setdefault("sinasc_events", sha256_file(sinasc_events_path))
        source_hashes.setdefault("sim_events_for_maternal_child", sha256_file(sim_events_path))
        if name == "RunConfig.json":
            payload["maternal_child_linkage"] = {
                "enabled": True,
                "births_total": summary.births_total,
                "infant_deaths": summary.infant_deaths,
                "neonatal_deaths": summary.neonatal_deaths,
                "postneonatal_deaths": summary.postneonatal_deaths,
                "support_alignment": _support_alignment(summary),
            }
        if name == "P_vector.json":
            systems = set(payload.get("source_systems") or [])
            systems.update(["SINASC", "SIM-DO", "SIDRA"])
            payload["source_systems"] = sorted(systems)
            payload["maternal_child_linkage"] = "enabled"
        if name == "ReproducibilityManifest.json":
            payload.setdefault("maternal_child_linkage", {})
            payload["maternal_child_linkage"].update({
                "enabled": True,
                "summary_table": "Tables/maternal_child_linkage_summary.parquet",
                "fields": sorted(FIELD_IDS),
            })
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def attach_maternal_child_compile_fields(
    *,
    run_dir: str | Path,
    sinasc_events_path: str | Path,
    sim_events_path: str | Path,
    municipality_cod6: str | None,
    datasus_uf_prefix: str,
) -> Path:
    run_dir = Path(run_dir)
    sinasc_events_path = Path(sinasc_events_path)
    sim_events_path = Path(sim_events_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not sinasc_events_path.exists():
        raise FileNotFoundError(f"SINASC events not found: {sinasc_events_path}")
    if not sim_events_path.exists():
        raise FileNotFoundError(f"SIM events not found: {sim_events_path}")

    cod7 = None
    if municipality_cod6 is not None:
        cod7 = datasus_cod6_to_ibge_cod7(municipality_cod6, strict=True)
        if cod7 is None:
            raise ValueError(f"Cannot crosswalk DATASUS municipality cod6 to IBGE/SIDRA cod7: {municipality_cod6}")

    v_rows = _read_rows(run_dir / "V_fields.parquet")
    population_row = _field_by_name(v_rows, "SIDRAPopulationTotalAnchor")
    population_value = _population_value(population_row)
    summary = summarize_maternal_child_linkage(
        sinasc_events_path=sinasc_events_path,
        sim_events_path=sim_events_path,
        municipality_cod6=municipality_cod6,
        municipality_ibge_cod7=cod7,
        datasus_uf_prefix=datasus_uf_prefix,
        denominator_population=population_value,
    )
    if not summary.births_total:
        raise ValueError("Cannot attach maternal-child fields without nonzero live-birth support.")

    if municipality_cod6 is not None:
        if summary.municipalities_ibge_cod7 != [cod7]:
            raise ValueError(
                f"Maternal-child support mismatch: expected cod7={cod7}, observed={summary.municipalities_ibge_cod7}"
            )
    else:
        if len(summary.municipalities_cod6) <= 1:
            raise ValueError(
                "State-level maternal-child attachment requires multi-municipality DATASUS cod6 support; "
                f"observed={summary.municipalities_cod6}"
            )

    fields, q_rows, vd_rows, edges = _build_fields(summary, population_row)
    _append_rows(run_dir / "V_fields.parquet", fields, remove_column="field_id", remove_values=FIELD_IDS)
    _append_rows(run_dir / "Q_tensor.parquet", q_rows, remove_column="field_id", remove_values=FIELD_IDS)
    _append_rows(run_dir / "VariableDictionary.parquet", vd_rows, remove_column="field_id", remove_values=FIELD_IDS)
    _append_edges(run_dir / "E_DAG.parquet", edges)
    _append_rows(run_dir / "Warnings.parquet", _warning_rows(summary), remove_column="warning_id", remove_values={"slice3b_maternal_child_fixture_rates", "slice3b_mortality_birth_denominator_linkage"})
    _write_summary_table(run_dir, summary)
    _update_json_outputs(run_dir, sinasc_events_path=sinasc_events_path, sim_events_path=sim_events_path, summary=summary)
    return run_dir
