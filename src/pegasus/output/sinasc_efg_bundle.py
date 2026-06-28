"""SINASC maternal-child fixture bundle writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.table_io import append_replace_rows, write_rows_like
from pegasus.she.maternal_child import summarize_maternal_child_events


def _field(field_id: str, name: str, *, support: dict[str, Any], kind: str = "maternal_child") -> dict[str, Any]:
    return {
        "field_id": field_id,
        "name": name,
        "kind": kind,
        "carrier": "LiveBirths" if field_id == "sinasc_births_all" else "LiveBirths/LiveBirths",
        "unit": "births" if field_id == "sinasc_births_all" else "share",
        "aggregation": "additive" if field_id == "sinasc_births_all" else "rate",
        "role": json.dumps(["maternal_child", "outcome"], sort_keys=True),
        "source": json.dumps(["SINASC"], sort_keys=True),
        "support_json": json.dumps(support, sort_keys=True),
        "axes_json": json.dumps({"geography": "mun_residence_cod6", "time": "year"}, sort_keys=True),
        "operator": "maternal_child_primitive_summary",
        "provenance": json.dumps(["SINASC", "source_normalized"], sort_keys=True),
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": json.dumps(["blocked_missing_population_denominator_anchor"], sort_keys=True),
        "lineage_hash": field_id,
        "registry_hash": "maternal_child_fixture",
        "materialization_state": "materialized",
        "path": "",
    }


def _q(field_id: str, n: float) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "n_events": n,
        "n_denom": n,
        "n_eff": n,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 1.0,
        "cv": 0.0,
        "moran_i": 0.0,
        "temporal_roughness": 0.0,
        "spatial_entropy": 0.0,
        "provenance_risk": 0.0,
        "race_axis_source": "",
        "race_axis_target": "",
        "missing_race_share": 0.0,
        "emission_prior_strength": 0.0,
        "race_bridge_cv": 0.0,
        "sensitivity_width": 0.0,
        "bridge_mode": "",
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": json.dumps(["blocked_missing_population_denominator_anchor"]),
        "computed_at": "sinasc_fixture_bundle",
        "q_schema_version": "1.0",
    }


def _vd(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": row["field_id"],
        "display_name": row["name"],
        "technical_name": row["name"],
        "definition": f"SINASC maternal-child fixture field {row['name']}.",
        "estimand_label": row["kind"],
        "source_systems": row["source"],
        "carrier": row["carrier"],
        "unit": row["unit"],
        "support_description": row["support_json"],
        "axis_description": row["axes_json"],
        "provenance_description": row["provenance"],
        "state": row["state"],
        "dashboard_safe": row["dashboard_safe"],
        "interpretation_warning": row["warnings"],
    }


def _failed(reason: str) -> dict[str, Any]:
    return {
        "failed_branch_id": reason,
        "attempted_operator": "maternal_child_rate_or_mortality",
        "parent_field_ids": "[]",
        "failure_stage": "precondition",
        "failed_terms": json.dumps([reason]),
        "reason": reason,
        "warnings": json.dumps([reason]),
        "created_at": "sinasc_fixture_bundle",
    }


def write_sinasc_fixture_efg_bundle(*, sinasc_events_path: str | Path, run_dir: str | Path) -> Path:
    run = create_empty_output_bundle(run_dir)
    summary = summarize_maternal_child_events(sinasc_events_path, datasus_uf_prefix="27")
    support = {
        "support": "municipality_year",
        "municipalities": summary.municipalities_cod6,
        "years": summary.years,
        "n_events": summary.births_total,
    }
    fields = [
        _field("sinasc_births_all", "SINASCBirthsAll", support=support, kind="event_count"),
        _field("sinasc_low_birth_weight_prevalence", "SINASCLowBirthWeightPrevalence", support=support),
        _field("sinasc_congenital_anomaly_prevalence", "SINASCAnomalyPrevalence", support=support),
        _field("sinasc_low_apgar5_prevalence", "SINASCLowApgar5Prevalence", support=support),
        _field("sinasc_insufficient_prenatal_share", "SINASCInsufficientPrenatalShare", support=support),
    ]
    edges = [
        {
            "edge_id": f"edge_{row['field_id']}",
            "parent_field_id": "sinasc_births_all",
            "child_field_id": row["field_id"],
            "operator": row["operator"],
            "operator_params_json": "{}",
            "registry_versions_json": "{}",
            "created_at": "sinasc_fixture_bundle",
        }
        for row in fields
        if row["field_id"] != "sinasc_births_all"
    ]
    append_replace_rows(run / "V_fields.parquet", fields, id_column="field_id")
    append_replace_rows(run / "Q_tensor.parquet", [_q(row["field_id"], float(summary.births_total)) for row in fields], id_column="field_id")
    append_replace_rows(run / "VariableDictionary.parquet", [_vd(row) for row in fields], id_column="field_id")
    write_rows_like(run / "E_DAG.parquet", edges)
    append_replace_rows(
        run / "FailedBranches.parquet",
        [
            _failed("blocked_missing_population_denominator_anchor"),
            _failed("blocked_missing_sim_death_numerator_linkage"),
            _failed("blocked_missing_sim_neonatal_death_numerator_linkage"),
            _failed("blocked_missing_sim_postneonatal_death_numerator_linkage"),
        ],
        id_column="failed_branch_id",
    )
    return run


__all__ = ["write_sinasc_fixture_efg_bundle"]
