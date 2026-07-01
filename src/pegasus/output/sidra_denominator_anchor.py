"""Attach official SIDRA population anchors to canonical run bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pegasus.geo.support import SupportAlignmentError, assert_municipality_year_support_aligned
from pegasus.output.table_io import append_replace_rows, read_rows, write_rows_like
from pegasus.sidra.population_cube.anchor import SidraPopulationAnchor, load_sidra_population_total_anchor


def _remove_by_values(rows: list[dict[str, Any]], column: str, values: set[Any]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get(column) not in values]


def _get_field_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("name") == name:
            return row
    raise KeyError(name)


def _append_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    remove_column: str | None = None,
    remove_values: set[Any] | None = None,
) -> Path:
    path = Path(path)
    incoming = [dict(row) for row in rows]
    existing = read_rows(path) if path.exists() else []
    if remove_column is not None and remove_values is not None:
        existing = _remove_by_values(existing, remove_column, remove_values)
    return write_rows_like(path, existing + incoming)


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


def _population_support(anchor: SidraPopulationAnchor) -> tuple[dict[str, Any], dict[str, Any]]:
    year = int(anchor.period) if str(anchor.period).isdigit() else anchor.period
    if anchor.locality_level == "N3":
        return (
            {
                "support": "uf_year",
                "ufs": [anchor.locality_id],
                "years": [year],
                "source_table_id": anchor.table_id,
                "source_variable_id": anchor.variable_id,
            },
            {"geography": "uf", "time": "year"},
        )
    return (
        {
            "support": "municipality_year",
            "municipalities": [anchor.locality_id],
            "years": [year],
            "source_table_id": anchor.table_id,
            "source_variable_id": anchor.variable_id,
        },
        {"geography": "municipality_ibge_cod7", "time": "year"},
    )


def _population_v_field(anchor: SidraPopulationAnchor, *, facts_path: str | Path) -> dict[str, Any]:
    support, axes = _population_support(anchor)
    return {
        "field_id": f"sidra_population_anchor_{anchor.field_id}",
        "name": "SIDRAPopulationTotalAnchor",
        "kind": "SIDRAPopulationTotalAnchor",
        "carrier": "Population",
        "unit": "persons",
        "aggregation": "additive",
        "role": json.dumps(["population_denominator_seed", "source_field"], sort_keys=True),
        "source": json.dumps(["SIDRA"], sort_keys=True),
        "support_json": json.dumps(support, sort_keys=True),
        "axes_json": json.dumps(axes, sort_keys=True),
        "operator": "official_sidra_anchor",
        "provenance": json.dumps(["SIDRA", "official", "bounded_total_category_anchor"], sort_keys=True),
        "state": "fragile",
        "dashboard_safe": "warning",
        "warnings": json.dumps(["unvalidated_sidra_anchor"], sort_keys=True),
        "lineage_hash": anchor.field_id,
        "registry_hash": anchor.metadata_hash,
        "materialization_state": "materialized",
        "path": str(facts_path),
    }


def _q_row(field_id: str, *, n_events: float, n_denom: float = 0.0, state: str = "fragile") -> dict[str, Any]:
    return {
        "field_id": field_id,
        "n_events": n_events,
        "n_denom": n_denom,
        "n_eff": n_events if n_events else n_denom,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0 if state == "verified" else 0.2,
        "cv": 0.0,
        "moran_i": 0.0,
        "temporal_roughness": 0.0,
        "spatial_entropy": 0.0,
        "provenance_risk": 0.1,
        "race_axis_source": "",
        "race_axis_target": "",
        "missing_race_share": 0.0,
        "emission_prior_strength": 0.0,
        "race_bridge_cv": 0.0,
        "sensitivity_width": 0.0,
        "bridge_mode": "",
        "state": state,
        "dashboard_safe": "warning" if state == "fragile" else "False",
        "warnings": "[]",
        "computed_at": "sidra_anchor_attach",
        "q_schema_version": "1.0",
    }


def _vd_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": row["field_id"],
        "display_name": row["name"],
        "technical_name": row["name"],
        "definition": f"Compiled field {row['name']}.",
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


def _uf_year_support_alignment(
    *,
    anchor: SidraPopulationAnchor,
    numerator_support: dict[str, Any],
    numerator_axes: dict[str, Any],
    denominator_axes: dict[str, Any],
) -> dict[str, Any] | None:
    if anchor.locality_level != "N3" or denominator_axes.get("geography") != "uf":
        return None
    uf = str(anchor.locality_id)
    municipalities = [str(x) for x in numerator_support.get("municipalities", [])]
    outside = sorted(code for code in municipalities if not code.startswith(uf))
    if outside:
        raise ValueError(f"municipalities outside the denominator UF {uf}: {outside}")
    years = [str(x) for x in numerator_support.get("years", [])]
    if str(anchor.period) not in years:
        raise ValueError("time_support_mismatch")
    return {
        "aligned": True,
        "support": "uf_year",
        "uf": uf,
        "municipality_count": len(municipalities),
        "numerator_geography": numerator_axes.get("geography"),
        "denominator_geography": denominator_axes.get("geography"),
    }


def attach_sidra_population_anchor_to_run(*, run_dir: str | Path, sidra_facts_path: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    anchor = load_sidra_population_total_anchor(sidra_facts_path)
    fields = read_rows(run / "V_fields.parquet")
    numerator = _get_field_by_name(fields, "SIMDeathsAll")
    numerator_support = _loads(numerator.get("support_json")) or {}
    numerator_axes = _loads(numerator.get("axes_json")) or {}
    denominator_support, denominator_axes = _population_support(anchor)
    uf_alignment = _uf_year_support_alignment(
        anchor=anchor,
        numerator_support=numerator_support,
        numerator_axes=numerator_axes,
        denominator_axes=denominator_axes,
    )
    if uf_alignment is None:
        alignment = assert_municipality_year_support_aligned(
            numerator_support=numerator_support,
            numerator_axes=numerator_axes,
            denominator_support=denominator_support,
            denominator_axes=denominator_axes,
        ).model()
    else:
        alignment = uf_alignment

    pop = _population_v_field(anchor, facts_path=sidra_facts_path)
    rate_support = {
        "support": "municipality_year",
        "municipalities": alignment.get("common_municipalities_ibge_cod7")
        or alignment.get("denominator_municipalities_ibge_cod7")
        or denominator_support.get("municipalities", []),
        "years": numerator_support.get("years", []),
        "support_alignment": alignment,
    }
    rate = {
        "field_id": "sim_crude_mortality_sidra_official",
        "name": "SIMCrudeMortalitySIDRAOfficial",
        "kind": "rate",
        "carrier": "Deaths/Population",
        "unit": "deaths_per_person",
        "aggregation": "rate",
        "role": json.dumps(["outcome", "rate"], sort_keys=True),
        "source": json.dumps(["SIM-DO", "SIDRA"], sort_keys=True),
        "support_json": json.dumps(rate_support, sort_keys=True),
        "axes_json": json.dumps({"geography": "mun_residence_cod6", "time": "year"}, sort_keys=True),
        "operator": "rate_from_official_sidra_anchor",
        "provenance": json.dumps(["SIM-DO", "SIDRA", "official"], sort_keys=True),
        "state": "fragile",
        "dashboard_safe": "warning",
        "warnings": json.dumps(["unvalidated_sidra_anchor"], sort_keys=True),
        "lineage_hash": f"{numerator['field_id']}::{pop['field_id']}",
        "registry_hash": anchor.metadata_hash,
        "materialization_state": "materialized",
        "path": "",
    }
    edge = {
        "edge_id": "edge_sim_crude_mortality_sidra_official",
        "parent_field_id": pop["field_id"],
        "child_field_id": rate["field_id"],
        "operator": "rate_from_official_sidra_anchor",
        "operator_params_json": "{}",
        "registry_versions_json": "{}",
        "created_at": "sidra_anchor_attach",
    }
    _append_rows(run / "V_fields.parquet", [pop, rate], remove_column="field_id", remove_values={pop["field_id"], rate["field_id"]})
    append_replace_rows(
        run / "Q_tensor.parquet",
        [_q_row(pop["field_id"], n_events=anchor.value), _q_row(rate["field_id"], n_events=float(numerator_support.get("n_events") or 0), n_denom=anchor.value)],
        id_column="field_id",
    )
    append_replace_rows(run / "VariableDictionary.parquet", [_vd_row(pop), _vd_row(rate)], id_column="field_id")
    append_replace_rows(run / "E_DAG.parquet", [edge], id_column="edge_id")
    return {"anchor_field_id": pop["field_id"], "rate_field_id": rate["field_id"], "support_alignment": alignment}


__all__ = [
    "_append_rows",
    "_get_field_by_name",
    "_population_v_field",
    "_remove_by_values",
    "_uf_year_support_alignment",
    "attach_sidra_population_anchor_to_run",
]
