from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import sha256_file, sha256_text
from pegasus.efg.race_bridge import (
    ADMIN_RACE_LABELS,
    RaceBridgePosterior,
    fixedc_dynamic_weight_bridge,
    load_race_bridge_prior,
    summarize_sim_admin_race_counts,
)
from pegasus.output.validate import validate_output_bundle

RACE_BRIDGE_FIELD_IDS = {
    *(f"SIMRaceAdminRawCount_{code}" for code in ADMIN_RACE_LABELS),
    "SIMRaceBridgeMissingRaceObserver",
    *(f"SIMRaceBridgePosteriorCount_{target}" for target in ["branca", "preta", "amarela", "parda", "indigena"]),
}
RACE_BRIDGE_WARNING_IDS = {
    "race_bridge_admin_axis_preserved",
    "race_bridge_bayesian_ecological_warning",
    "race_bridge_missing_preserved",
    "race_bridge_sensitivity_metadata",
    "race_bridge_posterior_sensitivity",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _append_rows(path: Path, rows: list[dict[str, Any]], *, id_column: str | None = None, remove_values: set[str] | None = None) -> None:
    existing = pl.read_parquet(path)
    if id_column and id_column in existing.columns:
        ids = set(remove_values or set())
        ids.update(str(row[id_column]) for row in rows if row.get(id_column) is not None)
        if ids:
            existing = existing.filter(~pl.col(id_column).cast(pl.Utf8).is_in(sorted(ids)))
    if not rows:
        existing.write_parquet(path)
        return
    new = pl.DataFrame(rows)
    for column, dtype in existing.schema.items():
        if column not in new.columns:
            new = new.with_columns(pl.lit(None).cast(dtype).alias(column))
        else:
            new = new.with_columns(pl.col(column).cast(dtype, strict=False))
    for column in new.columns:
        if column not in existing.columns:
            new = new.drop(column)
    new = new.select(existing.columns)
    pl.concat([existing, new], how="vertical").write_parquet(path)


def _field_row(
    *,
    field_id: str,
    name: str,
    kind: str,
    carrier: str,
    unit: str,
    aggregation: str,
    support: dict[str, Any],
    axes: dict[str, Any],
    operator: str,
    provenance: list[str],
    warnings: list[str],
    state: str,
    dashboard_safe: bool,
    path: str | None = None,
) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "name": name,
        "kind": kind,
        "carrier": carrier,
        "unit": unit,
        "aggregation": aggregation,
        "role": _json(["observer", "demographic", "race_bridge"]),
        "source": _json(["SIM-DO", "Bridge_R"]),
        "support_json": _json(support),
        "axes_json": _json(axes),
        "operator": operator,
        "provenance": _json(provenance),
        "state": state,
        "dashboard_safe": "True" if dashboard_safe else "False",
        "warnings": _json(warnings),
        "lineage_hash": sha256_text(_json({"field_id": field_id, "support": support, "axes": axes, "operator": operator})),
        "registry_hash": axes.get("prior_hash") or "race_bridge_registry_unset",
        "materialization_state": "materialized",
        "path": path,
    }


def _q_row(
    *,
    field: dict[str, Any],
    n_events: float | int | None,
    warnings: list[str],
    missingness: float,
    denom_fragility: float,
    race_axis_source: str | None = None,
    race_axis_target: str | None = None,
    missing_race_share: float | None = None,
    race_bridge_cv: float | None = None,
    sensitivity_width: float | None = None,
    bridge_mode: str | None = None,
) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "n_events": float(n_events) if n_events is not None else None,
        "n_denom": None,
        "n_eff": float(n_events) if n_events is not None else None,
        "cov_S": None,
        "cov_T": None,
        "missingness": float(missingness),
        "zero_inflation": 0.0,
        "denom_fragility": float(denom_fragility),
        "cv": race_bridge_cv,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.65 if field["state"] != "verified" else 0.25,
        "race_axis_source": race_axis_source,
        "race_axis_target": race_axis_target,
        "missing_race_share": missing_race_share,
        "emission_prior_strength": None,
        "race_bridge_cv": race_bridge_cv,
        "sensitivity_width": sensitivity_width,
        "bridge_mode": bridge_mode,
        "state": field["state"],
        "dashboard_safe": field["dashboard_safe"],
        "warnings": _json(warnings),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def _vd_row(*, field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "display_name": field["name"],
        "technical_name": field["field_id"],
        "definition": definition,
        "estimand_label": estimand,
        "source_systems": _json(["SIM-DO", "Bridge_R"]),
        "carrier": field["carrier"],
        "unit": field["unit"],
        "support_description": "SIM death-event support filtered by municipality/year; raw administrative race preserved separately from bridge posterior target axis.",
        "axis_description": "Bridge_R maps SIM administrative race/color observer counts to a target self-declared race axis under a validated fixed-C emission prior.",
        "provenance_description": "Derived from normalized SIM events and race bridge prior; posterior counts are sensitivity-bounded observer fields.",
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
        "registry_versions_json": _json({"race_bridge": params.get("emission_matrix_registry_version")}),
        "created_at": _now(),
    }


def _warning_rows(posterior: RaceBridgePosterior) -> list[dict[str, Any]]:
    return [
        {
            "warning_id": "race_bridge_admin_axis_preserved",
            "field_id": "run",
            "severity": "warning",
            "message": "Raw SIM administrative race/color counts are preserved as raw declaration-process counts; they are not overwritten by IBGE self-declared race.",
            "created_at": _now(),
            "inherited_from": None,
        },
        {
            "warning_id": "race_bridge_bayesian_ecological_warning",
            "field_id": "run",
            "severity": "downgrade",
            "message": "Posterior race counts are bridge-derived observer fields, not direct self-declared measurements; Bridge_R uses a fixed-C emission prior and sensitivity bounds.",
            "created_at": _now(),
            "inherited_from": None,
        },
        {
            "warning_id": "race_bridge_missing_preserved",
            "field_id": "SIMRaceBridgeMissingRaceObserver",
            "severity": "warning",
            "message": f"Missing/invalid administrative race is preserved as an observer count: n={posterior.missing_count}, share={posterior.missing_share:.6f}.",
            "created_at": _now(),
            "inherited_from": None,
        },
        {
            "warning_id": "race_bridge_sensitivity_metadata",
            "field_id": "run",
            "severity": "downgrade",
            "message": f"Bridge_R posterior fields carry sensitivity interval width={posterior.sensitivity_width:.6f}; dashboard safety remains downgraded unless calibration later justifies promotion.",
            "created_at": _now(),
            "inherited_from": None,
        },
    ]


def _build_rows(posterior: RaceBridgePosterior, *, sim_events_path: Path, bridge_prior_path: Path) -> dict[str, list[dict[str, Any]]]:
    support = {
        **posterior.support,
        "n_missing_race": posterior.missing_count,
        "missing_race_share": posterior.missing_share,
    }
    metadata = posterior.metadata()
    common_warnings = [
        "race_bridge_admin_axis_preserved",
        "race_bridge_bayesian_ecological_warning",
        "race_bridge_sensitivity_metadata",
    ]
    fields: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    vd_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    table_path = "Tables/race_bridge_summary.parquet"

    for code, label in ADMIN_RACE_LABELS.items():
        field = _field_row(
            field_id=f"SIMRaceAdminRawCount_{code}",
            name=f"SIM Raw Administrative Race Count {label}",
            kind="extensive_measure",
            carrier="death_event",
            unit="deaths",
            aggregation="additive_count",
            support={**support, "race_color_admin": code, "n_events": posterior.raw_admin_counts.get(code, 0)},
            axes={"race_axis": posterior.prior.source_axis, "race_admin_code": code, "race_admin_label": label, "raw_admin_preserved": True},
            operator="raw_admin_race_count",
            provenance=["SIM-DO", str(sim_events_path)],
            warnings=["race_bridge_admin_axis_preserved"],
            state="verified",
            dashboard_safe=True,
            path=table_path,
        )
        fields.append(field)
        q_rows.append(_q_row(
            field=field,
            n_events=posterior.raw_admin_counts.get(code, 0),
            warnings=["race_bridge_admin_axis_preserved"],
            missingness=0.0,
            denom_fragility=0.0,
            race_axis_source=posterior.prior.source_axis,
        ))
        vd_rows.append(_vd_row(field=field, definition="Raw SIM administrative race/color death count preserved before Bridge_R.", estimand="raw_administrative_race_count", warning="Raw administrative race axis; not equivalent to IBGE self-declared race."))

    missing_field = _field_row(
        field_id="SIMRaceBridgeMissingRaceObserver",
        name="SIM Missing Administrative Race Observer",
        kind="observer_proxy",
        carrier="death_event",
        unit="deaths",
        aggregation="missingness_count",
        support={**support, "n_events": posterior.missing_count},
        axes={"race_axis": posterior.prior.source_axis, "missing_category_preserved": True, **metadata},
        operator="missing_race_observer",
        provenance=["SIM-DO", str(sim_events_path)],
        warnings=common_warnings,
        state="fragile" if posterior.missing_count else "verified",
        dashboard_safe=False if posterior.missing_count else True,
        path=table_path,
    )
    fields.append(missing_field)
    q_rows.append(_q_row(
        field=missing_field,
        n_events=posterior.missing_count,
        warnings=common_warnings,
        missingness=posterior.missing_share,
        denom_fragility=posterior.sensitivity_width,
        race_axis_source=posterior.prior.source_axis,
        missing_race_share=posterior.missing_share,
        race_bridge_cv=posterior.race_bridge_cv,
        sensitivity_width=posterior.sensitivity_width,
        bridge_mode=posterior.prior.mode,
    ))
    vd_rows.append(_vd_row(field=missing_field, definition="Observer field for SIM records with missing, ignored, sentinel, or invalid administrative race/color.", estimand="missing_race_observer", warning="Missing race is preserved and is not imputed into posterior target categories."))

    for target, value in posterior.posterior_counts.items():
        field_id = f"SIMRaceBridgePosteriorCount_{target}"
        axes = {
            "race_axis": posterior.prior.target_axis,
            "race_target_category": target,
            "raw_admin_fields": [f"SIMRaceAdminRawCount_{code}" for code in posterior.raw_admin_counts],
            **metadata,
            "lower_count": posterior.lower_counts[target],
            "upper_count": posterior.upper_counts[target],
        }
        state = "fragile" if posterior.sensitivity_width > 0.05 or posterior.missing_share > 0 else "verified"
        field = _field_row(
            field_id=field_id,
            name=f"SIM Race Bridge Posterior Count {target}",
            kind="bridge_module",
            carrier="death_event",
            unit="deaths_posterior",
            aggregation="posterior_count_from_fixedC",
            support={**support, "n_events": value},
            axes=axes,
            operator="Bridge_R_fixedC_dynamic_weight",
            provenance=["SIM-DO", "Bridge_R", str(sim_events_path), str(bridge_prior_path)],
            warnings=common_warnings,
            state=state,
            dashboard_safe=False if state != "verified" else True,
            path=table_path,
        )
        fields.append(field)
        q_rows.append(_q_row(
            field=field,
            n_events=value,
            warnings=common_warnings,
            missingness=posterior.missing_share,
            denom_fragility=posterior.sensitivity_width,
            race_axis_source=posterior.prior.source_axis,
            race_axis_target=posterior.prior.target_axis,
            missing_race_share=posterior.missing_share,
            race_bridge_cv=posterior.race_bridge_cv,
            sensitivity_width=posterior.sensitivity_width,
            bridge_mode=posterior.prior.mode,
        ))
        vd_rows.append(_vd_row(field=field, definition="Bridge_R posterior target-axis death count from raw SIM administrative race counts and a validated fixed-C emission prior.", estimand="race_bridge_posterior_count", warning="Bridge-derived observer field with sensitivity interval; not a direct measurement."))
        for source_code in posterior.raw_admin_counts:
            edges.append(_edge(
                f"edge_SIMRaceAdminRawCount_{source_code}_to_{field_id}",
                f"SIMRaceAdminRawCount_{source_code}",
                field_id,
                "Bridge_R_fixedC_dynamic_weight",
                metadata,
            ))
    return {"fields": fields, "q_rows": q_rows, "vd_rows": vd_rows, "edges": edges, "warnings": _warning_rows(posterior)}


def attach_race_bridge_to_run(
    *,
    run_dir: str | Path,
    sim_events_path: str | Path,
    bridge_prior_path: str | Path,
    municipality_cod6: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    sim_events_path = Path(sim_events_path)
    bridge_prior_path = Path(bridge_prior_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not sim_events_path.exists():
        raise FileNotFoundError(f"SIM events not found: {sim_events_path}")
    if not bridge_prior_path.exists():
        raise FileNotFoundError(f"Bridge prior not found: {bridge_prior_path}")

    prior = load_race_bridge_prior(bridge_prior_path)
    counts = summarize_sim_admin_race_counts(sim_events_path, municipality_cod6=municipality_cod6)
    posterior = fixedc_dynamic_weight_bridge(counts, prior)
    rows = _build_rows(posterior, sim_events_path=sim_events_path, bridge_prior_path=bridge_prior_path)

    summary_rows = []
    for target in prior.target_categories:
        summary_rows.append({
            "race_target_category": target,
            "posterior_count": posterior.posterior_counts[target],
            "lower_count": posterior.lower_counts[target],
            "upper_count": posterior.upper_counts[target],
            "sensitivity_width": posterior.sensitivity_width,
            "race_bridge_cv": posterior.race_bridge_cv,
            "missing_race_share": posterior.missing_share,
            "prior_hash": prior.prior_hash,
            "bridge_id": prior.bridge_id,
        })
    summary_path = run_dir / "Tables" / "race_bridge_summary.parquet"
    pl.DataFrame(summary_rows).write_parquet(summary_path)

    _append_rows(run_dir / "V_fields.parquet", rows["fields"], id_column="field_id", remove_values=RACE_BRIDGE_FIELD_IDS)
    _append_rows(run_dir / "Q_tensor.parquet", rows["q_rows"], id_column="field_id", remove_values=RACE_BRIDGE_FIELD_IDS)
    _append_rows(run_dir / "VariableDictionary.parquet", rows["vd_rows"], id_column="field_id", remove_values=RACE_BRIDGE_FIELD_IDS)
    _append_rows(run_dir / "E_DAG.parquet", rows["edges"], id_column="edge_id")
    _append_rows(run_dir / "Warnings.parquet", rows["warnings"], id_column="warning_id", remove_values=RACE_BRIDGE_WARNING_IDS)

    bridge_manifest = {
        "bridge_id": prior.bridge_id,
        "mode": prior.mode,
        "prior_hash": prior.prior_hash,
        "prior_path": str(bridge_prior_path),
        "source_axis": prior.source_axis,
        "target_axis": prior.target_axis,
        "missing_race_share": posterior.missing_share,
        "sensitivity_width": posterior.sensitivity_width,
        "race_bridge_cv": posterior.race_bridge_cv,
        "raw_admin_counts_preserved": True,
        "missing_category_preserved": True,
        "attach_stage": "race_bridge",
        "posterior_field_ids": [f"SIMRaceBridgePosteriorCount_{target}" for target in prior.target_categories],
        "raw_admin_field_ids": [f"SIMRaceAdminRawCount_{code}" for code in ADMIN_RACE_LABELS],
    }
    source_updates = {
        "race_bridge_prior": sha256_file(bridge_prior_path),
        "race_bridge_summary": sha256_file(summary_path),
    }

    run_config_path = run_dir / "RunConfig.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    run_config["race_bridge"] = bridge_manifest
    run_config.setdefault("source_hashes", {}).update(source_updates)
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    manifest_path = run_dir / "ReproducibilityManifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("source_hashes", {}).update(source_updates)
        manifest["race_bridge"] = bridge_manifest
        manifest.setdefault("environment", {}).update({"python": sys.version.split()[0], "platform": platform.platform()})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    p_path = run_dir / "P_vector.json"
    if p_path.exists():
        p = json.loads(p_path.read_text(encoding="utf-8"))
        if isinstance(p, dict):
            p["race_bridge"] = bridge_manifest
            p.setdefault("source_hashes", {}).update(source_updates)
            p_path.write_text(json.dumps(p, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    validation = validate_output_bundle(run_dir=str(run_dir))
    if not validation.ok:
        raise RuntimeError("Race bridge attachment produced invalid output bundle: " + "; ".join(validation.errors))
    return run_dir
