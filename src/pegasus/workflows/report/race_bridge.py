from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.output.table_io import append_replace_rows, write_rows
from pegasus.output.validate import validate_output_bundle
from pegasus.core.schemas import UserIntent
from pegasus.measurement.race import (
    fixedc_dynamic_weight_bridge,
    load_race_bridge_prior,
    summarize_sim_admin_race_counts,
)
from pegasus.registries.race_bridge import resolve_race_bridge_plan


def _load_intent(path: str | Path) -> UserIntent:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return UserIntent.model_validate(payload)


def run_validate_race_bridge_prior(*, bridge_prior_path: str | Path) -> dict[str, Any]:
    prior = load_race_bridge_prior(bridge_prior_path)
    return {
        "bridge_id": prior.bridge_id,
        "mode": prior.mode,
        "prior_hash": prior.prior_hash,
        "source_axis": prior.source_axis,
        "target_axis": prior.target_axis,
        "source_categories": prior.source_categories,
        "target_categories": prior.target_categories,
        "sensitivity_width": prior.sensitivity_width,
    }


def run_plan_race_bridge(
    *,
    sim_events_path: str | Path,
    bridge_prior_path: str | Path | None = None,
    intent_path: str | Path | None = None,
    registry_path: str | Path = "config/registries/demographic/race_bridge_priors.yaml",
    municipality_cod6: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    if bridge_prior_path is None:
        if intent_path is None or municipality_cod6 is None:
            raise ValueError("Bridge prior path is required unless intent_path and municipality_cod6 are supplied for registry planning.")
        intent = _load_intent(intent_path)
        plan = resolve_race_bridge_plan(intent=intent, municipality_cod6=municipality_cod6, registry_path=registry_path)
        if plan.status != "planned" or plan.prior_path is None:
            raise ValueError(f"Race bridge plan is not attachable: {plan.as_manifest()}")
        bridge_prior_path = plan.prior_path
        plan_payload = plan.as_manifest()
    else:
        plan_payload = None
    prior = load_race_bridge_prior(bridge_prior_path)
    counts = summarize_sim_admin_race_counts(sim_events_path, municipality_cod6=municipality_cod6, year=year)
    posterior = fixedc_dynamic_weight_bridge(counts, prior)
    summary = {
        "bridge_id": prior.bridge_id,
        "prior_hash": prior.prior_hash,
        "support": counts.support,
        "raw_admin_counts": counts.raw_admin_counts,
        "missing_count": counts.missing_count,
        "missing_share": counts.missing_share,
        "posterior_counts": posterior.posterior_counts,
        "lower_counts": posterior.lower_counts,
        "upper_counts": posterior.upper_counts,
        "sensitivity_width": posterior.sensitivity_width,
        "race_bridge_cv": posterior.race_bridge_cv,
        "bridge_uncertainty_mode": posterior.bridge_mode,
        "local_target_pi": posterior.support.get("local_target_pi"),
        "local_pi_source": posterior.support.get("local_pi_source"),
        "will_downgrade_dashboard_safety": posterior.sensitivity_width > 0.05 or posterior.missing_share > 0,
        "plan": plan_payload,
    }
    return {"summary": summary, "summary_json": json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)}


def _field_row(field_id: str, *, name: str, axes: dict[str, Any], support: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "name": name,
        "kind": "race_bridge",
        "carrier": "Deaths",
        "unit": "count",
        "aggregation": "additive",
        "role": json.dumps(["race_bridge", "observer_process"], sort_keys=True),
        "source": json.dumps(["SIM-DO", "IBGE"], sort_keys=True),
        "support_json": json.dumps(support, sort_keys=True),
        "axes_json": json.dumps(axes, sort_keys=True),
        "operator": "Bridge_R_localPi_posteriorC",
        "provenance": json.dumps(["SIM-DO", "IBGE", "race_bridge"], sort_keys=True),
        "state": "fragile",
        "dashboard_safe": "False",
        "warnings": json.dumps(warnings, sort_keys=True),
        "lineage_hash": field_id,
        "registry_hash": str(axes.get("prior_hash") or ""),
        "materialization_state": "materialized",
        "path": "",
    }


def _q_row(field_id: str, *, n_events: float, summary: dict[str, Any], prior: Any) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "n_events": n_events,
        "n_denom": 0.0,
        "n_eff": n_events,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": float(summary.get("missing_share") or 0.0),
        "zero_inflation": 0.0,
        "denom_fragility": 0.0,
        "cv": float(summary.get("race_bridge_cv") or 0.0),
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.2,
        "race_axis_source": prior.source_axis,
        "race_axis_target": prior.target_axis,
        "missing_race_share": float(summary.get("missing_share") or 0.0),
        "emission_prior_strength": 1.0,
        "race_bridge_cv": float(summary.get("race_bridge_cv") or 0.0),
        "sensitivity_width": float(summary.get("sensitivity_width") or 0.0),
        "bridge_mode": str(summary.get("bridge_uncertainty_mode") or prior.mode),
        "state": "fragile",
        "dashboard_safe": "False",
        "warnings": json.dumps(["bayesian_ecological_bridge_warning", "race_bridge_posterior_uncertainty"]),
        "computed_at": "race_bridge_attach",
        "q_schema_version": "1.0",
    }


def _vd_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": row["field_id"],
        "display_name": row["name"],
        "technical_name": row["name"],
        "definition": f"Race bridge field {row['name']}.",
        "estimand_label": "race_bridge",
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


def _update_json(path: Path, key: str, payload: dict[str, Any]) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = payload
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def run_attach_race_bridge(
    *,
    run_dir: str | Path,
    sim_events_path: str | Path,
    bridge_prior_path: str | Path,
    municipality_cod6: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    prior = load_race_bridge_prior(bridge_prior_path)
    counts = summarize_sim_admin_race_counts(sim_events_path, municipality_cod6=municipality_cod6, year=year)
    posterior = fixedc_dynamic_weight_bridge(counts, prior)
    summary = {
        "bridge_id": prior.bridge_id,
        "prior_hash": prior.prior_hash,
        "support": counts.support,
        "raw_admin_counts": counts.raw_admin_counts,
        "missing_count": counts.missing_count,
        "missing_share": counts.missing_share,
        "posterior_counts": posterior.posterior_counts,
        "lower_counts": posterior.lower_counts,
        "upper_counts": posterior.upper_counts,
        "sensitivity_width": posterior.sensitivity_width,
        "race_bridge_cv": posterior.race_bridge_cv,
        "bridge_uncertainty_mode": posterior.bridge_mode,
        "local_target_pi": posterior.support.get("local_target_pi"),
        "local_pi_source": posterior.support.get("local_pi_source"),
    }
    base_axes = {
        "numerator_axis_source": prior.source_axis,
        "denominator_axis_target": prior.target_axis,
        "bridge_operator": "Bridge_R_localPi_posteriorC",
        "emission_matrix_registry_version": prior.bridge_id,
        "bridge_mode": prior.mode,
        "bridge_uncertainty_mode": posterior.bridge_mode,
        "missing_race_share": counts.missing_share,
        "race_bridge_cv": posterior.race_bridge_cv,
        "sensitivity_width": posterior.sensitivity_width,
        "race_axis_warning": "bayesian_ecological_bridge_warning",
        "bayesian_ecological_bridge_warning": True,
        "prior_hash": prior.prior_hash,
    }
    fields = []
    q_rows = []
    for category, value in posterior.posterior_counts.items():
        axes = dict(base_axes)
        axes.update({
            "race_category": category,
            "lower_count": posterior.lower_counts.get(category),
            "upper_count": posterior.upper_counts.get(category),
            "bridge_uncertainty_mode": posterior.bridge_mode,
        })
        fid = f"SIMRaceBridgePosteriorCount_{category}"
        row = _field_row(
            fid,
            name=f"SIMRaceBridgePosteriorCount_{category}",
            axes=axes,
            support=counts.support,
            warnings=["bayesian_ecological_bridge_warning"],
        )
        fields.append(row)
        q_rows.append(_q_row(fid, n_events=float(value), summary=summary, prior=prior))
    missing_axes = {
        "missing_category_preserved": True,
        "missing_race_share": counts.missing_share,
        "bridge_mode": prior.mode,
        "bridge_uncertainty_mode": posterior.bridge_mode,
    }
    missing = _field_row(
        "SIMRaceBridgeMissingRaceObserver",
        name="SIMRaceBridgeMissingRaceObserver",
        axes=missing_axes,
        support=counts.support,
        warnings=["missing_race_observer_process"],
    )
    fields.append(missing)
    q_rows.append(_q_row("SIMRaceBridgeMissingRaceObserver", n_events=float(counts.missing_count), summary=summary, prior=prior))
    append_replace_rows(run / "V_fields.parquet", fields, id_column="field_id")
    append_replace_rows(run / "Q_tensor.parquet", q_rows, id_column="field_id")
    append_replace_rows(run / "VariableDictionary.parquet", [_vd_row(row) for row in fields], id_column="field_id")
    tables = run / "Tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_rows(tables / "race_bridge_summary.parquet", [summary])
    metadata = {
        "bridge_id": prior.bridge_id,
        "mode": prior.mode,
        "prior_hash": prior.prior_hash,
        "source_axis": prior.source_axis,
        "target_axis": prior.target_axis,
        "missing_race_share": counts.missing_share,
        "sensitivity_width": posterior.sensitivity_width,
        "race_bridge_cv": posterior.race_bridge_cv,
        "bridge_uncertainty_mode": posterior.bridge_mode,
        "local_pi_source": posterior.support.get("local_pi_source"),
        "raw_admin_counts_preserved": True,
        "missing_category_preserved": True,
        "attach_stage": "race_bridge",
    }
    _update_json(run / "RunConfig.json", "race_bridge", metadata)
    _update_json(run / "ReproducibilityManifest.json", "race_bridge", metadata)
    validation = validate_output_bundle(run_dir=str(run))
    return {"summary": summary, "validation": validation}
