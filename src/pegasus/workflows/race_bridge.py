from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.core.schemas import UserIntent
from pegasus.efg.race_bridge import (
    fixedc_dynamic_weight_bridge,
    load_race_bridge_prior,
    summarize_sim_admin_race_counts,
)
from pegasus.output.race_bridge_attach import attach_race_bridge_to_run
from pegasus.output.validate import validate_output_bundle
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
    registry_path: str | Path = "config/registries/race_bridge_priors.yaml",
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
        "will_downgrade_dashboard_safety": posterior.sensitivity_width > 0.05 or posterior.missing_share > 0,
        "plan": plan_payload,
    }
    return {"summary": summary, "summary_json": json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)}


def run_attach_race_bridge(
    *,
    run_dir: str | Path,
    sim_events_path: str | Path,
    bridge_prior_path: str | Path,
    municipality_cod6: str | None = None,
) -> dict[str, Any]:
    output = attach_race_bridge_to_run(
        run_dir=run_dir,
        sim_events_path=sim_events_path,
        bridge_prior_path=bridge_prior_path,
        municipality_cod6=municipality_cod6,
    )
    validation = validate_output_bundle(run_dir=str(output))
    run_config = {}
    config_path = Path(output) / "RunConfig.json"
    if config_path.exists():
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
    return {"run_dir": output, "validation": validation, "race_bridge": run_config.get("race_bridge")}
