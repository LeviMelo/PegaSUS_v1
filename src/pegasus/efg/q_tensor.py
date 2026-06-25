from __future__ import annotations

from datetime import datetime, timezone

from pegasus.core.enums import FieldState
from pegasus.core.schemas import FieldNode, QState, WarningRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def provenance_risk(provenance: list[str]) -> float:
    p = set(provenance)
    if "synthetic" in p:
        return 1.0
    if "latent" in p:
        return 0.8
    if "SIM_informed_denominator_prior" in p or "facility_linkage_filtered" in p:
        return 0.5
    if "reconstructed" in p or "geneallocated" in p:
        return 0.4
    if "classification_projected" in p or "longitudinally_stitched" in p:
        return 0.3
    if "harmonized" in p or "deflated" in p or "AMC_contracted" in p:
        return 0.2
    return 0.0


def classify_q_state(
    *,
    n_eff: float | None,
    denom_fragility: float | None,
    missingness: float | None,
    risk: float,
) -> FieldState:
    n_eff = 0.0 if n_eff is None else n_eff
    denom_fragility = 1.0 if denom_fragility is None else denom_fragility
    missingness = 1.0 if missingness is None else missingness

    if n_eff >= 100 and denom_fragility < 0.05 and missingness < 0.10 and risk < 0.5:
        return FieldState.verified
    if n_eff >= 30 and denom_fragility < 0.20:
        return FieldState.fragile
    return FieldState.quarantined_descriptive


def compute_q_state(
    *,
    field: FieldNode,
    tensor=None,
    denominator=None,
    provenance: list[str] | None = None,
    warnings: list[WarningRecord] | None = None,
    registries=None,
) -> QState:
    provenance = provenance or field.provenance
    risk = provenance_risk(provenance)

    n_events = field.support.get("n_events")
    n_denom = field.support.get("n_denom")
    n_eff = field.support.get("n_eff", n_events)
    missingness = min(1.0, field.support.get("missingness", 0.0) + field.support.get("invalid_flag_share", 0.0))
    denom_fragility = field.support.get("denom_fragility", 1.0 if n_denom is None else 0.0)
    zero_inflation = field.support.get("zero_inflation", 0.0)

    state = classify_q_state(
        n_eff=n_eff,
        denom_fragility=denom_fragility,
        missingness=missingness,
        risk=risk,
    )

    race_axis = field.axes.get("race_axis_type") or field.axes.get("race_axis")

    return QState(
        field_id=field.id,
        n_events=n_events,
        n_denom=n_denom,
        n_eff=n_eff,
        cov_S=field.support.get("cov_S"),
        cov_T=field.support.get("cov_T"),
        missingness=missingness,
        zero_inflation=zero_inflation,
        denom_fragility=denom_fragility,
        cv=field.support.get("cv"),
        moran_i=field.support.get("moran_i"),
        temporal_roughness=field.support.get("temporal_roughness"),
        spatial_entropy=field.support.get("spatial_entropy"),
        provenance_risk=risk,
        race_axis_source=race_axis,
        race_axis_target=None,
        missing_race_share=field.support.get("missing_race_share"),
        emission_prior_strength=None,
        race_bridge_cv=None,
        sensitivity_width=None,
        bridge_mode=None,
        state=state,
        dashboard_safe=False if state != FieldState.verified else field.dashboard_safe,
        warnings=field.warnings,
        computed_at=_now(),
        q_schema_version="1.0",
    )
