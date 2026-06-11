from __future__ import annotations

from pegasus.pirs.schemas import FieldCandidate, ModelFamily


def family_for_outcome(*, outcome: FieldCandidate, offset: FieldCandidate | None = None) -> ModelFamily:
    if outcome.carrier == "event_count" and offset is not None:
        return "poisson_count_with_log_offset"
    if outcome.carrier == "proportion":
        return "binomial_proportion"
    if outcome.carrier == "sih_cost_component":
        return "sih_gamma_cost_component"
    return "gaussian_identity"


def exposure_offset_source(*, family: ModelFamily, offset: FieldCandidate | None) -> str | None:
    if family == "poisson_count_with_log_offset":
        if offset is None:
            raise ValueError("poisson_count_model_requires_exposure_offset")
        if offset.carrier not in {"population_denominator", "person_time"}:
            raise ValueError(f"unsupported_exposure_offset_carrier:{offset.carrier}")
        return offset.field_id
    return None
