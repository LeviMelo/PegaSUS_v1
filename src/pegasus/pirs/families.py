from __future__ import annotations

import numpy as np

from pegasus.compute.glm import select_count_family
from pegasus.pirs.schemas import FieldCandidate, ModelFamily


_PROPORTION_DENOMINATORS = {
    "livebirths",
    "hospitaladmissions",
    "admissions",
    "births",
    "deaths",
}


def _carrier_parts(carrier: str) -> tuple[str, str] | None:
    text = str(carrier or "").strip()
    if "/" not in text:
        return None
    numerator, denominator = text.split("/", 1)
    numerator = numerator.strip()
    denominator = denominator.strip()
    if not numerator or not denominator:
        return None
    return numerator, denominator


def _carrier_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def is_population_rate_outcome(outcome: FieldCandidate) -> bool:
    parts = _carrier_parts(outcome.carrier)
    if parts is None:
        return False
    return _carrier_token(parts[1]) in {"population", "persons", "personyears", "persontime"}


def is_binomial_ratio_outcome(outcome: FieldCandidate) -> bool:
    if outcome.carrier == "proportion" or outcome.unit == "proportion":
        return True
    parts = _carrier_parts(outcome.carrier)
    if parts is None:
        return False
    return _carrier_token(parts[1]) in _PROPORTION_DENOMINATORS


def pirs_outcome_blocking_reason(outcome: FieldCandidate) -> str | None:
    if is_population_rate_outcome(outcome):
        return "population_rate_requires_count_response_with_log_exposure"
    return None


def family_for_outcome(*, outcome: FieldCandidate, offset: FieldCandidate | None = None) -> ModelFamily:
    blocked = pirs_outcome_blocking_reason(outcome)
    if blocked is not None:
        raise ValueError(blocked)
    if outcome.carrier == "event_count" and offset is not None:
        if outcome.variance is None:
            return "poisson_count_with_log_offset"
        sample = np.array([max(float(outcome.variance), 0.0)], dtype=float)
        return select_count_family(sample)  # type: ignore[return-value]
    if is_binomial_ratio_outcome(outcome):
        return "beta_binomial" if "overdispersed" in set(outcome.warnings) else "binomial_proportion"
    if outcome.carrier in {"composition", "simplex"}:
        return "dirichlet"
    if outcome.carrier in {"skewed_positive", "cost_component"}:
        return "lognormal" if "zero_mass" not in set(outcome.warnings) else "two_part_lognormal"
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
