from __future__ import annotations

import numpy as np

from pegasus.compute.glm import select_count_family
from pegasus.pirs.schemas import FieldCandidate, ModelFamily


def family_for_outcome(*, outcome: FieldCandidate, offset: FieldCandidate | None = None) -> ModelFamily:
    if outcome.carrier == "event_count" and offset is not None:
        if outcome.variance is None:
            return "poisson_count_with_log_offset"
        sample = np.array([max(float(outcome.variance), 0.0)], dtype=float)
        return select_count_family(sample)  # type: ignore[return-value]
    if outcome.carrier == "proportion":
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
