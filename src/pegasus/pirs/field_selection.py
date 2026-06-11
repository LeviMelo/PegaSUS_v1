from __future__ import annotations

from pegasus.pirs.schemas import FieldCandidate, PIRSSelectionResult


BUDGET_TOP_K: dict[str, int] = {"fast": 10, "standard": 30, "deep": 100}


def rejection_reason(candidate: FieldCandidate) -> str | None:
    if candidate.q_state in {"illegal_excluded", "blocked"}:
        return f"q_state_{candidate.q_state}_not_model_eligible"
    if candidate.zero_variance:
        return "zero_variance_field_excluded_from_design_matrix"
    return None


def select_fields_for_pirs(candidates: list[FieldCandidate], *, budget: str = "fast") -> PIRSSelectionResult:
    if budget not in BUDGET_TOP_K:
        raise ValueError(f"Unsupported PIRS budget: {budget!r}")
    top_k = BUDGET_TOP_K[budget]
    rejected: list[dict] = []
    eligible: list[FieldCandidate] = []
    for candidate in candidates:
        reason = rejection_reason(candidate)
        if reason is None:
            eligible.append(candidate)
        else:
            rejected.append({
                "field_id": candidate.field_id,
                "role": candidate.role,
                "reason": reason,
                "q_state": candidate.q_state,
                "variance": candidate.variance,
            })
    outcomes = sorted((c for c in eligible if c.role == "outcome"), key=lambda c: c.utility, reverse=True)
    covariates = sorted((c for c in eligible if c.role == "covariate"), key=lambda c: c.utility, reverse=True)
    offsets = sorted((c for c in eligible if c.role == "offset"), key=lambda c: c.utility, reverse=True)
    return PIRSSelectionResult(
        selected_outcome=outcomes[0] if outcomes else None,
        selected_covariates=tuple(covariates[:top_k]),
        selected_offset=offsets[0] if offsets else None,
        rejected=tuple(rejected),
        budget=budget,  # type: ignore[arg-type]
        top_k=top_k,
    )
