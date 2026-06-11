from __future__ import annotations

from pegasus.she.stdfm.schema import STDFMOutputSchema


def blocked_solver_pending(*, field_id: str, reason: str | None = None) -> STDFMOutputSchema:
    return STDFMOutputSchema(
        field_id=field_id,
        status="blocked_solver_pending",
        solver_backend="pytorch_cuda_pending_calibration",
        certification_id=None,
        uncertainty=None,
        warnings=("blocked_solver_pending", "stdfm_certification_required"),
        reason=reason or "ST-DFM solver is architecturally declared but blocked until calibration/certification is available.",
    )
