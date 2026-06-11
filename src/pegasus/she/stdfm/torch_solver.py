from __future__ import annotations

from pegasus.she.stdfm.blocked import blocked_solver_pending
from pegasus.she.stdfm.schema import STDFMInputSchema, STDFMOutputSchema


def solve_stdfm(input_schema: STDFMInputSchema, *, allow_uncertified: bool = False) -> STDFMOutputSchema:
    if not allow_uncertified:
        return blocked_solver_pending(
            field_id=input_schema.field_id,
            reason="PyTorch ST-DFM solver is blocked until calibration and certification are supplied.",
        )
    return blocked_solver_pending(
        field_id=input_schema.field_id,
        reason="Uncertified ST-DFM execution is not available in Slice 7A.",
    )
