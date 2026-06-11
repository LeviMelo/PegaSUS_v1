from __future__ import annotations

from pegasus.sidra.regime import SIDRAContextRegimeResult, classify_sidra_context_regime


def stdfm_gate_for_sidra_context(
    *,
    anchors_bounded: bool,
    concept_compatible: bool,
    temporal_points: int,
    dynamics: str,
) -> SIDRAContextRegimeResult:
    return classify_sidra_context_regime(
        missing_t=True,
        schema_stable=False,
        schema_mismatch=False,
        projectable=False,
        unit="index",
        anchors_bounded=anchors_bounded,
        concept_compatible=concept_compatible,
        temporal_points=temporal_points,
        dynamics=dynamics,
    )
