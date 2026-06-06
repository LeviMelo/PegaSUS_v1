from __future__ import annotations

from pegasus.sidra.schemas import DenominatorContract


def official_sidra_anchor_contract(
    *,
    source: str = "SIDRA",
    warnings: list[str] | None = None,
) -> DenominatorContract:
    return DenominatorContract(
        mode="official_sidra_anchor",
        source=source,
        provenance=["official"],
        state="fragile",
        dashboard_safe="warning",
        allowed_for_rates=True,
        warnings=warnings or ["fixture_or_unvalidated_sidra_anchor"],
    )


def blocked_missing_population_contract(
    *,
    reason: str,
) -> DenominatorContract:
    return DenominatorContract(
        mode="blocked_missing",
        source="none",
        provenance=[],
        state="illegal_excluded",
        dashboard_safe=False,
        allowed_for_rates=False,
        warnings=[reason],
    )
