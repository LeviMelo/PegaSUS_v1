"""P3a — multi-denominator declaration + resolution (FEAT-P4).

The registry declares which denominators may normalize each quantity, one default; resolution is
deterministic and an inadmissible override refuses loudly (never a silent fallback).
"""

from __future__ import annotations

import pytest

from pegasus.output.query import (
    DenominatorRegistryError,
    QuerySpec,
    load_denominator_registry,
    resolve_denominator,
)
from pegasus.output.query.denominators import _parse_quantity


def test_default_resolution() -> None:
    opt = resolve_denominator("mortality_all_cause", QuerySpec(quantity="mortality_all_cause", kind="rate"))
    assert opt.id == "resident_population" and opt.default


def test_multi_denominator_override() -> None:
    # live_births defaults to women_15_49 (general fertility rate) but resident_population is
    # admissible (crude birth rate) — the FEAT-P4 capability.
    default = resolve_denominator("live_births", QuerySpec(quantity="live_births", kind="rate"))
    assert default.id == "women_15_49"
    override = resolve_denominator(
        "live_births", QuerySpec(quantity="live_births", kind="rate", denominator="resident_population")
    )
    assert override.id == "resident_population" and not override.default
    # the choice is provenance-recordable and distinguishable
    assert default.as_provenance()["denominator_id"] != override.as_provenance()["denominator_id"]


def test_inadmissible_override_refuses() -> None:
    with pytest.raises(DenominatorRegistryError, match="not admissible"):
        resolve_denominator(
            "mortality_all_cause", QuerySpec(quantity="mortality_all_cause", denominator="facility_count")
        )


def test_unknown_quantity_refuses() -> None:
    with pytest.raises(DenominatorRegistryError, match="no denominator declaration"):
        resolve_denominator("nonexistent_quantity", QuerySpec(quantity="nonexistent_quantity"))


def test_multi_default_is_rejected() -> None:
    bad = {"admissible": [{"id": "a", "default": True}, {"id": "b", "default": True}], "extensive": True}
    with pytest.raises(DenominatorRegistryError, match="exactly one denominator must be default"):
        _parse_quantity("q", bad)


def test_empty_admissible_is_rejected() -> None:
    with pytest.raises(DenominatorRegistryError, match="non-empty list"):
        _parse_quantity("q", {"admissible": [], "extensive": True})


def test_registry_loads_and_validates() -> None:
    reg = load_denominator_registry()
    assert "mortality_all_cause" in reg and reg["mortality_all_cause"].extensive
    assert len(reg["live_births"].options) == 2  # multi-denominator


def test_registry_tree_accepts_denominators() -> None:
    from pegasus.registries.validators import validate_registry_tree

    errors = [e for e in validate_registry_tree() if "denominators" in e]
    assert not errors, f"denominators.yaml failed tree validation: {errors}"
