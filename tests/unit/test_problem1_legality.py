from __future__ import annotations

from pathlib import Path

from pegasus.problem1.legality import RateLegalityRequest, evaluate_rate_legality
from pegasus.registries.index import RegistryIndex
from pegasus.registries.loader import load_registry_set
from pegasus.registries.validators import validate_registry_set


def _registry_index() -> RegistryIndex:
    root = Path(__file__).resolve().parents[2]
    registry_set = load_registry_set(root / "config" / "registries")
    validate_registry_set(registry_set)
    return RegistryIndex.from_registry_set(registry_set)


def test_deaths_over_population_is_legal_without_race_axis() -> None:
    index = _registry_index()

    request = RateLegalityRequest(
        numerator_carrier="Deaths",
        denominator_carrier="Population",
        role="mortality_rate",
        numerator_unit="counts",
        denominator_unit="person_years",
    )

    result = evaluate_rate_legality(request, index)

    assert result.legal is True
    assert result.failed_terms == []


def test_sim_source_over_ibge_total_population_is_legal_when_not_race_specific() -> None:
    index = _registry_index()

    request = RateLegalityRequest(
        numerator_carrier="Deaths",
        denominator_carrier="Population",
        role="mortality_rate",
        numerator_unit="counts",
        denominator_unit="person_years",
        numerator_source_system="SIM-DO",
        denominator_source_system="IBGE",
        declaration_axis_required=False,
    )

    result = evaluate_rate_legality(request, index)

    assert result.legal is True
    assert "delta_declaration" not in result.failed_terms


def test_sim_race_over_ibge_self_declared_denominator_is_illegal_without_bridge() -> None:
    index = _registry_index()

    request = RateLegalityRequest(
        numerator_carrier="Deaths",
        denominator_carrier="Population",
        role="mortality_rate",
        numerator_unit="counts",
        denominator_unit="person_years",
        numerator_source_system="SIM-DO",
        denominator_source_system="IBGE",
        declaration_axis_required=True,
    )

    result = evaluate_rate_legality(request, index)

    assert result.legal is False
    assert "delta_declaration" in result.failed_terms
    assert any("race_axis_noncommensurable" in warning for warning in result.warnings)


def test_sim_race_over_ibge_self_declared_denominator_is_legal_with_bridge() -> None:
    index = _registry_index()

    request = RateLegalityRequest(
        numerator_carrier="Deaths",
        denominator_carrier="Population",
        role="mortality_rate",
        numerator_unit="counts",
        denominator_unit="person_years",
        numerator_source_system="SIM-DO",
        denominator_source_system="IBGE",
        declaration_axis_required=True,
        bridge_applied="Bridge_R",
    )

    result = evaluate_rate_legality(request, index)

    assert result.legal is True
    assert result.delta_declaration == 1
    assert any("race_axis_bridge_applied" in warning for warning in result.warnings)


def test_wrong_carrier_pair_is_illegal() -> None:
    index = _registry_index()

    request = RateLegalityRequest(
        numerator_carrier="Deaths",
        denominator_carrier="HospitalAdmissions",
        role="mortality_rate",
        numerator_unit="counts",
        denominator_unit="counts",
    )

    result = evaluate_rate_legality(request, index)

    assert result.legal is False
    assert "delta_carrier" in result.failed_terms