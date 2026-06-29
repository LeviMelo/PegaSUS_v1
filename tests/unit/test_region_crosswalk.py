from __future__ import annotations

import pytest

from pegasus.geo.region_crosswalk import (
    RegionCrosswalkError,
    cod6_to_region_map,
    region_for_cod6,
)


def test_region_for_cod6_maps_to_real_ibge_microregion() -> None:
    # Água Branca/AL cod7=2700102 -> cod6=270010, IBGE microrregião 27001
    # (Serrana do Sertão Alagoano). Authoritative IBGE hierarchy, not derived.
    assert region_for_cod6("270010", "microregion") == "27001"
    assert region_for_cod6("270010", "mesoregion") == "2701"
    # Maceió cod7=2704302 -> cod6=270430 resolves to an immediate region id.
    assert region_for_cod6("270430", "immediate_region") is not None


def test_aggregation_is_many_to_one_and_denser_than_municipalities() -> None:
    micro = cod6_to_region_map("microregion")
    # National coverage: thousands of municipalities collapse into far fewer regions.
    assert len(micro) > 5000
    assert 1 < len(set(micro.values())) < len(micro)
    # Alagoas (cod6 prefix 27) collapses 102 municipalities to a handful of microregions.
    al_regions = {r for c, r in micro.items() if c.startswith("27")}
    assert 5 <= len(al_regions) <= 25


def test_unknown_level_rejected() -> None:
    with pytest.raises(RegionCrosswalkError):
        region_for_cod6("270010", "planet")  # type: ignore[arg-type]
