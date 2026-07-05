"""SCALE-01 foundation — national GeoScope + all-UF municipality validation (MSD-III §V)."""

from __future__ import annotations

from pegasus.geo.state_panel import (
    NATIONAL_UF_PREFIX,
    GeoScope,
    clean_datasus_municipalities,
    is_valid_datasus_municipality_cod6,
)


def test_national_validation_accepts_any_real_uf() -> None:
    # municipalities from different states are all valid under national scope
    assert is_valid_datasus_municipality_cod6("270430", uf_prefix=NATIONAL_UF_PREFIX)   # AL (Maceió)
    assert is_valid_datasus_municipality_cod6("355030", uf_prefix=NATIONAL_UF_PREFIX)   # SP (São Paulo)
    assert is_valid_datasus_municipality_cod6("530010", uf_prefix=NATIONAL_UF_PREFIX)   # DF (Brasília)
    # invalid: non-UF prefix, all-zero, or the state-total sentinel
    assert not is_valid_datasus_municipality_cod6("990000", uf_prefix=NATIONAL_UF_PREFIX)
    assert not is_valid_datasus_municipality_cod6("270000", uf_prefix=NATIONAL_UF_PREFIX)  # UF-total, not a muni


def test_state_validation_unchanged() -> None:
    # a specific UF prefix still restricts to that state (no regression)
    assert is_valid_datasus_municipality_cod6("270430", uf_prefix="27")
    assert not is_valid_datasus_municipality_cod6("355030", uf_prefix="27")   # SP muni under AL prefix


def test_clean_national_keeps_all_states() -> None:
    valid, invalid = clean_datasus_municipalities(
        ["270430", "355030", "530010", "990000"], uf_prefix=NATIONAL_UF_PREFIX
    )
    assert set(valid) == {"270430", "355030", "530010"}
    assert invalid == ["990000"]


def test_geoscope_national() -> None:
    scope = GeoScope.national()
    assert scope.datasus_uf_prefix == NATIONAL_UF_PREFIX
    assert scope.uf is None
    assert scope.expected_municipality_count == 5570
