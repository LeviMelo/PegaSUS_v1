from __future__ import annotations

import json

import pytest

from pegasus.output.sidra_denominator_anchor import (
    _population_v_field,
    _uf_year_support_alignment,
)
from pegasus.sidra.population_cube.anchor import SidraPopulationAnchor


def _anchor() -> SidraPopulationAnchor:
    return SidraPopulationAnchor(
        field_id="anchor",
        table_id="9606",
        variable_id="93",
        period="2022",
        locality_level="N3",
        locality_id="27",
        value=3_100_000.0,
        unit="persons",
        request_hash="request",
        metadata_hash="metadata",
        classification_tuple=[("2", "Sexo"), ("86", "Cor ou raça"), ("287", "Idade")],
        category_tuple=[("2", "6794"), ("86", "95251"), ("287", "100362")],
    )


def test_sidra_n3_anchor_is_uf_year_not_fake_municipality() -> None:
    row = _population_v_field(_anchor(), facts_path="sidra.parquet")
    support = json.loads(row["support_json"])
    axes = json.loads(row["axes_json"])

    assert support["support"] == "uf_year"
    assert support["ufs"] == ["27"]
    assert "municipalities" not in support
    assert axes["geography"] == "uf"


def test_uf_year_alignment_accepts_alagoas_datasus_cod6_prefix() -> None:
    alignment = _uf_year_support_alignment(
        anchor=_anchor(),
        numerator_support={"years": [2022], "municipalities": ["270030", "270430"]},
        numerator_axes={"geography": "mun_residence_cod6"},
        denominator_axes={"geography": "uf"},
    )

    assert alignment is not None
    assert alignment["aligned"] is True
    assert alignment["support"] == "uf_year"
    assert alignment["uf"] == "27"
    assert alignment["municipality_count"] == 2


def test_uf_year_alignment_rejects_out_of_state_municipality() -> None:
    with pytest.raises(ValueError, match="outside the denominator UF"):
        _uf_year_support_alignment(
            anchor=_anchor(),
            numerator_support={"years": [2022], "municipalities": ["270430", "280030"]},
            numerator_axes={"geography": "mun_residence_cod6"},
            denominator_axes={"geography": "uf"},
        )
