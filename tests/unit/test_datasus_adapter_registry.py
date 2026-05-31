from __future__ import annotations

import polars as pl

from pegasus.datasus.adapters.registry import get_datasus_adapter, registered_datasus_systems


def test_sim_do_adapter_is_registered() -> None:
    assert "SIM-DO" in registered_datasus_systems()

    adapter = get_datasus_adapter("SIM-DO")
    assert adapter.source_system == "SIM-DO"


def test_sim_do_adapter_normalizes_dataframe() -> None:
    adapter = get_datasus_adapter("SIM-DO")

    df = pl.DataFrame(
        {
            "DTOBITO": ["20220131"],
            "IDADE": ["4042"],
            "SEXO": ["1"],
            "RACACOR": ["4"],
            "CODMUNRES": ["270430"],
            "CAUSABAS": ["I219"],
        }
    )

    result = adapter.normalize(df, source_manifest_hash="x")

    assert result.source_system == "SIM-DO"
    assert result.normalized.height == 1
    assert result.normalized.row(0, named=True)["source_system"] == "SIM-DO"