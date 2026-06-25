from pathlib import Path

import polars as pl

from pegasus.datasus.normalize import normalize_sim_do_events
from pegasus.datasus.sinasc_normalize import normalize_sinasc_events
from pegasus.she.maternal_child_linkage import summarize_maternal_child_linkage


def test_maternal_child_linkage_summarizes_birth_and_death_support(tmp_path: Path):
    sinasc = tmp_path / "sinasc.parquet"
    sim = tmp_path / "sim.parquet"
    normalize_sinasc_events(
        input_path="tests/fixtures/datasus/sinasc_fixture.csv",
        output_path=sinasc,
        source_manifest_hash="fixture_manifest",
    )
    normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=sim,
        source_manifest_hash="fixture_manifest",
    )

    summary = summarize_maternal_child_linkage(
        sinasc_events_path=sinasc,
        sim_events_path=sim,
        municipality_cod6="270430",
        municipality_ibge_cod7="2704302",
        datasus_uf_prefix="27",
        denominator_population=957916,
    )
    assert summary.births_total == 3
    assert summary.low_birth_weight_births >= 1
    assert summary.municipalities_cod6 == ["270430"]
    assert summary.municipalities_ibge_cod7 == ["2704302"]
    assert summary.denominator_population == 957916
    rates = summary.rates()
    assert rates["crude_birth_rate"] == summary.births_total / 957916
    assert rates["infant_mortality"] is not None
