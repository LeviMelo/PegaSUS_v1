from pathlib import Path
import json

import polars as pl
import pytest

from pegasus.datasus.normalize import normalize_sim_do_events
from pegasus.geo.support import SupportAlignmentError
from pegasus.output.sidra_denominator_anchor import attach_sidra_population_anchor_to_run
from pegasus.output.validate import validate_output_bundle
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.she.population.sidra_anchor import load_sidra_population_total_anchor
from pegasus.workflows.build_efg import build_sim_fixture_efg_run


LIVE_SHAPE_PAYLOAD = [
    {
        "NC": "Nível Territorial (Código)",
        "NN": "Nível Territorial",
        "MC": "Unidade de Medida (Código)",
        "MN": "Unidade de Medida",
        "V": "Valor",
        "D1C": "Município (Código)",
        "D1N": "Município",
        "D2C": "Ano (Código)",
        "D2N": "Ano",
        "D3C": "Variável (Código)",
        "D3N": "Variável",
        "D4C": "Sexo (Código)",
        "D4N": "Sexo",
        "D5C": "Cor ou raça (Código)",
        "D5N": "Cor ou raça",
        "D6C": "Idade (Código)",
        "D6N": "Idade",
    },
    {
        "NC": "6",
        "NN": "Município",
        "MC": "45",
        "MN": "Pessoas",
        "V": "957916",
        "D1C": "2704302",
        "D1N": "Maceió (AL)",
        "D2C": "2022",
        "D2N": "2022",
        "D3C": "93",
        "D3N": "População residente",
        "D4C": "6794",
        "D4N": "Total",
        "D5C": "95251",
        "D5N": "Total",
        "D6C": "100362",
        "D6N": "Total",
    },
]


CHUNK_REQUEST = {
    "table_id": "9606",
    "variables": ["93"],
    "periods": ["2022"],
    "locality_level": "N6",
    "localities": ["2704302"],
    "classifications": {
        "86": ["95251"],
        "2": ["6794"],
        "287": ["100362"],
    },
}


def _sidra_facts(tmp_path: Path) -> Path:
    facts = normalize_sidra_payload_to_facts(
        LIVE_SHAPE_PAYLOAD,
        table_id="9606",
        request_hash="request_hash",
        metadata_hash="a" * 64,
        chunk_request=CHUNK_REQUEST,
        unit_by_variable=None,
        fetched_at="2026-06-06T00:00:00+00:00",
    )
    path = tmp_path / "sidra_facts.parquet"
    write_facts_parquet(facts, output_path=path)
    return path


def _normalized_sim_events(tmp_path: Path) -> Path:
    sim_events = tmp_path / "sim_events.parquet"
    normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=sim_events,
        source_manifest_hash="fixture_manifest_hash",
    )
    return sim_events


def test_total_population_anchor_loader(tmp_path: Path):
    facts_path = _sidra_facts(tmp_path)
    anchor = load_sidra_population_total_anchor(facts_path)

    assert anchor.table_id == "9606"
    assert anchor.variable_id == "93"
    assert anchor.period == "2022"
    assert anchor.locality_level == "N6"
    assert anchor.locality_id == "2704302"
    assert anchor.value == 957916.0
    assert anchor.metadata_hash == "a" * 64


def test_attach_rejects_unaligned_fixture_denominator(tmp_path: Path):
    sim_events = _normalized_sim_events(tmp_path)
    run_dir = tmp_path / "run_unfiltered"
    sidra_facts = _sidra_facts(tmp_path)

    build_sim_fixture_efg_run(sim_events_path=sim_events, run_dir=run_dir)

    with pytest.raises(SupportAlignmentError, match="municipality_support_mismatch"):
        attach_sidra_population_anchor_to_run(
            run_dir=run_dir,
            sidra_facts_path=sidra_facts,
        )

    result = validate_output_bundle(run_dir=str(run_dir))
    assert result.ok, result.errors
    v = pl.read_parquet(run_dir / "V_fields.parquet")
    assert "SIMCrudeMortalitySIDRAOfficial" not in set(v["name"].to_list())


def test_attach_sidra_anchor_to_valid_run_bundle_after_support_filter(tmp_path: Path):
    sim_events = _normalized_sim_events(tmp_path)
    run_dir = tmp_path / "run_maceio"
    sidra_facts = _sidra_facts(tmp_path)

    build_sim_fixture_efg_run(
        sim_events_path=sim_events,
        run_dir=run_dir,
        municipality_cod6="270430",
    )

    attach_sidra_population_anchor_to_run(
        run_dir=run_dir,
        sidra_facts_path=sidra_facts,
    )

    result = validate_output_bundle(run_dir=str(run_dir))
    assert result.ok, result.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    names = set(v["name"].to_list())

    assert "SIDRAPopulationTotalAnchor" in names
    assert "SIMCrudeMortalitySIDRAOfficial" in names

    pop = v.filter(pl.col("name") == "SIDRAPopulationTotalAnchor").row(0, named=True)
    rate = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").row(0, named=True)

    assert pop["carrier"] == "Population"
    assert pop["source"] == '["SIDRA"]'
    assert "bounded_total_category_anchor" in pop["provenance"]

    rate_support = json.loads(rate["support_json"])
    assert rate["carrier"] == "Deaths/Population"
    assert "SIDRA" in rate["source"]
    assert rate_support["municipalities"] == ["2704302"]
    assert rate_support["support_alignment"]["aligned"] is True
    assert rate_support["support_alignment"]["numerator_municipalities_source"] == ["270430"]
    assert rate_support["support_alignment"]["denominator_municipalities_source"] == ["2704302"]

    e = pl.read_parquet(run_dir / "E_DAG.parquet")
    assert rate["field_id"] in set(e["child_field_id"].to_list())

    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    assert pop["field_id"] in set(q["field_id"].to_list())
    assert rate["field_id"] in set(q["field_id"].to_list())
