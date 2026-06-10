from pathlib import Path
import json

import polars as pl

from pegasus.datasus.normalize import normalize_sim_do_events
from pegasus.output.validate import validate_output_bundle
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.workflows.build_efg import build_sim_fixture_efg_run
from pegasus.workflows.efg import run_attach_sidra_denominator


def _sidra_facts(tmp_path: Path) -> Path:
    payload = [
        {
            "NC": "Nível Territorial (Código)", "NN": "Nível Territorial", "MC": "Unidade de Medida (Código)",
            "MN": "Unidade de Medida", "V": "Valor", "D1C": "Município (Código)", "D1N": "Município",
            "D2C": "Ano (Código)", "D2N": "Ano", "D3C": "Variável (Código)", "D3N": "Variável",
            "D4C": "Sexo (Código)", "D4N": "Sexo", "D5C": "Cor ou raça (Código)", "D5N": "Cor ou raça",
            "D6C": "Idade (Código)", "D6N": "Idade",
        },
        {
            "NC": "6", "NN": "Município", "MC": "45", "MN": "Pessoas", "V": "957916",
            "D1C": "2704302", "D1N": "Maceió (AL)", "D2C": "2022", "D2N": "2022",
            "D3C": "93", "D3N": "População residente", "D4C": "6794", "D4N": "Total",
            "D5C": "95251", "D5N": "Total", "D6C": "100362", "D6N": "Total",
        },
    ]
    chunk_request = {
        "table_id": "9606",
        "variables": ["93"],
        "periods": ["2022"],
        "locality_level": "N6",
        "localities": ["2704302"],
        "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
    }
    facts = normalize_sidra_payload_to_facts(
        payload,
        table_id="9606",
        request_hash="request_hash",
        metadata_hash="b" * 64,
        chunk_request=chunk_request,
        fetched_at="2026-06-06T00:00:00+00:00",
    )
    out = tmp_path / "sidra_facts.parquet"
    write_facts_parquet(facts, output_path=out)
    return out


def test_maceio_filtered_sim_sidra_denominator_integration_validates(tmp_path: Path):
    sim_events = tmp_path / "sim_events.parquet"
    run_dir = tmp_path / "run"
    sidra_facts = _sidra_facts(tmp_path)

    normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=sim_events,
        source_manifest_hash="fixture_manifest_hash",
    )
    build_sim_fixture_efg_run(
        sim_events_path=sim_events,
        run_dir=run_dir,
        municipality_cod6="270430",
    )
    result = run_attach_sidra_denominator(run_dir=run_dir, sidra_facts_path=sidra_facts)
    assert result["validation"].ok, result["validation"].errors

    bundle_validation = validate_output_bundle(run_dir=str(run_dir))
    assert bundle_validation.ok, bundle_validation.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    rate = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").row(0, named=True)
    support = json.loads(rate["support_json"])
    assert support["support_alignment"]["aligned"] is True
    assert support["support_alignment"]["crosswalk"] == "datasus_cod6_to_ibge_cod7"
    assert support["municipalities"] == ["2704302"]

    warnings = pl.read_parquet(run_dir / "Warnings.parquet")
    assert "support_aligned_by_municipality_crosswalk" in set(warnings["warning_id"].to_list())
