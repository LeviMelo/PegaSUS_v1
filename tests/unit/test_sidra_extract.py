from pathlib import Path

import polars as pl

from pegasus.sidra.api import SidraClient, SidraClientConfig
from pegasus.sidra.cache import SidraJsonCache
from pegasus.sidra.extract import extract_one_chunk
from pegasus.sidra.schemas import SIDRAChunk


def test_extract_one_chunk_writes_raw_and_facts_from_real_flat_shape(tmp_path: Path):
    payload = [
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

    def transport(url, params, timeout):
        return 200, payload

    client = SidraClient(
        config=SidraClientConfig(max_retries=0),
        cache=SidraJsonCache(tmp_path / "cache"),
        transport=transport,
    )

    chunk = SIDRAChunk(
        chunk_id="chunk123",
        table_id="9606",
        variables=["93"],
        periods=["2022"],
        locality_level="N6",
        localities=["2704302"],
        classifications={"86": ["95251"], "2": ["6794"], "287": ["100362"]},
        estimated_cells=1,
        request_url="unused",
        request_params={
            "table_id": "9606",
            "variables": ["93"],
            "periods": ["2022"],
            "locality_level": "N6",
            "localities": ["2704302"],
            "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
        },
    )

    result = extract_one_chunk(
        chunk,
        client=client,
        raw_dir=tmp_path / "raw",
        facts_root=tmp_path / "facts",
        metadata_hash="metadata_hash",
        unit_by_variable=None,
    )

    assert result.status == "success"
    assert result.row_count == 1
    assert Path(result.raw_path).exists()
    assert Path(result.facts_path).exists()

    df = pl.read_parquet(result.facts_path)
    assert df.height == 1

    row = df.row(0, named=True)
    assert row["table_id"] == "9606"
    assert row["variable_id"] == "93"
    assert row["period"] == "2022"
    assert row["locality_level"] == "N6"
    assert row["locality_id"] == "2704302"
    assert row["value_numeric"] == 957916.0
    assert row["unit"] == "Pessoas"
    assert '"2","Sexo"' in row["classification_tuple"].replace(" ", "")
    assert '"2","6794"' in row["category_tuple"].replace(" ", "")
