from pathlib import Path

import polars as pl

from pegasus.sidra.metadata import (
    normalize_official_table_metadata,
    read_normalized_metadata_tables,
    write_normalized_metadata_tables,
)


def test_official_metadata_normalizer_shape(tmp_path: Path):
    metadata_json = {
        "id": "9606",
        "nome": "População residente por cor, sexo e idade",
        "variaveis": [{"id": "93", "nome": "População residente", "unidade": "Pessoas"}],
        "classificacoes": [
            {"id": "86", "nome": "Cor ou raça", "categorias": [{"id": "95251", "nome": "Total"}, {"id": "2776", "nome": "Branca"}]},
            {"id": "2", "nome": "Sexo", "categorias": [{"id": "6794", "nome": "Total"}, {"id": "4", "nome": "Homens"}]},
        ],
    }
    periods_json = [{"id": "2022", "nome": "2022"}]
    localities_json = [{"id": "2704302", "nome": "Maceió"}]

    table = normalize_official_table_metadata(
        table_id="9606",
        metadata_json=metadata_json,
        periods_json=periods_json,
        localities_json=localities_json,
        locality_level="N6",
    )

    assert table.table_id == "9606"
    assert table.variables == ["93"]
    assert table.classifications["86"] == ["95251", "2776"]
    assert table.localities_by_level["N6"] == ["2704302"]


def test_metadata_parquet_roundtrip(tmp_path: Path):
    metadata_json = {
        "id": "9606",
        "nome": "População residente por cor, sexo e idade",
        "variaveis": [{"id": "93", "nome": "População residente", "unidade": "Pessoas"}],
        "classificacoes": [{"id": "86", "nome": "Cor ou raça", "categorias": [{"id": "95251", "nome": "Total"}]}],
    }
    table = normalize_official_table_metadata(
        table_id="9606",
        metadata_json=metadata_json,
        periods_json=[{"id": "2022"}],
        localities_json=[{"id": "2704302"}],
        locality_level="N6",
    )
    from pegasus.sidra.schemas import SIDRAMetadata
    metadata = SIDRAMetadata(tables={"9606": table})

    write_normalized_metadata_tables(metadata, output_dir=tmp_path)
    loaded = read_normalized_metadata_tables(tmp_path)

    assert loaded.tables["9606"].variables == ["93"]
    assert loaded.tables["9606"].localities_by_level["N6"] == ["2704302"]
