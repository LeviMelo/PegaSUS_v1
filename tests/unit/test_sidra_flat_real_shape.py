import json

from pegasus.sidra.normalize import flat_response_to_records, normalize_sidra_payload_to_facts


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


def test_real_sidra_flat_shape_maps_core_olap_fields():
    records = flat_response_to_records(
        LIVE_SHAPE_PAYLOAD,
        table_id="9606",
        chunk_request=CHUNK_REQUEST,
    )

    assert len(records) == 1
    row = records[0]

    assert row["table_id"] == "9606"
    assert row["variable_id"] == "93"
    assert row["period"] == "2022"
    assert row["locality_level"] == "N6"
    assert row["locality_id"] == "2704302"
    assert row["value"] == "957916"
    assert row["unit"] == "Pessoas"

    assert row["classification_tuple"] == [
        ("2", "Sexo"),
        ("86", "Cor ou raça"),
        ("287", "Idade"),
    ]
    assert row["category_tuple"] == [
        ("2", "6794"),
        ("86", "95251"),
        ("287", "100362"),
    ]


def test_real_sidra_flat_shape_normalizes_to_fact():
    facts = normalize_sidra_payload_to_facts(
        LIVE_SHAPE_PAYLOAD,
        table_id="9606",
        request_hash="request_hash",
        metadata_hash="metadata_hash",
        chunk_request=CHUNK_REQUEST,
        unit_by_variable=None,
        fetched_at="2026-06-06T00:00:00+00:00",
    )

    assert len(facts) == 1
    fact = facts[0]

    assert fact.table_id == "9606"
    assert fact.variable_id == "93"
    assert fact.period == "2022"
    assert fact.locality_level == "N6"
    assert fact.locality_id == "2704302"
    assert fact.value_raw == "957916"
    assert fact.value_numeric == 957916.0
    assert fact.value_status == "numeric"
    assert fact.unit == "Pessoas"
    assert fact.classification_tuple == (
        ("2", "Sexo"),
        ("86", "Cor ou raça"),
        ("287", "Idade"),
    )
    assert fact.category_tuple == (
        ("2", "6794"),
        ("86", "95251"),
        ("287", "100362"),
    )
