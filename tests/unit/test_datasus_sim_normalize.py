from __future__ import annotations

import polars as pl

from pegasus.datasus.normalize_sim import normalize_sim_do
from pegasus.datasus.schemas import SIM_DO_NORMALIZED_SCHEMA


def test_normalize_sim_do_emits_contract_columns() -> None:
    raw = pl.DataFrame(
        {
            "DTOBITO": ["20220131"],
            "DTNASC": ["19800131"],
            "IDADE": ["4042"],
            "SEXO": ["1"],
            "RACACOR": ["4"],
            "CODMUNRES": ["270430"],
            "CODMUNOCOR": ["270430"],
            "CODESTAB": ["1234567"],
            "CAUSABAS": ["I219"],
            "DIFDATA": ["3"],
            "TPPOS": ["1"],
            "ALTCAUSA": ["2"],
            "ASSISTMED": ["1"],
            "GRAVIDEZ": ["2"],
            "PUERPERIO": ["3"],
            "TIPOBITO": ["2"],
            "SEMAGESTAC": ["36"],
            "LINHAA": ["*A419"],
            "LINHAB": ["R99"],
        }
    )

    normalized = normalize_sim_do(raw, source_manifest_hash="manifest_hash")

    assert normalized.height == 1
    assert normalized.columns == list(SIM_DO_NORMALIZED_SCHEMA.keys())

    row = normalized.row(0, named=True)

    assert row["source_system"] == "SIM-DO"
    assert row["source_manifest_hash"] == "manifest_hash"

    assert row["death_date"] == "2022-01-31"
    assert row["death_date_state"] == "valid"

    assert row["birth_date"] == "1980-01-31"
    assert row["birth_date_state"] == "valid"

    assert row["age_state"] == "valid"
    assert row["age_source"] == "date_difference"

    assert row["sex"] == "1"
    assert row["sex_state"] == "valid"

    assert row["race_color_admin"] == "4"
    assert row["race_axis_type"] == "administrative_death_declaration"

    assert row["mun_residence_cod6"] == "270430"
    assert row["mun_residence_state"] == "valid"

    assert row["facility_code"] == "1234567"

    assert row["underlying_icd_norm"] == "I21.9"
    assert row["underlying_icd_parse_state"] == "valid"

    assert row["reporting_delay"] == 3.0
    assert row["reporting_delay_state"] == "valid"

    assert row["medical_assistance"] == "1"
    assert row["death_during_pregnancy"] == "2"
    assert row["death_during_puerperium"] == "3"
    assert row["death_type"] == "2"
    assert row["gestational_weeks_death"] == 36.0

    assert row["cause_chain_raw"] == ["*A419", "R99"]
    assert row["cause_chain_norm"] == ["A41.9", "R99"]
    assert row["cause_chain_parse_states"] == ["valid", "ill_defined"]


def test_normalize_sim_do_preserves_missingness_and_invalid_states() -> None:
    raw = pl.DataFrame(
        {
            "DTOBITO": ["00000000"],
            "DTNASC": [None],
            "IDADE": ["9001"],
            "SEXO": ["9"],
            "RACACOR": [None],
            "CODMUNRES": ["123"],
            "CODMUNOCOR": [None],
            "CAUSABAS": [""],
        }
    )

    normalized = normalize_sim_do(raw)
    row = normalized.row(0, named=True)

    assert row["death_date_state"] == "invalid"
    assert row["birth_date_state"] == "missing"

    assert row["age_state"] == "invalid"

    assert row["sex_state"] == "unknown"

    assert row["race_color_admin"] is None
    assert row["race_color_state"] == "missing"
    assert row["race_missingness_state"] == "missing"

    assert row["mun_residence_state"] == "invalid"
    assert row["mun_occurrence_state"] == "missing"

    assert row["underlying_icd_norm"] is None
    assert row["underlying_icd_parse_state"] == "blank"


def test_normalize_sim_do_is_deterministic_for_same_input() -> None:
    raw = pl.DataFrame(
        {
            "DTOBITO": ["20220131"],
            "IDADE": ["4042"],
            "SEXO": ["1"],
            "RACACOR": ["4"],
            "CODMUNRES": ["270430"],
            "CAUSABAS": ["I219"],
        }
    )

    a = normalize_sim_do(raw, source_manifest_hash="x")
    b = normalize_sim_do(raw, source_manifest_hash="x")

    assert a.row(0, named=True)["event_id"] == b.row(0, named=True)["event_id"]
    assert a.row(0, named=True)["raw_record_hash"] == b.row(0, named=True)["raw_record_hash"]