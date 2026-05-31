from __future__ import annotations

import json

import polars as pl

from pegasus.datasus.workflow import process_sim_do_local_files


def test_process_sim_do_local_files_writes_manifest_profiles_comparison_and_normalized(
    tmp_path,
) -> None:
    raw_path = tmp_path / "sim_raw.csv"
    processed_path = tmp_path / "sim_processed.csv"
    out_root = tmp_path / "out"

    raw_df = pl.DataFrame(
        {
            "DTOBITO": ["20220131", "00000000"],
            "DTNASC": ["19800131", None],
            "IDADE": ["4042", "9001"],
            "SEXO": ["1", "9"],
            "RACACOR": ["4", None],
            "CODMUNRES": ["270430", "123"],
            "CODMUNOCOR": ["270430", None],
            "CODESTAB": ["1234567", None],
            "CAUSABAS": ["I219", ""],
            "DIFDATA": ["3", None],
            "TPPOS": ["1", None],
            "ALTCAUSA": ["2", None],
            "ASSISTMED": ["1", None],
            "LINHAA": ["*A419", None],
        }
    )

    processed_df = raw_df.with_columns(
        pl.lit("processed_extra").alias("PROCESSED_ONLY_COLUMN")
    )

    raw_df.write_csv(raw_path)
    processed_df.write_csv(processed_path)

    manifest = process_sim_do_local_files(
        raw_path=raw_path,
        processed_path=processed_path,
        out_root=out_root,
        normalization_input_kind="raw",
    )

    assert manifest.status == "success"
    assert manifest.source_system == "SIM-DO"
    assert manifest.n_raw_rows == 2
    assert manifest.n_processed_rows == 2
    assert manifest.n_normalized_rows == 2

    assert (out_root / "raw_profile.json").exists()
    assert (out_root / "processed_profile.json").exists()
    assert (out_root / "schema_comparison.json").exists()
    assert (out_root / "sim_do_normalized.parquet").exists()
    assert (out_root / "manifest.json").exists()

    normalized = pl.read_parquet(out_root / "sim_do_normalized.parquet")
    assert normalized.height == 2

    first = normalized.row(0, named=True)
    assert first["source_manifest_hash"] == manifest.workflow_id
    assert first["underlying_icd_norm"] == "I21.9"
    assert first["race_axis_type"] == "administrative_death_declaration"

    second = normalized.row(1, named=True)
    assert second["death_date_state"] == "invalid"
    assert second["underlying_icd_parse_state"] == "blank"

    with (out_root / "schema_comparison.json").open("r", encoding="utf-8") as f:
        comparison = json.load(f)

    assert "PROCESSED_ONLY_COLUMN" in comparison["processed_only_columns"]


def test_process_sim_do_local_files_can_normalize_processed_input(tmp_path) -> None:
    raw_path = tmp_path / "sim_raw.csv"
    processed_path = tmp_path / "sim_processed.csv"
    out_root = tmp_path / "out"

    raw_df = pl.DataFrame(
        {
            "DTOBITO": ["20220131"],
            "IDADE": ["4042"],
            "SEXO": ["1"],
            "RACACOR": ["4"],
            "CODMUNRES": ["270430"],
            "CAUSABAS": ["I219"],
        }
    )

    processed_df = raw_df.with_columns(
        pl.lit("270430").alias("CODMUNOCOR"),
    )

    raw_df.write_csv(raw_path)
    processed_df.write_csv(processed_path)

    manifest = process_sim_do_local_files(
        raw_path=raw_path,
        processed_path=processed_path,
        out_root=out_root,
        normalization_input_kind="processed",
    )

    normalized = pl.read_parquet(out_root / "sim_do_normalized.parquet")
    row = normalized.row(0, named=True)

    assert manifest.normalization_input_kind == "processed"
    assert row["mun_occurrence_cod6"] == "270430"