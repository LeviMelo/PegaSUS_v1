from __future__ import annotations

import polars as pl

from pegasus.datasus.profile import infer_name_based_role, profile_dataframe
from pegasus.datasus.schema_compare import compare_profiles


def test_infer_name_based_roles() -> None:
    assert infer_name_based_role("CODMUNRES") == "municipality_code"
    assert infer_name_based_role("CODESTAB") == "facility_identifier"
    assert infer_name_based_role("CGC_HOSP") == "corporate_identifier"
    assert infer_name_based_role("CAUSABAS") == "diagnosis_or_cause_code"
    assert infer_name_based_role("PROC_REA") == "procedure_code"
    assert infer_name_based_role("VAL_TOT") == "monetary_or_numeric_measure"
    assert infer_name_based_role("DTOBITO") == "date_or_period"


def test_profile_dataframe_detects_basic_column_shapes() -> None:
    df = pl.DataFrame(
        {
            "DTOBITO": ["20220101", "20220102", "00000000"],
            "CODMUNRES": ["270430", "270430", "270030"],
            "CAUSABAS": ["I219", "R99", ""],
            "CGC_HOSP": ["00000000000000", "12345678000199", None],
            "VAL_TOT": ["1,25", "2.50", ""],
        }
    )

    profile = profile_dataframe(df, source_system="SIM-DO", artifact_kind="raw")

    by_col = {column.column: column for column in profile.columns}

    assert by_col["DTOBITO"].name_based_role == "date_or_period"
    assert by_col["DTOBITO"].date_parse_rate > 0

    assert by_col["CODMUNRES"].name_based_role == "municipality_code"
    assert "digit_length_6" in by_col["CODMUNRES"].shape_flags

    assert by_col["CAUSABAS"].name_based_role == "diagnosis_or_cause_code"

    assert by_col["CGC_HOSP"].name_based_role == "corporate_identifier"
    assert "contains_zero_filler_identifier" in by_col["CGC_HOSP"].shape_flags
    assert "corporate_identifier_zero_filler_present" in by_col["CGC_HOSP"].warnings

    assert by_col["VAL_TOT"].numeric_parse_rate > 0


def test_schema_compare_detects_raw_processed_changes() -> None:
    raw_df = pl.DataFrame(
        {
            "A": ["1", "2", "3", ""],
            "B": ["x", "y", "z", "w"],
            "RAW_ONLY": ["a", "b", "c", "d"],
        }
    )

    processed_df = pl.DataFrame(
        {
            "A": ["1", None, None, None],
            "B": ["x", "y", "z", "w"],
            "PROCESSED_ONLY": ["new", "new", "new", "new"],
        }
    )

    raw_profile = profile_dataframe(raw_df, source_system="TEST", artifact_kind="raw")
    processed_profile = profile_dataframe(
        processed_df,
        source_system="TEST",
        artifact_kind="processed",
    )

    comparison = compare_profiles(raw_profile, processed_profile)

    assert comparison.raw_only_columns == ["RAW_ONLY"]
    assert comparison.processed_only_columns == ["PROCESSED_ONLY"]

    by_col = {column.column: column for column in comparison.column_comparisons}

    assert "missingness_changed" in by_col["A"].flags
    assert "raw_only" in by_col["RAW_ONLY"].flags
    assert "processed_only" in by_col["PROCESSED_ONLY"].flags