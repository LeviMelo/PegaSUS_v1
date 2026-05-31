from __future__ import annotations

from typing import Any, TypeAlias

import polars as pl


Schema: TypeAlias = dict[str, Any]


SIM_DO_NORMALIZED_SCHEMA: Schema = {
    "event_id": pl.Utf8,
    "source_system": pl.Utf8,
    "source_manifest_hash": pl.Utf8,
    "row_number": pl.Int64,

    "death_date": pl.Utf8,
    "death_date_state": pl.Utf8,
    "birth_date": pl.Utf8,
    "birth_date_state": pl.Utf8,

    "raw_age": pl.Utf8,
    "age_years": pl.Float64,
    "age_days": pl.Float64,
    "age_state": pl.Utf8,
    "age_source": pl.Utf8,

    "sex": pl.Utf8,
    "sex_state": pl.Utf8,

    "race_color_admin": pl.Utf8,
    "race_color_state": pl.Utf8,
    "race_axis_type": pl.Utf8,
    "race_missingness_state": pl.Utf8,

    "mun_residence_cod6": pl.Utf8,
    "mun_residence_cod7": pl.Utf8,
    "mun_residence_state": pl.Utf8,
    "mun_occurrence_cod6": pl.Utf8,
    "mun_occurrence_cod7": pl.Utf8,
    "mun_occurrence_state": pl.Utf8,

    "facility_code": pl.Utf8,
    "facility_code_state": pl.Utf8,

    "underlying_icd_raw": pl.Utf8,
    "underlying_icd_norm": pl.Utf8,
    "underlying_icd_parse_state": pl.Utf8,
    "underlying_icd_warnings": pl.List(pl.Utf8),

    "cause_chain_raw": pl.List(pl.Utf8),
    "cause_chain_norm": pl.List(pl.Utf8),
    "cause_chain_parse_states": pl.List(pl.Utf8),
    "cause_chain_warnings": pl.List(pl.Utf8),

    "reporting_delay": pl.Float64,
    "reporting_delay_state": pl.Utf8,
    "investigation_status": pl.Utf8,
    "investigation_status_state": pl.Utf8,
    "cause_altered": pl.Utf8,
    "cause_altered_state": pl.Utf8,
    "medical_assistance": pl.Utf8,
    "medical_assistance_state": pl.Utf8,

    "death_during_pregnancy": pl.Utf8,
    "death_during_pregnancy_state": pl.Utf8,
    "death_during_puerperium": pl.Utf8,
    "death_during_puerperium_state": pl.Utf8,
    "death_type": pl.Utf8,
    "death_type_state": pl.Utf8,
    "gestational_weeks_death": pl.Float64,
    "gestational_weeks_death_state": pl.Utf8,

    "raw_record_hash": pl.Utf8,
    "processed_record_hash": pl.Utf8,
    "normalization_warnings": pl.List(pl.Utf8),
}


def empty_frame(schema: Schema) -> pl.DataFrame:
    return pl.DataFrame(
        {name: pl.Series(name=name, values=[], dtype=dtype) for name, dtype in schema.items()}
    )


def enforce_schema(df: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    """Return a dataframe with exactly schema columns and dtypes.

    Missing columns are created as nulls. Extra columns are removed. This is used
    for normalized event tables so that downstream code can depend on column
    contracts.
    """
    if df.is_empty() and df.width == 0:
        return empty_frame(schema)

    out = df

    for name, dtype in schema.items():
        if name not in out.columns:
            out = out.with_columns(pl.lit(None).cast(dtype).alias(name))

    out = out.select([pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()])
    return out