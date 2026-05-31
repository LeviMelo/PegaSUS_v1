from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from pegasus.datasus.parsers import clean_scalar_string, parse_date, parse_numeric


ObservedKind = Literal[
    "empty",
    "numeric_like",
    "date_like",
    "categorical_like",
    "string_like",
    "mixed",
]


class ColumnProfile(BaseModel):
    column: str
    dtype_observed: str
    n_rows: int
    n_null: int
    n_blank: int
    n_blank_or_null: int
    missing_rate: float
    n_nonblank: int
    n_unique: int
    unique_rate: float
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    numeric_parse_rate: float
    numeric_min: float | None = None
    numeric_max: float | None = None
    date_parse_rate: float
    date_min: str | None = None
    date_max: str | None = None
    digit_lengths: dict[str, int] = Field(default_factory=dict)
    value_length_distribution: dict[str, int] = Field(default_factory=dict)
    shape_flags: list[str] = Field(default_factory=list)
    name_based_role: str | None = None
    observed_kind: ObservedKind
    warnings: list[str] = Field(default_factory=list)


class DataFrameProfile(BaseModel):
    source_system: str
    artifact_kind: Literal["raw", "processed", "normalized", "unknown"]
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]


def infer_name_based_role(column: str) -> str | None:
    c = column.upper()

    municipality_names = {
        "CODMUNRES",
        "CODMUNOCOR",
        "CODMUNNASC",
        "CODUFMUN",
        "MUNIC_RES",
        "MUNIC_MOV",
    }
    facility_names = {
        "CNES",
        "CODESTAB",
        "CO_CNES",
    }
    cnpj_names = {
        "CGC_HOSP",
        "CNPJ",
        "CNPJ_MANT",
    }
    icd_names = {
        "CAUSABAS",
        "CAUSABAS_O",
        "DIAG_PRINC",
        "DIAG_SECUN",
        "CODANOMAL",
    }

    if c in municipality_names:
        return "municipality_code"

    if c in facility_names or c.startswith("CNES_"):
        return "facility_identifier"

    if c in cnpj_names or "CNPJ" in c or "CGC" in c:
        return "corporate_identifier"

    if c in icd_names or c.startswith("CID_") or c.startswith("LINHA"):
        return "diagnosis_or_cause_code"

    if c.startswith("PROC_") or c in {"PROC_REA", "PROC_SOLIC"}:
        return "procedure_code"

    if c.startswith("VAL_") or c in {"VAL_TOT"}:
        return "monetary_or_numeric_measure"

    if c.startswith("DT") or "DATA" in c or "DATE" in c:
        return "date_or_period"

    return None


def _iter_nonblank_values(series: pl.Series, sample_limit: int | None = None) -> list[Any]:
    values: list[Any] = []
    count = 0

    for value in series:
        text = clean_scalar_string(value)
        if text is None or text == "":
            continue

        values.append(value)
        count += 1

        if sample_limit is not None and count >= sample_limit:
            break

    return values


def _count_blank(series: pl.Series) -> int:
    count = 0
    for value in series:
        text = clean_scalar_string(value)
        if text == "":
            count += 1
    return count


def _top_values(values: list[Any], limit: int = 10) -> list[tuple[str, int]]:
    counter = Counter(str(v) for v in values)
    return counter.most_common(limit)


def _digit_length_distribution(values: list[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for value in values:
        text = str(value).strip()
        if text.isdigit():
            counter[str(len(text))] += 1

    return dict(counter)


def _value_length_distribution(values: list[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter(str(len(str(value))) for value in values)
    return dict(counter)


def _numeric_stats(values: list[Any]) -> tuple[float, float | None, float | None]:
    if not values:
        return 0.0, None, None

    parsed_values: list[float] = []

    for value in values:
        parsed = parse_numeric(value)
        if parsed.state == "valid" and isinstance(parsed.normalized, float):
            parsed_values.append(parsed.normalized)

    rate = len(parsed_values) / len(values)

    if not parsed_values:
        return rate, None, None

    return rate, min(parsed_values), max(parsed_values)


def _date_stats(values: list[Any]) -> tuple[float, str | None, str | None]:
    if not values:
        return 0.0, None, None

    parsed_values: list[str] = []

    for value in values:
        parsed = parse_date(value)
        if parsed.state == "valid" and parsed.normalized is not None:
            parsed_values.append(parsed.normalized)

    rate = len(parsed_values) / len(values)

    if not parsed_values:
        return rate, None, None

    return rate, min(parsed_values), max(parsed_values)


def _shape_flags(
    *,
    values: list[Any],
    digit_lengths: dict[str, int],
    name_based_role: str | None,
    numeric_parse_rate: float,
    date_parse_rate: float,
) -> list[str]:
    flags: list[str] = []

    if digit_lengths:
        flags.append("digit_code_like")
        for length in sorted(digit_lengths):
            flags.append(f"digit_length_{length}")

    if numeric_parse_rate >= 0.95:
        flags.append("numeric_like")

    if date_parse_rate >= 0.80:
        flags.append("date_like")

    text_values = [str(v).strip() for v in values]
    if any(v.startswith("*") for v in text_values):
        flags.append("contains_prefixed_asterisk_values")

    if name_based_role == "corporate_identifier" and any(set(v) == {"0"} for v in text_values):
        flags.append("contains_zero_filler_identifier")

    return flags


def _observed_kind(
    *,
    n_nonblank: int,
    n_unique: int,
    numeric_parse_rate: float,
    date_parse_rate: float,
) -> ObservedKind:
    if n_nonblank == 0:
        return "empty"

    if date_parse_rate >= 0.80:
        return "date_like"

    if numeric_parse_rate >= 0.95:
        return "numeric_like"

    unique_rate = n_unique / n_nonblank if n_nonblank else 0.0

    if unique_rate <= 0.20 or n_unique <= 30:
        return "categorical_like"

    if numeric_parse_rate > 0.05:
        return "mixed"

    return "string_like"


def profile_column(
    series: pl.Series,
    *,
    sample_limit: int | None = None,
    top_values_limit: int = 10,
) -> ColumnProfile:
    n_rows = len(series)
    n_null = series.null_count()
    n_blank = _count_blank(series)
    n_blank_or_null = n_null + n_blank
    n_nonblank = max(n_rows - n_blank_or_null, 0)

    sampled_nonblank = _iter_nonblank_values(series, sample_limit=sample_limit)
    full_nonblank = _iter_nonblank_values(series, sample_limit=None)

    n_unique = len({str(v) for v in full_nonblank})
    unique_rate = n_unique / n_nonblank if n_nonblank else 0.0
    missing_rate = n_blank_or_null / n_rows if n_rows else 0.0

    numeric_parse_rate, numeric_min, numeric_max = _numeric_stats(sampled_nonblank)
    date_parse_rate, date_min, date_max = _date_stats(sampled_nonblank)

    digit_lengths = _digit_length_distribution(sampled_nonblank)
    value_lengths = _value_length_distribution(sampled_nonblank)
    name_based_role = infer_name_based_role(series.name)

    flags = _shape_flags(
        values=sampled_nonblank,
        digit_lengths=digit_lengths,
        name_based_role=name_based_role,
        numeric_parse_rate=numeric_parse_rate,
        date_parse_rate=date_parse_rate,
    )

    observed_kind = _observed_kind(
        n_nonblank=n_nonblank,
        n_unique=n_unique,
        numeric_parse_rate=numeric_parse_rate,
        date_parse_rate=date_parse_rate,
    )

    warnings: list[str] = []

    if name_based_role == "date_or_period" and date_parse_rate < 0.80 and n_nonblank > 0:
        warnings.append("date_named_column_low_date_parse_rate")

    if name_based_role == "municipality_code" and not (
        "digit_length_6" in flags or "digit_length_7" in flags
    ):
        warnings.append("municipality_named_column_unexpected_digit_shape")

    if name_based_role == "corporate_identifier" and "contains_zero_filler_identifier" in flags:
        warnings.append("corporate_identifier_zero_filler_present")

    return ColumnProfile(
        column=series.name,
        dtype_observed=str(series.dtype),
        n_rows=n_rows,
        n_null=n_null,
        n_blank=n_blank,
        n_blank_or_null=n_blank_or_null,
        missing_rate=missing_rate,
        n_nonblank=n_nonblank,
        n_unique=n_unique,
        unique_rate=unique_rate,
        top_values=_top_values(full_nonblank, limit=top_values_limit),
        numeric_parse_rate=numeric_parse_rate,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        date_parse_rate=date_parse_rate,
        date_min=date_min,
        date_max=date_max,
        digit_lengths=digit_lengths,
        value_length_distribution=value_lengths,
        shape_flags=flags,
        name_based_role=name_based_role,
        observed_kind=observed_kind,
        warnings=warnings,
    )


def profile_dataframe(
    df: pl.DataFrame,
    *,
    source_system: str,
    artifact_kind: Literal["raw", "processed", "normalized", "unknown"],
    sample_limit: int | None = 100_000,
    top_values_limit: int = 10,
) -> DataFrameProfile:
    columns = [
        profile_column(
            df[column],
            sample_limit=sample_limit,
            top_values_limit=top_values_limit,
        )
        for column in df.columns
    ]

    return DataFrameProfile(
        source_system=source_system,
        artifact_kind=artifact_kind,
        n_rows=df.height,
        n_columns=df.width,
        columns=columns,
    )