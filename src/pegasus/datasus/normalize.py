from __future__ import annotations

from pegasus.datasus.declarative_normalize import normalize_sim_do_record as _registry_normalize_sim_do_record, normalize_sinasc_record as _registry_normalize_sinasc_record

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.datasus.decoders import (
    decode_count2,
    decode_physical_scalar,
    decode_sim_idade,
)
from pegasus.datasus.icd_parser import parse_icd


SIM_DO_NORMALIZED_COLUMNS = [
    "event_id",
    "source_system",
    "year",
    "death_date",
    "death_hour",
    "birth_date",
    "age_source",
    "age_days",
    "age_years",
    "age_unit",
    "raw_age_code",
    "sex",
    "race_color_admin",
    "race_axis_type",
    "race_missingness_state",
    "mun_residence_cod6",
    "mun_residence_cod7",
    "mun_occurrence_cod6",
    "mun_occurrence_cod7",
    "place_of_death",
    "facility_code",
    "facility_code_state",
    "underlying_icd_raw",
    "underlying_icd_norm",
    "underlying_icd_parse_state",
    "cause_chain_raw",
    "cause_chain_norm",
    "cause_chain_parse_states",
    "associated_conditions_raw",
    "associated_conditions_norm",
    "associated_conditions_parse_states",
    "death_type",
    "fetal_or_liveborn_status_source",
    "maternal_age_years",
    "maternal_education_legacy",
    "maternal_education_2010",
    "maternal_occupation_cbo",
    "maternal_living_children_count",
    "maternal_deceased_children_count",
    "pregnancy_type",
    "gestational_weeks_death",
    "gestational_age_group_death",
    "delivery_type_death_context",
    "death_timing_relative_to_delivery",
    "birth_weight_death_context_grams",
    "death_during_pregnancy",
    "death_during_puerperium",
    "medical_assistance",
    "exam_performed",
    "surgery_performed",
    "autopsy_performed",
    "svo_iml_municipality",
    "certificate_date",
    "reporting_delay",
    "investigation_status",
    "investigation_date",
    "cause_altered",
    "raw_record_hash",
    "processed_record_hash",
    "source_manifest_hash",
]


INVALID_DATE_VALUES = {
    "",
    "0",
    "00000000",
    "0000-00-00",
    "00/00/0000",
    "99999999",
    "9999-99-99",
    "99/99/9999",
    "NA",
    "NAN",
    "NULL",
}


def _raw(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    return text


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, infer_schema_length=1000, ignore_errors=False)
    if suffix in {".json", ".ndjson"}:
        return pl.read_ndjson(path)

    raise ValueError(f"Unsupported SIM-DO input format: {path}")


def _parse_datasus_date(value: Any) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None

    upper = text.upper()
    if upper in INVALID_DATE_VALUES:
        return None

    # Field-specific date parser. Do not apply this generically to identifiers.
    candidates = [
        "%d%m%Y",
        "%Y%m%d",
        "%d/%m/%Y",
        "%Y-%m-%d",
    ]

    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    return None


def _parse_hour(value: Any) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None

    digits = re.sub(r"\D", "", text)
    if digits == "":
        return None

    if len(digits) <= 2:
        hour = int(digits)
        minute = 0
    else:
        padded = digits.zfill(4)
        hour = int(padded[:2])
        minute = int(padded[2:4])

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    return f"{hour:02d}:{minute:02d}:00"


def _year_from_date(iso_date: str | None, fallback: Any = None) -> int | None:
    if iso_date:
        return int(iso_date[:4])
    if fallback is not None:
        text = _clean_str(fallback)
        if text and text.isdigit() and len(text) == 4:
            return int(text)
    return None


def _mun_codes(value: Any) -> tuple[str | None, str | None]:
    text = _clean_str(value)
    if text is None:
        return None, None

    digits = re.sub(r"\D", "", text)
    if len(digits) == 6:
        return digits, None
    if len(digits) == 7:
        return digits[:6], digits

    return None, None


def _facility_code(value: Any) -> tuple[str | None, str]:
    text = _clean_str(value)
    if text is None:
        return None, "missing"
    digits = re.sub(r"\D", "", text)
    if digits == "" or set(digits) == {"0"}:
        return None, "missing"
    return digits, "valid"


def _race_state(value: Any) -> tuple[str | None, str]:
    text = _clean_str(value)
    if text is None:
        return None, "missing"

    if text in {"9", "99"}:
        return text, "unknown"

    return text, "valid"


def _int_or_none(value: Any) -> int | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        return int(float(text.replace(",", ".")))
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _days_between(start_iso: str | None, end_iso: str | None) -> int | None:
    if not start_iso or not end_iso:
        return None
    try:
        start = datetime.strptime(start_iso, "%Y-%m-%d").date()
        end = datetime.strptime(end_iso, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (end - start).days


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _raw(df: pl.DataFrame, column: str) -> pl.Expr:
    """Raw column as Utf8, or a null literal if absent — so the decoder never
    crashes on a source file that omits an optional field."""
    if column in df.columns:
        return pl.col(column).cast(pl.Utf8)
    return pl.lit(None, dtype=pl.Utf8)


def _datasus_year(column_expr: pl.Expr) -> pl.Expr:
    """DATASUS dates are DDMMYYYY strings; the year is the trailing 4 digits."""
    s = column_expr.str.strip_chars()
    return (
        pl.when(s.str.len_chars() >= 8)
        .then(s.str.slice(4, 4).cast(pl.Int64, strict=False))
        .otherwise(None)
    )


def _cod6(column_expr: pl.Expr) -> pl.Expr:
    return column_expr.str.strip_chars().str.extract(r"(\d{6})", 1)


def _icd_norm(column_expr: pl.Expr) -> pl.Expr:
    return column_expr.str.replace_all(r"[^A-Za-z0-9]", "").str.to_uppercase()


def _icd_parse_state(norm_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(norm_expr.is_null() | (norm_expr.str.len_chars() == 0))
        .then(pl.lit("missing"))
        .when(norm_expr.str.contains(r"^[A-Z][0-9]{2,4}$"))
        .then(pl.lit("valid"))
        .otherwise(pl.lit("invalid"))
    )


def normalize_sim_do_events(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    """Real vectorized SIM-DO raw→canonical SHE decoder (MSD §2.4.0/§2.4.1).

    Replaces the previous per-row registry-routed path, which looked up *raw*
    DATASUS column names in a registry keyed by *canonical* names, matched
    nothing, and emitted 60 canonical columns that were 100% null. Here every
    canonical field is computed by a Polars expression over the raw columns,
    using the registered composite decoders (e.g. SIM IDADE) — fast and real."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = _read_table(input_path)
    n = df.height

    idade = _raw(df, "IDADE").str.strip_chars().str.zfill(3)
    u = idade.str.slice(0, 1).cast(pl.Int64, strict=False)
    m = idade.str.slice(1, 2).cast(pl.Int64, strict=False)
    age_years = (
        pl.when(u == 1).then(m / (24.0 * 365.25))
        .when(u == 2).then(m / 365.25)
        .when(u == 3).then(m / 12.0)
        .when(u == 4).then(m.cast(pl.Float64))
        .when(u == 5).then((100 + m).cast(pl.Float64))
        .otherwise(None)
    )
    age_days = (
        pl.when(u == 1).then(m / 24.0)
        .when(u == 2).then(m.cast(pl.Float64))
        .when(u == 3).then(30.4375 * m)
        .when(u == 4).then(365.25 * m)
        .when(u == 5).then(365.25 * (100 + m))
        .otherwise(None)
    )
    age_unit = (
        pl.when(u == 1).then(pl.lit("hours")).when(u == 2).then(pl.lit("days"))
        .when(u == 3).then(pl.lit("months")).when(u.is_in([4, 5])).then(pl.lit("years"))
        .otherwise(pl.lit("unknown"))
    )
    sex = (
        pl.when(_raw(df, "SEXO").str.strip_chars() == "1").then(pl.lit("male"))
        .when(_raw(df, "SEXO").str.strip_chars() == "2").then(pl.lit("female"))
        .otherwise(pl.lit("unknown"))
    )
    icd_norm = _icd_norm(_raw(df, "CAUSABAS"))

    out = df.with_row_index("_row").with_columns(
        pl.format("sim_{}_{}", pl.col("_row"), pl.lit(source_manifest_hash[:8])).alias("event_id"),
        pl.lit("SIM-DO").alias("source_system"),
        _datasus_year(_raw(df, "DTOBITO")).alias("year"),
        _raw(df, "DTOBITO").alias("death_date"),
        _raw(df, "DTNASC").alias("birth_date"),
        age_years.alias("age_years"),
        age_days.alias("age_days"),
        age_unit.alias("age_unit"),
        _raw(df, "IDADE").alias("raw_age_code"),
        sex.alias("sex"),
        _raw(df, "RACACOR").str.strip_chars().alias("race_color_admin"),
        pl.lit("administrative_death_declaration").alias("race_axis_type"),
        _cod6(_raw(df, "CODMUNRES")).alias("mun_residence_cod6"),
        _cod6(_raw(df, "CODMUNOCOR")).alias("mun_occurrence_cod6"),
        _raw(df, "CAUSABAS").alias("underlying_icd_raw"),
        icd_norm.alias("underlying_icd_norm"),
        _icd_parse_state(icd_norm).alias("underlying_icd_parse_state"),
        pl.lit(source_manifest_hash).alias("source_manifest_hash"),
    )
    for column in SIM_DO_NORMALIZED_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None).alias(column))
    out = out.select(SIM_DO_NORMALIZED_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": out.height,
        "column_count": len(out.columns),
        "columns": out.columns,
    }

# ---- Hardline MSD SHE registry-routed entrypoint ----
def normalize_sim_do_record(row: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
    registry_root = kwargs.get('registry_root', 'config/registries')
    return _registry_normalize_sim_do_record(row, registry_root=registry_root)

