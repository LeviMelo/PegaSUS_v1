from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.datasus.decoders import clamp_bool, filter_cnpj
from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7

CAPACITY_PREFIXES = ("QTINST", "QTLEIT")
FLAG_PREFIXES = ("GESPRG", "SERAP")
FLAG_NAMES = {"NIVATE_A", "NIVATE_H", "ATENDAMB", "ATENDHOS", "URGEMERG", "CENTRCIR", "CENTROBS", "LEITHOSP"}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return None
    return text


def _digits(value: Any) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or None


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pl.read_csv(path, infer_schema_length=0, ignore_errors=False)
    raise ValueError(f"Unsupported CNES-ST input format: {path}")


def _period(value: Any) -> tuple[int | None, int | None, str]:
    digits = _digits(value)
    if digits is None:
        return None, None, "missing"
    if len(digits) >= 6:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= month <= 12:
            return year, month, "valid"
    if len(digits) == 4:
        return int(digits), None, "year_only"
    return None, None, "invalid"


def _mun(value: Any) -> tuple[str | None, str | None, str]:
    digits = _digits(value)
    if digits is None:
        return None, None, "missing"
    if len(digits) == 6:
        try:
            return digits, datasus_cod6_to_ibge_cod7(digits, strict=True), "datasus_cod6"
        except Exception:
            return digits, None, "unmapped_datasus_cod6"
    if len(digits) == 7:
        return digits[:6], digits, "ibge_cod7"
    return None, None, "invalid"


def _int_nonnegative(value: Any) -> tuple[int | None, str]:
    raw = _clean(value)
    if raw is None:
        return None, "missing"
    # Parse via float to preserve sign information. _digits() strips "-" and
    # would silently convert "-5" → 5 (capacity fields cannot be negative).
    try:
        f = float(raw.replace(",", "."))
    except ValueError:
        return None, "invalid"
    if f != int(f):
        return None, "invalid"
    parsed = int(f)
    return (parsed, "valid") if parsed >= 0 else (None, "invalid")


def _capacity_columns(columns: list[str]) -> list[str]:
    return sorted(c for c in columns if c.upper().startswith(CAPACITY_PREFIXES))


def _flag_columns(columns: list[str]) -> list[str]:
    return sorted(c for c in columns if c.upper() in FLAG_NAMES or c.upper().startswith(FLAG_PREFIXES))


def normalize_cnes_st_record(row: dict[str, Any], *, source_manifest_hash: str) -> dict[str, Any]:
    raw_payload = {str(k): v for k, v in row.items()}
    year, month, period_state = _period(row.get("COMPETEN") or row.get("ANO_CMPT") or row.get("year"))
    cod6, cod7, mun_state = _mun(row.get("CODMUN") or row.get("MUNIC_RES") or row.get("facility_municipality"))
    facility = _clean(row.get("CNES") or row.get("facility_id"))
    cnpj = filter_cnpj(row.get("CPF_CNPJ") or row.get("facility_cnpj"))
    cnpj_man = filter_cnpj(row.get("CNPJ_MAN") or row.get("maintainer_cnpj"))
    capacity_values: dict[str, int | None] = {}
    capacity_states: dict[str, str] = {}
    for col in _capacity_columns(list(row)):
        value, state = _int_nonnegative(row.get(col))
        capacity_values[col.upper()] = value
        capacity_states[col.upper()] = state
    flag_values: dict[str, int | None] = {}
    flag_states: dict[str, str] = {}
    for col in _flag_columns(list(row)):
        decoded = clamp_bool(row.get(col))
        flag_values[col.upper()] = decoded.value
        flag_states[col.upper()] = decoded.state
    invalid_flags = sum(1 for state in flag_states.values() if state in {"InvalidFlagState", "UnparseableFlag"})
    identity_state = "valid" if facility is not None and year is not None and cod6 is not None else "invalid_identity"
    return {
        "facility_id": facility,
        "source_system": "CNES-ST",
        "year": year,
        "month": month,
        "period_state": period_state,
        "mun_facility_cod6": cod6,
        "mun_facility_cod7": cod7,
        "municipality_code_state": mun_state,
        "facility_cnpj": cnpj.cnpj,
        "facility_cnpj_state": cnpj.state,
        "maintainer_cnpj": cnpj_man.cnpj,
        "maintainer_cnpj_state": cnpj_man.state,
        "clinical_bed_capacity": capacity_values.get("QTLEITP1"),
        "clinical_bed_capacity_state": capacity_states.get("QTLEITP1", "missing"),
        "surgical_bed_capacity": capacity_values.get("QTLEITP2"),
        "surgical_bed_capacity_state": capacity_states.get("QTLEITP2", "missing"),
        "obstetric_bed_capacity": capacity_values.get("QTLEITP3"),
        "obstetric_bed_capacity_state": capacity_states.get("QTLEITP3", "missing"),
        "capacity_vector_json": json.dumps(capacity_values, ensure_ascii=False, sort_keys=True),
        "capacity_state_json": json.dumps(capacity_states, ensure_ascii=False, sort_keys=True),
        "flag_vector_json": json.dumps(flag_values, ensure_ascii=False, sort_keys=True),
        "flag_state_json": json.dumps(flag_states, ensure_ascii=False, sort_keys=True),
        "invalid_flag_count": invalid_flags,
        "flag_count": len(flag_states),
        "record_state": identity_state,
        "source_manifest_hash": source_manifest_hash,
        "row_hash": _stable_hash(raw_payload),
        "raw_json": json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str),
    }


def normalize_cnes_st_events(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, Any]:
    """Real vectorized CNES-ST raw→canonical SHE decoder (MSD §2.4.4, §2.4.0.4).

    Facility stock: facility id, facility municipality, competence year, the full
    bed/infrastructure capacity vector (QTLEIT*/QTINST* as a JSON map), typed
    bed primitives for the MSD-named QTLEIT indices, the boolean service-flag
    vector, and the invalid-flag count (values > 1 are InvalidFlagState, never
    coerced to true)."""
    df = _read_table(input_path).with_row_index("_row_idx")
    capacity_columns = _capacity_columns(df.columns)
    flag_columns = _flag_columns(df.columns)

    def col(name: str) -> pl.Expr:
        return pl.col(name) if name in df.columns else pl.lit(None)

    def first(*names: str) -> pl.Expr:
        return pl.coalesce([col(name) for name in names])

    def clean_expr(expr: pl.Expr) -> pl.Expr:
        text = expr.cast(pl.Utf8).str.strip_chars()
        return pl.when(text.str.to_uppercase().is_in(["", "NA", "NAN", "NULL", "NONE"])).then(None).otherwise(text)

    def clean(*names: str) -> pl.Expr:
        return clean_expr(first(*names))

    def digits_expr(expr: pl.Expr) -> pl.Expr:
        digits = clean_expr(expr).str.replace_all(r"\D", "")
        return pl.when(digits == "").then(None).otherwise(digits)

    def digits(*names: str) -> pl.Expr:
        return digits_expr(first(*names))

    def period_expr(value: pl.Expr) -> list[pl.Expr]:
        d = digits_expr(value)
        year6 = d.str.slice(0, 4).cast(pl.Int64, strict=False)
        month6 = d.str.slice(4, 2).cast(pl.Int64, strict=False)
        year4 = d.cast(pl.Int64, strict=False)
        valid6 = (d.str.len_chars() >= 6) & month6.is_between(1, 12)
        return [
            pl.when(valid6).then(year6)
            .when(d.str.len_chars() == 4).then(year4)
            .otherwise(None)
            .alias("year"),
            pl.when(valid6).then(month6).otherwise(None).alias("month"),
            pl.when(d.is_null()).then(pl.lit("missing"))
            .when(valid6).then(pl.lit("valid"))
            .when(d.str.len_chars() == 4).then(pl.lit("year_only"))
            .otherwise(pl.lit("invalid"))
            .alias("period_state"),
        ]

    def municipality_expr(value: pl.Expr) -> list[pl.Expr]:
        d = digits_expr(value)
        cod6 = (
            pl.when(d.str.len_chars() == 6).then(d)
            .when(d.str.len_chars() == 7).then(d.str.slice(0, 6))
            .otherwise(None)
        )
        ignored = cod6.str.slice(2, 4) == "0000"
        valid = cod6.is_not_null() & ~ignored
        return [
            pl.when(valid).then(cod6).otherwise(None).alias("mun_facility_cod6"),
            pl.when(valid & (d.str.len_chars() == 7)).then(d)
            .when(valid).then(
                cod6.map_elements(lambda value: datasus_cod6_to_ibge_cod7(value, strict=False) if value else None, return_dtype=pl.Utf8)
            )
            .otherwise(None)
            .alias("mun_facility_cod7"),
            pl.when(d.is_null()).then(pl.lit("missing"))
            .when(ignored).then(pl.lit("ignored_municipality"))
            .when(d.str.len_chars() == 7).then(pl.lit("ibge_cod7"))
            .when(d.str.len_chars() == 6).then(pl.lit("datasus_cod6"))
            .otherwise(pl.lit("invalid"))
            .alias("municipality_code_state"),
        ]

    def cnpj_expr(prefix: str, value: pl.Expr) -> list[pl.Expr]:
        cleaned = clean_expr(value)
        return [
            cleaned.map_elements(lambda raw: filter_cnpj(raw).cnpj, return_dtype=pl.Utf8).alias(f"{prefix}_cnpj"),
            cleaned.map_elements(lambda raw: filter_cnpj(raw).state, return_dtype=pl.Utf8).alias(f"{prefix}_cnpj_state"),
        ]

    def nonnegative_expr(name: str) -> tuple[pl.Expr, pl.Expr]:
        raw = clean(name)
        value = raw.str.replace_all(",", ".").cast(pl.Float64, strict=False)
        valid = value.is_not_null() & (value >= 0) & (value == value.floor())
        parsed = pl.when(valid).then(value.cast(pl.Int64)).otherwise(None)
        state = (
            pl.when(raw.is_null()).then(pl.lit("missing"))
            .when(valid).then(pl.lit("valid"))
            .otherwise(pl.lit("invalid"))
        )
        return parsed.alias(name.upper()), state.alias(f"{name.upper()}_state")

    capacity_value_exprs: list[pl.Expr] = []
    capacity_state_exprs: list[pl.Expr] = []
    for capacity_column in capacity_columns:
        value_expr, state_expr = nonnegative_expr(capacity_column)
        capacity_value_exprs.append(value_expr)
        capacity_state_exprs.append(state_expr)

    def json_capacity_values(row: dict[str, Any]) -> str:
        return json.dumps({key: row.get(key) for key in sorted(row)}, ensure_ascii=False, sort_keys=True)

    def json_capacity_states(row: dict[str, Any]) -> str:
        return json.dumps({key.removesuffix("_state"): row.get(key) for key in sorted(row)}, ensure_ascii=False, sort_keys=True)

    def json_flag_values(row: dict[str, Any]) -> str:
        return json.dumps({key: clamp_bool(value).value for key, value in sorted(row.items())}, ensure_ascii=False, sort_keys=True)

    def json_flag_states(row: dict[str, Any]) -> str:
        return json.dumps({key: clamp_bool(value).state for key, value in sorted(row.items())}, ensure_ascii=False, sort_keys=True)

    def invalid_flag_count(row: dict[str, Any]) -> int:
        return sum(1 for value in row.values() if clamp_bool(value).state in {"InvalidFlagState", "UnparseableFlag"})

    flag_struct = pl.struct([clean(column).alias(column.upper()) for column in flag_columns]) if flag_columns else pl.struct([])
    capacity_value_struct = pl.struct(capacity_value_exprs) if capacity_value_exprs else pl.struct([])
    capacity_state_struct = pl.struct(capacity_state_exprs) if capacity_state_exprs else pl.struct([])

    out = (
        df.with_columns(
            clean("CNES", "facility_id").alias("facility_id"),
            pl.lit("CNES-ST").alias("source_system"),
            *period_expr(first("COMPETEN", "ANO_CMPT", "year")),
            *municipality_expr(first("CODMUN", "MUNIC_RES", "facility_municipality")),
            *cnpj_expr("facility", first("CPF_CNPJ", "facility_cnpj")),
            *cnpj_expr("maintainer", first("CNPJ_MAN", "maintainer_cnpj")),
            *capacity_value_exprs,
            *capacity_state_exprs,
            capacity_value_struct.map_elements(json_capacity_values, return_dtype=pl.Utf8).alias("capacity_vector_json"),
            capacity_state_struct.map_elements(json_capacity_states, return_dtype=pl.Utf8).alias("capacity_state_json"),
            flag_struct.map_elements(json_flag_values, return_dtype=pl.Utf8).alias("flag_vector_json"),
            flag_struct.map_elements(json_flag_states, return_dtype=pl.Utf8).alias("flag_state_json"),
            flag_struct.map_elements(invalid_flag_count, return_dtype=pl.Int64).alias("invalid_flag_count"),
            pl.lit(len(flag_columns)).alias("flag_count"),
        )
        .with_columns(
            pl.col("QTLEITP1").alias("clinical_bed_capacity") if "QTLEITP1" in capacity_columns else pl.lit(None, dtype=pl.Int64).alias("clinical_bed_capacity"),
            pl.col("QTLEITP1_state").alias("clinical_bed_capacity_state") if "QTLEITP1" in capacity_columns else pl.lit("missing").alias("clinical_bed_capacity_state"),
            pl.col("QTLEITP2").alias("surgical_bed_capacity") if "QTLEITP2" in capacity_columns else pl.lit(None, dtype=pl.Int64).alias("surgical_bed_capacity"),
            pl.col("QTLEITP2_state").alias("surgical_bed_capacity_state") if "QTLEITP2" in capacity_columns else pl.lit("missing").alias("surgical_bed_capacity_state"),
            pl.col("QTLEITP3").alias("obstetric_bed_capacity") if "QTLEITP3" in capacity_columns else pl.lit(None, dtype=pl.Int64).alias("obstetric_bed_capacity"),
            pl.col("QTLEITP3_state").alias("obstetric_bed_capacity_state") if "QTLEITP3" in capacity_columns else pl.lit("missing").alias("obstetric_bed_capacity_state"),
            pl.when(pl.col("facility_id").is_not_null() & pl.col("year").is_not_null() & pl.col("mun_facility_cod6").is_not_null())
            .then(pl.lit("valid"))
            .otherwise(pl.lit("invalid_identity"))
            .alias("record_state"),
            pl.lit(source_manifest_hash).alias("source_manifest_hash"),
            pl.concat_str([pl.lit(source_manifest_hash), pl.lit(":"), pl.col("_row_idx").cast(pl.Utf8)]).hash().cast(pl.Utf8).alias("row_hash"),
            pl.lit(None, dtype=pl.Utf8).alias("raw_json"),
        )
        .select(
            [
                "facility_id",
                "source_system",
                "year",
                "month",
                "period_state",
                "mun_facility_cod6",
                "mun_facility_cod7",
                "municipality_code_state",
                "facility_cnpj",
                "facility_cnpj_state",
                "maintainer_cnpj",
                "maintainer_cnpj_state",
                "clinical_bed_capacity",
                "clinical_bed_capacity_state",
                "surgical_bed_capacity",
                "surgical_bed_capacity_state",
                "obstetric_bed_capacity",
                "obstetric_bed_capacity_state",
                "capacity_vector_json",
                "capacity_state_json",
                "flag_vector_json",
                "flag_state_json",
                "invalid_flag_count",
                "flag_count",
                "record_state",
                "source_manifest_hash",
                "row_hash",
                "raw_json",
            ]
        )
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return {
        "row_count": out.height,
        "output_path": str(out_path),
        "column_count": len(out.columns),
        "columns": out.columns,
        "zero_facility_cnpj_rows": int((out.get_column("facility_cnpj_state") == "NullifiedZeroCNPJ").sum()) if out.height else 0,
        "invalid_flag_rows": int((out.get_column("invalid_flag_count") > 0).sum()) if out.height else 0,
        "capacity_components": capacity_columns,
    }
