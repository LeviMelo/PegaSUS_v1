from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.datasus.decoders import clamp_bool, filter_cnpj
from pegasus.datasus.vec import Cols, read_raw_table, row_hash, struct_json
from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7, load_municipality_crosswalk

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
    return read_raw_table(path)


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


def _cnes_vectorized_frame(df: pl.DataFrame, *, source_manifest_hash: str) -> pl.DataFrame:
    """Fully vectorized CNES-ST raw→canonical decode (MSD §2.4.4, §2.4.0.4).

    Produces the same canonical schema as ``normalize_cnes_st_record`` — typed
    QTLEIT bed primitives, the full capacity/flag vectors as JSON, and the
    ``filter_cnpj`` §2.4.0.5 linkage gate — via the shared ``vec.Cols`` primitives
    (single source of truth, XCUT-02) instead of a per-row Python loop. Only the
    CNES-specific COMPETEN period split lives here.
    """
    cx = Cols(df)
    capacity_cols = _capacity_columns(list(df.columns))
    flag_cols = _flag_columns(list(df.columns))

    # COMPETEN year/month split (CNES-specific): YYYYMM → (year, month) valid;
    # 4-digit → year_only; else invalid.
    comp = cx.digits("COMPETEN", "ANO_CMPT", "year")
    year_full = comp.str.slice(0, 4).cast(pl.Int64, strict=False)
    month_full = comp.str.slice(4, 2).cast(pl.Int64, strict=False)
    has6 = comp.str.len_chars() >= 6
    month_ok = has6 & (month_full >= 1) & (month_full <= 12)
    year = pl.when(month_ok).then(year_full).when(comp.str.len_chars() == 4).then(comp.cast(pl.Int64, strict=False)).otherwise(None)
    month = pl.when(month_ok).then(month_full).otherwise(None)
    period_state = (
        pl.when(comp.is_null()).then(pl.lit("missing"))
        .when(month_ok).then(pl.lit("valid"))
        .when(comp.str.len_chars() == 4).then(pl.lit("year_only"))
        .otherwise(pl.lit("invalid"))
    )

    cap_val = {col: cx.nonneg_int(col) for col in capacity_cols}
    fac_cnpj, fac_cnpj_state = cx.cnpj("CPF_CNPJ", "facility_cnpj")
    man_cnpj, man_cnpj_state = cx.cnpj("CNPJ_MAN", "maintainer_cnpj")
    clin_v, clin_s = cx.nonneg_int("QTLEITP1")
    surg_v, surg_s = cx.nonneg_int("QTLEITP2")
    obst_v, obst_s = cx.nonneg_int("QTLEITP3")
    flag = {col: cx.flag_int(col) for col in flag_cols}

    invalid_flag_count = (
        sum((flag[col][1].is_in(["InvalidFlagState", "UnparseableFlag"]).cast(pl.Int64) for col in flag_cols), pl.lit(0))
        if flag_cols else pl.lit(0)
    )
    facility = cx.clean("CNES", "facility_id")
    mun = cx.municipality("mun_facility", "CODMUN", "MUNIC_RES", "facility_municipality", crosswalk=load_municipality_crosswalk())

    out = df.with_row_index("_i").with_columns(
        facility.alias("facility_id"),
        pl.lit("CNES-ST").alias("source_system"),
        year.alias("year"),
        month.alias("month"),
        period_state.alias("period_state"),
        *mun,
        fac_cnpj.alias("facility_cnpj"), fac_cnpj_state.alias("facility_cnpj_state"),
        man_cnpj.alias("maintainer_cnpj"), man_cnpj_state.alias("maintainer_cnpj_state"),
        clin_v.alias("clinical_bed_capacity"), clin_s.alias("clinical_bed_capacity_state"),
        surg_v.alias("surgical_bed_capacity"), surg_s.alias("surgical_bed_capacity_state"),
        obst_v.alias("obstetric_bed_capacity"), obst_s.alias("obstetric_bed_capacity_state"),
        struct_json([cap_val[col][0].alias(col.upper()) for col in capacity_cols]).alias("capacity_vector_json"),
        struct_json([cap_val[col][1].alias(col.upper()) for col in capacity_cols]).alias("capacity_state_json"),
        struct_json([flag[col][0].alias(col.upper()) for col in flag_cols]).alias("flag_vector_json"),
        struct_json([flag[col][1].alias(col.upper()) for col in flag_cols]).alias("flag_state_json"),
        invalid_flag_count.alias("invalid_flag_count"),
        pl.lit(len(flag_cols)).cast(pl.Int64).alias("flag_count"),
        pl.lit(source_manifest_hash).alias("source_manifest_hash"),
        pl.lit(None, dtype=pl.Utf8).alias("raw_json"),
    ).with_columns(
        pl.when(facility.is_not_null() & year.is_not_null() & pl.col("mun_facility_cod6").is_not_null())
          .then(pl.lit("valid")).otherwise(pl.lit("invalid_identity")).alias("record_state"),
        row_hash(source_manifest_hash).alias("row_hash"),
    )
    ordered = [
        "facility_id", "source_system", "year", "month", "period_state",
        "mun_facility_cod6", "mun_facility_cod7", "municipality_code_state",
        "facility_cnpj", "facility_cnpj_state", "maintainer_cnpj", "maintainer_cnpj_state",
        "clinical_bed_capacity", "clinical_bed_capacity_state",
        "surgical_bed_capacity", "surgical_bed_capacity_state",
        "obstetric_bed_capacity", "obstetric_bed_capacity_state",
        "capacity_vector_json", "capacity_state_json", "flag_vector_json", "flag_state_json",
        "invalid_flag_count", "flag_count", "record_state",
        "source_manifest_hash", "row_hash", "raw_json",
    ]
    return out.rename({"mun_facility_state": "municipality_code_state"}).select(ordered)


def normalize_cnes_st_events(
    *, input_path: str | Path, output_path: str | Path, source_manifest_hash: str
) -> dict[str, Any]:
    """Batch CNES-ST raw→canonical SHE normalizer (MSD §2.4.4, §2.4.0.4), fully
    vectorized via ``_cnes_vectorized_frame`` (shared ``vec.Cols`` primitives).

    Emits the typed QTLEIT bed primitives plus the full capacity/flag vectors as
    JSON, the ``clamp_bool`` service-flag semantics (values outside {0,1} are
    InvalidFlagState, never coerced to true), and the ``filter_cnpj`` §2.4.0.5
    linkage gate. ``normalize_cnes_st_record`` is retained as the record-level
    correctness oracle the equivalence stress-check pins against.
    """
    frame = _cnes_vectorized_frame(_read_table(input_path), source_manifest_hash=source_manifest_hash)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    capacity_components: set[str] = set()
    if frame.height:
        try:
            capacity_components.update(json.loads(frame["capacity_vector_json"][0]).keys())
        except (ValueError, TypeError):
            pass
    return {
        "row_count": frame.height,
        "output_path": str(out_path),
        "column_count": len(frame.columns),
        "columns": frame.columns,
        "zero_facility_cnpj_rows": int((frame["facility_cnpj_state"] == "NullifiedZeroCNPJ").sum()),
        "invalid_flag_rows": int((frame["invalid_flag_count"] > 0).sum()),
        "capacity_components": sorted(capacity_components),
    }
