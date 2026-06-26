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
    digits = _digits(raw)
    if digits is None:
        return None, "invalid"
    parsed = int(digits)
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
        "capacity_vector_json": json.dumps(capacity_values, ensure_ascii=False, sort_keys=True),
        "capacity_state_json": json.dumps(capacity_states, ensure_ascii=False, sort_keys=True),
        "capacity_total_observed": sum(v for v in capacity_values.values() if isinstance(v, int)),
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
    bed/infrastructure capacity vector (QTLEIT*/QTINST* as a JSON map plus the
    summed observed total), the boolean service-flag vector, and the invalid-flag
    count (values > 1 are InvalidFlagState, never coerced to true)."""
    from pegasus.datasus.normalize import _cod6, _raw

    df = _read_table(input_path)
    cols = df.columns
    capacity_cols = [c for c in cols if c.startswith(("QTLEIT", "QTINST"))]
    flag_cols = [
        c for c in cols
        if c.startswith(("GESPRG", "NIVATE", "SERAP"))
        or c in {"ATENDAMB", "ATENDHOS", "URGEMERG", "CENTRCIR", "CENTROBS", "LEITHOSP"}
    ]
    cap_int = [pl.col(c).cast(pl.Int64, strict=False).fill_null(0) for c in capacity_cols]
    capacity_total = pl.sum_horizontal(cap_int) if cap_int else pl.lit(0)
    # §2.4.0.4 boolean clamp: a flag value > 1 is an InvalidFlagState outlier.
    inv = [(pl.col(c).cast(pl.Int64, strict=False) > 1).cast(pl.Int64).fill_null(0) for c in flag_cols]
    invalid_count = pl.sum_horizontal(inv) if inv else pl.lit(0)

    columns: list[pl.Expr] = [
        _raw(df, "CNES").str.strip_chars().alias("facility_id"),
        _cod6(_raw(df, "CODUFMUN")).alias("mun_facility_cod6"),
        _raw(df, "COMPETEN").str.slice(0, 4).cast(pl.Int64, strict=False).alias("year"),
        capacity_total.cast(pl.Int64).alias("capacity_total_observed"),
        invalid_count.cast(pl.Int64).alias("invalid_flag_count"),
        pl.lit(source_manifest_hash).alias("source_manifest_hash"),
    ]
    columns.append(
        pl.struct(capacity_cols).struct.json_encode().alias("capacity_vector_json")
        if capacity_cols else pl.lit("{}").alias("capacity_vector_json")
    )
    columns.append(
        pl.struct(flag_cols).struct.json_encode().alias("flag_vector_json")
        if flag_cols else pl.lit("{}").alias("flag_vector_json")
    )
    out = df.with_columns(columns).select(
        ["facility_id", "mun_facility_cod6", "year", "capacity_total_observed",
         "capacity_vector_json", "flag_vector_json", "invalid_flag_count", "source_manifest_hash"]
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return {"row_count": out.height, "output_path": str(out_path), "column_count": len(out.columns), "columns": out.columns}
