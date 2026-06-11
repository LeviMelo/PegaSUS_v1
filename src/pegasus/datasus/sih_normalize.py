from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.datasus.decoders import decode_sih_age
from pegasus.datasus.icd_parser import parse_icd
from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7

SECONDARY_DIAG_COLUMNS = tuple(f"DIAGSEC{i}" for i in range(1, 10))
SECONDARY_TYPE_COLUMNS = tuple(f"TPDISEC{i}" for i in range(1, 10))
COST_COMPONENTS = ("VAL_SH", "VAL_SP", "VAL_UTI", "VAL_TOT")


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
    raise ValueError(f"Unsupported SIH-RD input format: {path}")


def _parse_date(value: Any) -> tuple[str | None, int | None, str]:
    text = _clean(value)
    if text is None:
        return None, None, "missing"
    digits = _digits(text)
    candidates: list[str] = []
    if digits and len(digits) == 8:
        candidates.append(f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}")
        candidates.append(f"{digits[4:8]}-{digits[2:4]}-{digits[0:2]}")
    candidates.append(text)
    for candidate in candidates:
        try:
            d = datetime.fromisoformat(candidate).date()
            return d.isoformat(), d.year, "valid"
        except Exception:
            pass
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


def _number(value: Any) -> tuple[float | None, str, str | None]:
    raw = _clean(value)
    if raw is None:
        return None, "missing", None
    try:
        parsed = float(raw.replace(",", "."))
    except ValueError:
        return None, "invalid", raw
    return (parsed, "valid", raw) if parsed >= 0 else (None, "invalid", raw)


def _int_nonnegative(value: Any) -> tuple[int | None, str]:
    number, state, _ = _number(value)
    return (int(number), state) if number is not None else (None, state)


def _death_flag(value: Any) -> tuple[bool | None, str]:
    text = _clean(value)
    if text is None:
        return None, "missing"
    norm = text.casefold()
    if norm in {"1", "sim", "yes", "true", "t", "morte", "death"}:
        return True, "valid"
    if norm in {"0", "não", "nao", "no", "false", "f"}:
        return False, "valid"
    return None, "invalid"


def normalize_sih_rd_record(row: dict[str, Any], *, source_manifest_hash: str) -> dict[str, Any]:
    raw_payload = {str(k): v for k, v in row.items()}
    admission_date, admission_year, admission_state = _parse_date(row.get("DT_INTER") or row.get("DTINTERN") or row.get("admission_date"))
    discharge_date, _, discharge_state = _parse_date(row.get("DT_SAIDA") or row.get("DTSAIDA") or row.get("discharge_date"))
    cod6, cod7, mun_state = _mun(row.get("MUNIC_RES") or row.get("CODMUNRES") or row.get("municipality"))
    age = decode_sih_age(row.get("COD_IDADE") or row.get("CODIDADE") or "4", row.get("IDADE") or row.get("age"))
    principal = parse_icd(row.get("DIAG_PRINC") or row.get("principal_icd"), topology_role="sih_principal_diagnosis", source_field="DIAG_PRINC")
    secondary_raw: dict[str, Any] = {}
    secondary_norm: dict[str, Any] = {}
    secondary_states: dict[str, Any] = {}
    for col in SECONDARY_DIAG_COLUMNS:
        if col in row:
            parsed = parse_icd(row.get(col), topology_role="sih_secondary_diagnosis", source_field=col)
            secondary_raw[col] = None if row.get(col) is None else str(row.get(col))
            secondary_norm[col] = parsed.normalized
            secondary_states[col] = parsed.parse_state
    secondary_type = {col: _clean(row.get(col)) for col in SECONDARY_TYPE_COLUMNS if col in row}
    stay_days, stay_state = _int_nonnegative(row.get("DIAS_PERM") or row.get("QT_DIARIAS") or row.get("stay_length_days"))
    icu_days_month, icu_month_state = _int_nonnegative(row.get("UTI_MES_TO"))
    icu_days_adm, icu_adm_state = _int_nonnegative(row.get("UTI_INT_TO"))
    death, death_state = _death_flag(row.get("MORTE") or row.get("OBITO"))
    costs: dict[str, float | None] = {}
    cost_states: dict[str, str] = {}
    cost_raw: dict[str, str | None] = {}
    for col in COST_COMPONENTS:
        value, state, raw = _number(row.get(col))
        costs[col] = value
        cost_states[col] = state
        cost_raw[col] = raw
    admission_id = _clean(row.get("AIH") or row.get("N_AIH") or row.get("admission_id")) or _stable_hash(raw_payload)[:16]
    record_state = "valid"
    if admission_year is None or cod6 is None or principal.parse_state in {"missing", "blank", "unparseable", "invalid"}:
        record_state = "invalid_identity_or_principal_diagnosis"
    return {
        "admission_id": f"SIH-{admission_id}",
        "source_system": "SIH-RD",
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "admission_year": admission_year,
        "admission_date_state": admission_state,
        "discharge_date_state": discharge_state,
        "mun_residence_cod6": cod6,
        "mun_residence_cod7": cod7,
        "municipality_code_state": mun_state,
        "age_days": age.age_days,
        "age_years": age.age_years,
        "age_unit": age.age_unit,
        "age_state": age.state,
        "sex": _clean(row.get("SEXO")),
        "race_color_billing": _clean(row.get("RACA_COR")),
        "race_axis_type": "sih_billing_race_color",
        "principal_icd_raw": principal.raw,
        "principal_icd_norm": principal.normalized,
        "principal_icd_parse_state": principal.parse_state,
        "secondary_icd_raw_json": json.dumps(secondary_raw, ensure_ascii=False, sort_keys=True),
        "secondary_icd_norm_json": json.dumps(secondary_norm, ensure_ascii=False, sort_keys=True),
        "secondary_icd_parse_states_json": json.dumps(secondary_states, ensure_ascii=False, sort_keys=True),
        "secondary_diagnosis_type_json": json.dumps(secondary_type, ensure_ascii=False, sort_keys=True),
        "procedure_requested": _clean(row.get("PROC_SOLIC")),
        "procedure_performed": _clean(row.get("PROC_REA")),
        "stay_length_days": stay_days,
        "stay_length_state": stay_state,
        "icu_type_mark": _clean(row.get("MARCA_UTI")),
        "icu_days_month_total": icu_days_month,
        "icu_days_month_state": icu_month_state,
        "icu_days_hospitalization_total": icu_days_adm,
        "icu_days_hospitalization_state": icu_adm_state,
        "death_flag": death,
        "death_flag_state": death_state,
        "hospital_service_cost_real": costs["VAL_SH"],
        "professional_service_cost_real": costs["VAL_SP"],
        "icu_cost_real": costs["VAL_UTI"],
        "total_admission_cost_real": costs["VAL_TOT"],
        "cost_state_json": json.dumps(cost_states, ensure_ascii=False, sort_keys=True),
        "cost_raw_json": json.dumps(cost_raw, ensure_ascii=False, sort_keys=True),
        "record_state": record_state,
        "source_manifest_hash": source_manifest_hash,
        "row_hash": _stable_hash(raw_payload),
        "raw_json": json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str),
    }


def normalize_sih_rd_events(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, Any]:
    rows = [normalize_sih_rd_record(row, source_manifest_hash=source_manifest_hash) for row in _read_table(input_path).to_dicts()]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)
    return {
        "row_count": len(rows),
        "output_path": str(out),
        "valid_rows": sum(1 for r in rows if r["record_state"] == "valid"),
        "deaths": sum(1 for r in rows if r["death_flag"] is True),
        "principal_diagnosis_states": sorted({r["principal_icd_parse_state"] for r in rows}),
        "cost_components": list(COST_COMPONENTS),
    }
