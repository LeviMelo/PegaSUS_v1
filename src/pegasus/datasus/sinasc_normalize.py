from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

ICD_LIKE = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]?")


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _digits(value: Any) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or None


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=0, ignore_errors=False)
    raise ValueError(f"Unsupported SINASC input format: {path}")


def _raw(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def parse_sinasc_date(value: Any) -> tuple[str | None, int | None, str]:
    text = _clean(value)
    if text is None:
        return None, None, "missing"
    digits = _digits(text)
    candidates: list[str] = []
    if digits and len(digits) == 8:
        # SINASC DBF exports may appear either as YYYYMMDD-like or DDMMYYYY-like strings.
        candidates.append(f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}")
        candidates.append(f"{digits[4:8]}-{digits[2:4]}-{digits[0:2]}")
    candidates.append(text)
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate).date()
            return dt.isoformat(), dt.year, "valid"
        except Exception:
            pass
    return None, None, "invalid"


def municipality_codes(value: Any) -> tuple[str | None, str | None, str]:
    digits = _digits(value)
    if digits is None:
        return None, None, "missing"
    if len(digits) == 6:
        return digits, None, "datasus_cod6"
    if len(digits) == 7:
        return digits[:6], digits, "ibge_cod7"
    return None, None, "invalid"


def int_or_none(value: Any) -> int | None:
    digits = _digits(value)
    if digits is None:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def decode_count_preserve_leading_zero(
    value: Any,
    *,
    sentinels: set[str] | None = None,
    upper: int | None = None,
) -> tuple[int | None, str, str | None]:
    raw = _clean(value)
    if raw is None:
        return None, "missing", None
    sentinels = sentinels or set()
    digits = _digits(raw)
    if digits is None:
        return None, "invalid", raw
    if digits in sentinels:
        return None, "sentinel", digits
    value_int = int(digits)
    if upper is not None and value_int > upper:
        return None, "invalid", digits
    return value_int, "valid", digits


def decode_birth_weight(value: Any) -> tuple[int | None, str, bool | None]:
    weight = int_or_none(value)
    if weight is None:
        return None, "missing", None
    if weight in {0, 9999}:
        return None, "sentinel", None
    if weight < 300 or weight > 7000:
        return None, "invalid", None
    return weight, "valid", weight < 2500


def decode_gestational_age(value: Any) -> tuple[int | None, str, bool | None]:
    weeks = int_or_none(value)
    if weeks is None:
        return None, "missing", None
    if weeks in {0, 99}:
        return None, "sentinel", None
    if weeks < 20 or weeks > 45:
        return None, "invalid", None
    return weeks, "valid", weeks < 37


def decode_apgar(value: Any) -> tuple[int | None, str, bool | None]:
    score = int_or_none(value)
    if score is None:
        return None, "missing", None
    if score == 99:
        return None, "sentinel", None
    if score < 0 or score > 10:
        return None, "invalid", None
    return score, "valid", score < 7


def decode_delivery_mode(value: Any) -> tuple[str | None, str, bool | None]:
    code = _digits(value)
    if code is None:
        return None, "missing", None
    if code == "9":
        return code, "sentinel", None
    if code == "1":
        return code, "valid", False
    if code == "2":
        return code, "valid", True
    return code, "invalid", None


def decode_race(value: Any) -> tuple[str | None, str]:
    code = _digits(value)
    if code is None:
        return None, "missing"
    if code in {"1", "2", "3", "4", "5"}:
        return code, "valid_admin_race"
    if code == "9":
        return None, "ignored_sentinel"
    return code, "invalid"


def decode_anomaly_flag(value: Any) -> tuple[bool | None, str]:
    code = _digits(value)
    if code is None:
        return None, "missing"
    if code == "1":
        return False, "valid_absent"
    if code == "2":
        return True, "valid_present"
    if code == "9":
        return None, "sentinel"
    return None, "invalid"


def normalize_anomaly_icd(value: Any, anomaly_flag: bool | None) -> tuple[str | None, str, bool]:
    text = _clean(value)
    if text is None:
        if anomaly_flag is True:
            return None, "flag_present_code_missing", True
        if anomaly_flag is False:
            return None, "absent", False
        return None, "unknown", False
    token = re.split(r"[;|,\s]+", text.upper().strip())[0]
    match = ICD_LIKE.match(token)
    if not match:
        return None, "invalid", bool(anomaly_flag)
    code = match.group(0)
    return code, "valid", code.startswith("Q") or bool(anomaly_flag)


def normalize_sinasc_record(row: dict[str, Any], *, source_manifest_hash: str) -> dict[str, Any]:
    raw_payload = {str(k): v for k, v in row.items()}
    birth_date, birth_year, birth_date_state = parse_sinasc_date(_raw(row, "DTNASC", "DT_NASC", "NASCIMENTO"))
    mun_cod6, mun_cod7, mun_state = municipality_codes(_raw(row, "CODMUNRES", "CODMUNNASC", "MUN_RES"))
    weight, weight_state, low_weight = decode_birth_weight(_raw(row, "PESO", "PESO_NASC"))
    gest_weeks, gest_state, preterm = decode_gestational_age(_raw(row, "SEMAGESTAC", "GESTACAO", "QTSEMANAS"))
    apgar1, apgar1_state, low_apgar1 = decode_apgar(_raw(row, "APGAR1", "APGAR_1"))
    apgar5, apgar5_state, low_apgar5 = decode_apgar(_raw(row, "APGAR5", "APGAR_5"))
    delivery_code, delivery_state, cesarean = decode_delivery_mode(_raw(row, "PARTO", "TPPARTO"))
    consultations, consultations_state, consultations_raw = decode_count_preserve_leading_zero(
        _raw(row, "CONSULTAS", "QTCONSULTAS", "CONSPRENAT"),
        sentinels={"99"},
        upper=40,
    )
    mother_age = int_or_none(_raw(row, "IDADEMAE", "IDADE_MAE"))
    mother_race, mother_race_state = decode_race(_raw(row, "RACACORMAE", "RACA_COR_MAE"))
    newborn_race, newborn_race_state = decode_race(_raw(row, "RACACOR", "RACA_COR"))
    anomaly_flag, anomaly_flag_state = decode_anomaly_flag(_raw(row, "IDANOMAL", "ANOMALIA_FLAG"))
    anomaly_code, anomaly_state, anomaly_any = normalize_anomaly_icd(_raw(row, "CODANOMAL", "ANOMALIA"), anomaly_flag)

    event_key = _clean(_raw(row, "NUMERODN", "DN", "ID"))
    if event_key is None:
        event_key = _stable_hash(raw_payload)[:16]
    event_id = f"SINASC-{event_key}"

    record_state = "valid"
    if birth_date_state == "invalid" or mun_state == "invalid" or birth_year is None or mun_cod6 is None:
        record_state = "invalid_identity"

    return {
        "event_id": event_id,
        "birth_date": birth_date,
        "birth_year": birth_year,
        "birth_date_state": birth_date_state,
        "mun_residence_cod6": mun_cod6,
        "mun_residence_cod7": mun_cod7,
        "municipality_code_state": mun_state,
        "mother_age_years": mother_age,
        "adolescent_mother_flag": mother_age is not None and mother_age < 20,
        "advanced_maternal_age_flag": mother_age is not None and mother_age >= 35,
        "mother_race_admin_code": mother_race,
        "mother_race_state": mother_race_state,
        "newborn_race_admin_code": newborn_race,
        "newborn_race_state": newborn_race_state,
        "sex_code": _digits(_raw(row, "SEXO")),
        "birth_weight_g": weight,
        "birth_weight_state": weight_state,
        "low_birth_weight_flag": low_weight,
        "gestational_age_weeks": gest_weeks,
        "gestational_age_state": gest_state,
        "prematurity_flag": preterm,
        "apgar1": apgar1,
        "apgar1_state": apgar1_state,
        "low_apgar1_flag": low_apgar1,
        "apgar5": apgar5,
        "apgar5_state": apgar5_state,
        "low_apgar5_flag": low_apgar5,
        "delivery_mode_code": delivery_code,
        "delivery_mode_state": delivery_state,
        "cesarean_flag": cesarean,
        "prenatal_consult_count": consultations,
        "prenatal_consult_state": consultations_state,
        "prenatal_consult_raw_digits": consultations_raw,
        "insufficient_prenatal_flag": consultations is not None and consultations < 7,
        "anomaly_flag": anomaly_flag,
        "anomaly_flag_state": anomaly_flag_state,
        "anomaly_icd_raw": _clean(_raw(row, "CODANOMAL", "ANOMALIA")),
        "anomaly_icd_code": anomaly_code,
        "anomaly_icd_state": anomaly_state,
        "congenital_anomaly_flag": anomaly_any,
        "record_state": record_state,
        "source_manifest_hash": source_manifest_hash,
        "row_hash": _stable_hash(raw_payload),
        "raw_json": json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str),
    }


def normalize_sinasc_events(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, Any]:
    df = _read_table(input_path)
    rows = [normalize_sinasc_record(row, source_manifest_hash=source_manifest_hash) for row in df.to_dicts()]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)
    return {
        "row_count": len(rows),
        "output_path": str(out),
        "valid_rows": sum(1 for row in rows if row["record_state"] == "valid"),
        "low_birth_weight_rows": sum(1 for row in rows if row["low_birth_weight_flag"] is True),
        "prematurity_rows": sum(1 for row in rows if row["prematurity_flag"] is True),
        "cesarean_rows": sum(1 for row in rows if row["cesarean_flag"] is True),
        "low_apgar5_rows": sum(1 for row in rows if row["low_apgar5_flag"] is True),
        "insufficient_prenatal_rows": sum(1 for row in rows if row["insufficient_prenatal_flag"] is True),
        "anomaly_rows": sum(1 for row in rows if row["congenital_anomaly_flag"] is True),
    }
