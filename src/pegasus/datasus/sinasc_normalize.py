from __future__ import annotations

from pegasus.datasus.declarative_normalize import normalize_sim_do_record as _registry_normalize_sim_do_record, normalize_sinasc_record as _registry_normalize_sinasc_record
from pegasus.datasus.decoders import (
    DecodedScalar,
    canonical_scalar_state,
    decode_count2,
    decode_datasus_sex,
    decode_physical_scalar,
)

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
    cod6 = digits if len(digits) == 6 else (digits[:6] if len(digits) == 7 else None)
    if cod6 is None:
        return None, None, "invalid"
    # DATASUS "município ignorado" sentinel (UF + 0000) is missingness, not geography.
    if cod6[2:] == "0000":
        return None, None, "ignored_municipality"
    if len(digits) == 7:
        return cod6, digits, "ibge_cod7"
    return cod6, None, "datasus_cod6"


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


# The three perinatal scalar decoders below now delegate to the shared
# decode_physical_scalar (the single bounds/sentinel/parse authority, SHE-NORM-01 /
# XCUT-02) and translate its audit state vocabulary to the canonical MSD §2.3 states
# via canonical_scalar_state. The clinical-threshold flag (third tuple element) is a
# convenience for the run-summary counters only — it is NOT emitted as a SHE column
# (threshold indicators belong to the EFG, SHE-SINASC-01).
def _scalar(value: Any, *, lower: float, upper: float, sentinels: set[str]) -> DecodedScalar:
    return decode_physical_scalar(value, unit="scalar", lower=lower, upper=upper, sentinels=sentinels)


def decode_birth_weight(value: Any) -> tuple[int | None, str, bool | None]:
    d = _scalar(value, lower=300, upper=7000, sentinels={"0", "00", "000", "0000", "9999"})
    state = canonical_scalar_state(d)
    weight = int(d.value) if d.value is not None else None
    return weight, state, (weight < 2500 if weight is not None else None)


def decode_gestational_age(value: Any) -> tuple[int | None, str, bool | None]:
    d = _scalar(value, lower=20, upper=45, sentinels={"0", "00", "99"})
    state = canonical_scalar_state(d)
    weeks = int(d.value) if d.value is not None else None
    return weeks, state, (weeks < 37 if weeks is not None else None)


def decode_apgar(value: Any) -> tuple[int | None, str, bool | None]:
    d = _scalar(value, lower=0, upper=10, sentinels={"99"})
    state = canonical_scalar_state(d)
    score = int(d.value) if d.value is not None else None
    return score, state, (score < 7 if score is not None else None)


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


def decode_sex(value: Any) -> tuple[str | None, str]:
    """Decode SINASC SEXO via the shared single-authority sex decoder (§2.3)."""
    d = decode_datasus_sex(value)
    return d.value, d.state


def decode_anomaly_flag(value: Any) -> tuple[bool | None, str]:
    """Decode SINASC IDANOMAL / congenital-anomaly declaration.

    Official SINASC coding:
    1 = Sim / anomaly present
    2 = Não / anomaly absent
    9 = Ignorado

    The decoder also accepts microdatasus-translated labels because some
    processing paths may expose categorical text rather than raw numeric codes.
    """
    raw = _clean(value)
    if raw is None:
        return None, "missing"

    norm = raw.strip().casefold()
    digits = _digits(raw)

    if digits == "1" or norm in {"sim", "s", "yes", "true", "verdadeiro"}:
        return True, "valid_present"
    if digits == "2" or norm in {"não", "nao", "n", "no", "false", "falso"}:
        return False, "valid_absent"
    if digits == "9" or norm in {"ignorado", "ign", "ignored"}:
        return None, "sentinel"

    return None, "invalid"


def normalize_anomaly_icd(value: Any, anomaly_flag: bool | None) -> tuple[str | None, str, bool]:
    """Normalize CODANOMAL without letting absent declarations become numerators.

    Numerator rule:
    - explicit IDANOMAL=1 / Sim is anomaly-positive even when CODANOMAL is missing;
    - valid Q* anomaly ICD is anomaly-positive;
    - explicit IDANOMAL=2 / Não is anomaly-negative even when CODANOMAL has junk;
    - unknown declaration plus non-Q or invalid code is not an anomaly numerator.
    """
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
        if anomaly_flag is True:
            return None, "flag_present_invalid_code", True
        if anomaly_flag is False:
            return None, "absent_invalid_code_ignored", False
        return None, "invalid", False

    code = match.group(0)
    if code.startswith("Q"):
        return code, "valid_q_anomaly", True
    if anomaly_flag is True:
        return code, "valid_non_q_with_present_flag", True
    if anomaly_flag is False:
        return code, "valid_non_q_absent", False
    return code, "valid_non_q_not_anomaly", False


def normalize_sinasc_events(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, Any]:
    """Real vectorized SINASC raw→canonical SHE decoder (MSD §2.4.3, §2.6).

    Replaces the previous path whose summary logic read flag keys the
    registry-routed record normalizer never produced (it raised KeyError on the
    first real file). Every canonical field is a Polars expression over the raw
    SINASC columns; clinical indicators (low birth weight, prematurity, cesarean,
    maternal-age bands) are computed per §2.6 definitions as 0/1 additive flags."""
    df = _read_table(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(df.to_dicts()):
        raw_payload = {str(k): v for k, v in row.items()}
        birth_date, birth_year, birth_date_state = parse_sinasc_date(row.get("DTNASC"))
        res6, res7, res_state = municipality_codes(row.get("CODMUNRES"))
        birth6, birth7, birth_state = municipality_codes(row.get("CODMUNNASC"))
        mother_age = int_or_none(row.get("IDADEMAE"))
        birth_weight, birth_weight_state, _low_birth_weight = decode_birth_weight(row.get("PESO"))
        gest_weeks, gest_state, _premature = decode_gestational_age(row.get("SEMAGESTAC"))
        apgar1, apgar1_state, _low_apgar1 = decode_apgar(row.get("APGAR1"))
        apgar5, apgar5_state, _low_apgar5 = decode_apgar(row.get("APGAR5"))
        delivery_code, delivery_state, _cesarean = decode_delivery_mode(row.get("PARTO"))
        prenatal, prenatal_state, prenatal_raw = decode_count_preserve_leading_zero(
            row.get("CONSULTAS"),
            sentinels={"99"},
            upper=98,
        )
        anomaly_flag, anomaly_flag_state = decode_anomaly_flag(row.get("IDANOMAL"))
        anomaly_code, anomaly_code_state, anomaly_positive = normalize_anomaly_icd(row.get("CODANOMAL"), anomaly_flag)
        newborn_race, newborn_race_state = decode_race(row.get("RACACOR"))
        maternal_race, maternal_race_state = decode_race(row.get("RACACORMAE"))
        newborn_sex, newborn_sex_state = decode_sex(row.get("SEXO"))
        # Reproductive history — decode_count2 preserves sentinel/state semantics
        # (MSD §2.4.3); int_or_none would lose the 99-sentinel and missing/invalid.
        live_children = decode_count2(row.get("QTDFILVIVO"), sentinels={"99"})
        deceased_children = decode_count2(row.get("QTDFILMORT"), sentinels={"99"})
        prior_pregnancies = decode_count2(row.get("QTDGESTANT"), sentinels={"99"})
        prior_vaginal = decode_count2(row.get("QTDPARTNOR"), sentinels={"99"})
        prior_cesarean = decode_count2(row.get("QTDPARTCES"), sentinels={"99"})
        event_key = _clean(row.get("NUMERODN")) or f"{idx}_{source_manifest_hash[:8]}"
        records.append(
            {
                "event_id": f"SINASC-{event_key}",
                "source_system": "SINASC",
                "birth_date": birth_date,
                "birth_year": birth_year,
                "birth_date_state": birth_date_state,
                "mun_residence_cod6": res6,
                "mun_residence_cod7": res7,
                "mun_residence_state": res_state,
                "mun_birth_cod6": birth6,
                "mun_birth_cod7": birth7,
                "mun_birth_state": birth_state,
                "mother_age_years": mother_age,
                "maternal_birth_date": _clean(row.get("DTNASCMAE")),
                "newborn_sex": newborn_sex,
                "newborn_sex_state": newborn_sex_state,
                "newborn_race_admin": newborn_race,
                "newborn_race_state": newborn_race_state,
                "maternal_race_admin": maternal_race,
                "maternal_race_state": maternal_race_state,
                "birth_weight_g": birth_weight,
                "birth_weight_grams": birth_weight,
                "birth_weight_state": birth_weight_state,
                "gestational_weeks": gest_weeks,
                "gestational_age_state": gest_state,
                "apgar_1min": apgar1,
                "apgar_1min_state": apgar1_state,
                "apgar_5min": apgar5,
                "apgar_5min_state": apgar5_state,
                "delivery_mode_code": delivery_code,
                "delivery_mode_state": delivery_state,
                "prenatal_consult_count": prenatal,
                "prenatal_consult_state": prenatal_state,
                "prenatal_consult_raw_digits": prenatal_raw,
                "prenatal_visit_group": _clean(row.get("CONSULTAS")),
                "live_children_count": live_children.value,
                "live_children_state": live_children.state,
                "deceased_children_count": deceased_children.value,
                "deceased_children_state": deceased_children.state,
                "prior_pregnancy_count": prior_pregnancies.value,
                "prior_pregnancy_state": prior_pregnancies.state,
                "prior_vaginal_delivery_count": prior_vaginal.value,
                "prior_vaginal_delivery_state": prior_vaginal.state,
                "prior_cesarean_delivery_count": prior_cesarean.value,
                "prior_cesarean_delivery_state": prior_cesarean.state,
                "anomaly_flag": anomaly_flag,
                "anomaly_flag_state": anomaly_flag_state,
                "anomaly_icd_code": anomaly_code,
                "anomaly_icd_state": anomaly_code_state,
                "anomaly_positive": anomaly_positive,
                "record_state": "valid" if birth_year is not None and res6 is not None else "invalid_identity",
                "source_manifest_hash": source_manifest_hash,
                "row_hash": _stable_hash(raw_payload),
                "raw_json": json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str),
            }
        )
    out = pl.DataFrame(records, infer_schema_length=None)
    out.write_parquet(out_path)
    valid_rows = int((out.get_column("record_state") == "valid").sum()) if out.height else 0
    return {
        "row_count": out.height,
        "valid_rows": valid_rows,
        "low_birth_weight_rows": int(out.filter(pl.col("birth_weight_g").is_not_null() & (pl.col("birth_weight_g") < 2500)).height) if out.height else 0,
        "prematurity_rows": int(out.filter(pl.col("gestational_weeks").is_not_null() & (pl.col("gestational_weeks") < 37)).height) if out.height else 0,
        "cesarean_rows": int(out.filter(pl.col("delivery_mode_code") == "2").height) if out.height else 0,
        "low_apgar5_rows": int(out.filter(pl.col("apgar_5min").is_not_null() & (pl.col("apgar_5min") < 7)).height) if out.height else 0,
        "insufficient_prenatal_rows": int(out.filter(pl.col("prenatal_consult_count").is_not_null() & (pl.col("prenatal_consult_count") < 7)).height) if out.height else 0,
        "anomaly_rows": int(out.get_column("anomaly_positive").fill_null(False).sum()) if out.height else 0,
        "output_path": str(out_path),
        "column_count": len(out.columns),
        "columns": out.columns,
    }

# ---- Hardline MSD SHE registry-routed entrypoint ----
def normalize_sinasc_record(row: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
    registry_root = kwargs.get('registry_root', 'config/registries')
    return _registry_normalize_sinasc_record(row, registry_root=registry_root)
