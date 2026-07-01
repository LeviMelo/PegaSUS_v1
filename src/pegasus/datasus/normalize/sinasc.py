from __future__ import annotations

from pegasus.datasus.normalize.records import normalize_sim_do_record as _registry_normalize_sim_do_record, normalize_sinasc_record as _registry_normalize_sinasc_record
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

from pegasus.datasus.normalize.completeness import check_raw_completeness
from pegasus.datasus.normalize.primitives import Cols, read_raw_table

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
    return read_raw_table(path)


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
    df = _read_table(input_path).with_row_index("_row_idx")
    missing_columns = check_raw_completeness(df, "SINASC")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared decode primitives (single source of truth, XCUT-02). SINASC carries no
    # cod6→cod7 crosswalk (only a 7-digit input yields cod7), matching Cols'
    # crosswalk=None path. The scalar/count2 builders below keep SINASC's own §2.3
    # output-state vocabulary (missing/sentinel/invalid/valid, MissingCount/…).
    cx = Cols(df)
    clean = cx.clean
    digits = cx.digits
    int_digits = cx.int_digits
    municipality = cx.municipality

    def scalar(name: str, out_name: str, state_name: str, *, lower: int, upper: int, sentinels: list[str]) -> list[pl.Expr]:
        d = digits(name)
        value = d.cast(pl.Int64, strict=False)
        sentinel = d.is_in(sentinels)
        valid = value.is_not_null() & ~sentinel & (value >= lower) & (value <= upper)
        return [
            pl.when(valid).then(value).otherwise(None).alias(out_name),
            pl.when(d.is_null()).then(pl.lit("missing"))
            .when(sentinel).then(pl.lit("sentinel"))
            .when(value.is_null() | (value < lower) | (value > upper)).then(pl.lit("invalid"))
            .otherwise(pl.lit("valid"))
            .alias(state_name),
        ]

    def count2(name: str, out_name: str, state_name: str, *, upper: int = 98) -> list[pl.Expr]:
        d = digits(name)
        value = d.cast(pl.Int64, strict=False)
        sentinel = d == "99"
        valid = value.is_not_null() & ~sentinel & (value >= 0) & (value <= upper)
        return [
            pl.when(valid).then(value).otherwise(None).alias(out_name),
            pl.when(d.is_null()).then(pl.lit("MissingCount"))
            .when(sentinel).then(pl.lit("InvalidCount"))
            .when(value.is_null() | (value < 0) | (value > upper)).then(pl.lit("InvalidCount"))
            .otherwise(pl.lit("valid"))
            .alias(state_name),
        ]

    birth_digits = digits("DTNASC")
    birth_ymd = birth_digits.str.strptime(pl.Date, "%Y%m%d", strict=False)
    birth_dmy = birth_digits.str.strptime(pl.Date, "%d%m%Y", strict=False)
    birth_dt = pl.coalesce([birth_ymd, birth_dmy])
    anomaly_flag_digits = digits("IDANOMAL")
    anomaly_flag = (
        pl.when(anomaly_flag_digits == "1").then(True)
        .when(anomaly_flag_digits == "2").then(False)
        .otherwise(None)
    )
    anomaly_code = clean("CODANOMAL").str.to_uppercase().str.extract(r"([A-Z][0-9]{2}[0-9A-Z]?)", 1)
    event_key = pl.coalesce([clean("NUMERODN"), clean("contador"), clean("CONTADOR"), pl.col("_row_idx").cast(pl.Utf8) + pl.lit(f"_{source_manifest_hash[:8]}")])

    out = df.with_columns(
        birth_dt.alias("_birth_dt"),
        birth_dt.dt.year().alias("_birth_year"),
        *municipality("mun_residence", "CODMUNRES"),
        *municipality("mun_birth", "CODMUNNASC"),
        int_digits("IDADEMAE").alias("mother_age_years"),
        *scalar("PESO", "birth_weight_g", "birth_weight_state", lower=300, upper=7000, sentinels=["0", "00", "000", "0000", "9999"]),
        *scalar("SEMAGESTAC", "gestational_weeks", "gestational_age_state", lower=20, upper=45, sentinels=["0", "00", "99"]),
        *scalar("APGAR1", "apgar_1min", "apgar_1min_state", lower=0, upper=10, sentinels=["99"]),
        *scalar("APGAR5", "apgar_5min", "apgar_5min_state", lower=0, upper=10, sentinels=["99"]),
        *count2("QTDFILVIVO", "live_children_count", "live_children_state"),
        *count2("QTDFILMORT", "deceased_children_count", "deceased_children_state"),
        *count2("QTDGESTANT", "prior_pregnancy_count", "prior_pregnancy_state"),
        *count2("QTDPARTNOR", "prior_vaginal_delivery_count", "prior_vaginal_delivery_state"),
        *count2("QTDPARTCES", "prior_cesarean_delivery_count", "prior_cesarean_delivery_state"),
        event_key.alias("_event_key"),
        anomaly_flag.alias("anomaly_flag"),
        anomaly_code.alias("anomaly_icd_code"),
    ).with_columns(
        pl.concat_str([pl.lit("SINASC-"), pl.col("_event_key")]).alias("event_id"),
        pl.lit("SINASC").alias("source_system"),
        pl.col("_birth_dt").cast(pl.Utf8).alias("birth_date"),
        pl.col("_birth_year").alias("birth_year"),
        pl.when(clean("DTNASC").is_null()).then(pl.lit("missing"))
        .when(pl.col("_birth_dt").is_null()).then(pl.lit("invalid"))
        .otherwise(pl.lit("valid")).alias("birth_date_state"),
        clean("DTNASCMAE").alias("maternal_birth_date"),
        pl.when(digits("SEXO") == "1").then(pl.lit("male"))
        .when(digits("SEXO") == "2").then(pl.lit("female"))
        .when(digits("SEXO") == "9").then(None)
        .otherwise(None).alias("newborn_sex"),
        pl.when(digits("SEXO").is_null()).then(pl.lit("missing"))
        .when(digits("SEXO").is_in(["1", "2"])).then(pl.lit("valid"))
        .when(digits("SEXO") == "9").then(pl.lit("unknown"))
        .otherwise(pl.lit("invalid")).alias("newborn_sex_state"),
        pl.when(digits("RACACOR").is_in(["1", "2", "3", "4", "5"])).then(digits("RACACOR")).otherwise(None).alias("newborn_race_admin"),
        pl.when(digits("RACACOR").is_null()).then(pl.lit("missing"))
        .when(digits("RACACOR").is_in(["1", "2", "3", "4", "5"])).then(pl.lit("valid_admin_race"))
        .when(digits("RACACOR") == "9").then(pl.lit("ignored_sentinel"))
        .otherwise(pl.lit("invalid")).alias("newborn_race_state"),
        pl.when(digits("RACACORMAE").is_in(["1", "2", "3", "4", "5"])).then(digits("RACACORMAE")).otherwise(None).alias("maternal_race_admin"),
        pl.when(digits("RACACORMAE").is_null()).then(pl.lit("missing"))
        .when(digits("RACACORMAE").is_in(["1", "2", "3", "4", "5"])).then(pl.lit("valid_admin_race"))
        .when(digits("RACACORMAE") == "9").then(pl.lit("ignored_sentinel"))
        .otherwise(pl.lit("invalid")).alias("maternal_race_state"),
        pl.col("birth_weight_g").alias("birth_weight_grams"),
        pl.when(digits("PARTO").is_in(["1", "2", "9"])).then(digits("PARTO")).otherwise(digits("PARTO")).alias("delivery_mode_code"),
        pl.when(digits("PARTO").is_null()).then(pl.lit("missing"))
        .when(digits("PARTO") == "9").then(pl.lit("sentinel"))
        .when(digits("PARTO").is_in(["1", "2"])).then(pl.lit("valid"))
        .otherwise(pl.lit("invalid")).alias("delivery_mode_state"),
        pl.when((digits("CONSULTAS").is_not_null()) & (digits("CONSULTAS") != "99") & (digits("CONSULTAS").cast(pl.Int64, strict=False) <= 98))
        .then(digits("CONSULTAS").cast(pl.Int64, strict=False)).otherwise(None).alias("prenatal_consult_count"),
        pl.when(digits("CONSULTAS").is_null()).then(pl.lit("missing"))
        .when(digits("CONSULTAS") == "99").then(pl.lit("sentinel"))
        .when(digits("CONSULTAS").cast(pl.Int64, strict=False) > 98).then(pl.lit("invalid"))
        .otherwise(pl.lit("valid")).alias("prenatal_consult_state"),
        digits("CONSULTAS").alias("prenatal_consult_raw_digits"),
        clean("CONSULTAS").alias("prenatal_visit_group"),
        pl.when(digits("IDANOMAL").is_null()).then(pl.lit("missing"))
        .when(digits("IDANOMAL") == "1").then(pl.lit("valid_present"))
        .when(digits("IDANOMAL") == "2").then(pl.lit("valid_absent"))
        .when(digits("IDANOMAL") == "9").then(pl.lit("sentinel"))
        .otherwise(pl.lit("invalid")).alias("anomaly_flag_state"),
        pl.when(pl.col("anomaly_icd_code").is_null() & (pl.col("anomaly_flag") == True)).then(pl.lit("flag_present_code_missing"))
        .when(pl.col("anomaly_icd_code").is_null() & (pl.col("anomaly_flag") == False)).then(pl.lit("absent"))
        .when(pl.col("anomaly_icd_code").is_null()).then(pl.lit("unknown"))
        .when(pl.col("anomaly_icd_code").str.starts_with("Q")).then(pl.lit("valid_q_anomaly"))
        .when(pl.col("anomaly_flag") == True).then(pl.lit("valid_non_q_with_present_flag"))
        .when(pl.col("anomaly_flag") == False).then(pl.lit("valid_non_q_absent"))
        .otherwise(pl.lit("valid_non_q_not_anomaly")).alias("anomaly_icd_state"),
        ((pl.col("anomaly_flag") == True) | pl.col("anomaly_icd_code").str.starts_with("Q")).fill_null(False).alias("anomaly_positive"),
        # Codebook-driven additions (in-house microdatasus process_sinasc port):
        # LOCNASC/ESTCIVMAE weren't decoded at all before; delivery_mode_label is a
        # canonical-label sibling to the existing delivery_mode_code (kept as-is —
        # downstream cesarean-rate logic compares its raw "1"/"2" digits).
        cx.categorical_value("local_of_birth", "LOCNASC").alias("place_of_birth"),
        cx.categorical_value("marital_status", "ESTCIVMAE").alias("maternal_marital_status"),
        cx.categorical_value("delivery_type", "PARTO").alias("delivery_mode_label"),
        pl.when(pl.col("_birth_year").is_not_null() & pl.col("mun_residence_cod6").is_not_null()).then(pl.lit("valid")).otherwise(pl.lit("invalid_identity")).alias("record_state"),
        pl.lit(source_manifest_hash).alias("source_manifest_hash"),
        pl.concat_str([pl.lit(source_manifest_hash), pl.lit(":"), pl.col("_row_idx").cast(pl.Utf8)]).hash().cast(pl.Utf8).alias("row_hash"),
        pl.lit(None, dtype=pl.Utf8).alias("raw_json"),
    ).select([
        "event_id", "source_system", "birth_date", "birth_year", "birth_date_state",
        "mun_residence_cod6", "mun_residence_cod7", "mun_residence_state",
        "mun_birth_cod6", "mun_birth_cod7", "mun_birth_state",
        "mother_age_years", "maternal_birth_date", "newborn_sex", "newborn_sex_state",
        "newborn_race_admin", "newborn_race_state", "maternal_race_admin", "maternal_race_state",
        "birth_weight_g", "birth_weight_grams", "birth_weight_state", "gestational_weeks",
        "gestational_age_state", "apgar_1min", "apgar_1min_state", "apgar_5min",
        "apgar_5min_state", "delivery_mode_code", "delivery_mode_state", "delivery_mode_label",
        "prenatal_consult_count", "prenatal_consult_state", "prenatal_consult_raw_digits",
        "prenatal_visit_group", "live_children_count", "live_children_state",
        "deceased_children_count", "deceased_children_state", "prior_pregnancy_count",
        "prior_pregnancy_state", "prior_vaginal_delivery_count", "prior_vaginal_delivery_state",
        "prior_cesarean_delivery_count", "prior_cesarean_delivery_state", "anomaly_flag",
        "anomaly_flag_state", "anomaly_icd_code", "anomaly_icd_state", "anomaly_positive",
        "place_of_birth", "maternal_marital_status",
        "record_state", "source_manifest_hash", "row_hash", "raw_json",
    ])
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
        "missing_required_columns": missing_columns,
    }

# ---- Hardline MSD SHE registry-routed entrypoint ----
def normalize_sinasc_record(row: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
    registry_root = kwargs.get('registry_root', 'config/registries')
    return _registry_normalize_sinasc_record(row, registry_root=registry_root)
