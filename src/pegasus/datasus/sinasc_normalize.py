from __future__ import annotations

from pegasus.datasus.declarative_normalize import normalize_sim_do_record as _registry_normalize_sim_do_record, normalize_sinasc_record as _registry_normalize_sinasc_record

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
    df = _read_table(input_path)
    rows = [normalize_sinasc_record(row, source_manifest_hash=source_manifest_hash) for row in df.to_dicts()]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    valid_rows = sum(1 for row in rows if row["record_state"] == "valid")
    anomaly_rows = sum(1 for row in rows if row["congenital_anomaly_flag"] is True)
    # Do not apply epidemiological plausibility thresholds to tiny materialized panels.
    # Real-source SINASC AL 2022 has tens of thousands of births; the threshold below
    # still catches inverted IDANOMAL/CODANOMAL semantics in production-sized data.
    if valid_rows >= 1000 and anomaly_rows / float(valid_rows) > 0.20:
        raise ValueError(
            "SINASC congenital anomaly numerator exceeds 20% of valid births; "
            "this usually indicates inverted IDANOMAL/CODANOMAL semantics."
        )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out)
    return {
        "row_count": len(rows),
        "output_path": str(out),
        "valid_rows": valid_rows,
        "low_birth_weight_rows": sum(1 for row in rows if row["low_birth_weight_flag"] is True),
        "prematurity_rows": sum(1 for row in rows if row["prematurity_flag"] is True),
        "cesarean_rows": sum(1 for row in rows if row["cesarean_flag"] is True),
        "low_apgar5_rows": sum(1 for row in rows if row["low_apgar5_flag"] is True),
        "insufficient_prenatal_rows": sum(1 for row in rows if row["insufficient_prenatal_flag"] is True),
        "anomaly_rows": anomaly_rows,
    }

# ---- Hardline MSD SHE registry-routed entrypoint ----
def normalize_sinasc_record(row: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
    registry_root = kwargs.get('registry_root', 'config/registries')
    return _registry_normalize_sinasc_record(row, registry_root=registry_root)
