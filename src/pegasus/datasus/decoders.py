from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict


class DecodedField(BaseModel):
    """Generic decoder result carrying a value plus an explicit MSD §2.3 state.

    Used by the registry-driven declarative normalizer for single-column decoders
    (dates, hours, municipality codes, facility codes, administrative race, plain
    integers) so that the declarative engine's generic ``value``/``state``
    extraction works uniformly without per-decoder special-casing.
    """

    model_config = ConfigDict(extra="forbid")
    value: Any = None
    state: str = "missing"
    raw_value: str | None = None
    warning: str | None = None


class DecodedAge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age_years: float | None
    age_days: float | None
    age_unit: str | None
    raw_value: str | None
    state: str
    warning: str | None = None


class DecodedScalar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float | None
    unit: str
    raw_value: str | None
    state: str
    warning: str | None = None


class DecodedCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int | None
    raw_value: str | None
    state: str
    warning: str | None = None


class DecodedBoolean(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int | None
    raw_value: str | None
    state: Literal[
        "ValidFalse",
        "ValidTrue",
        "MissingFlag",
        "InvalidFlagState",
        "UnparseableFlag",
    ]
    warning: str | None = None


class FilteredCNPJ(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cnpj: str | None
    raw_value: str | None
    state: Literal[
        "ValidCNPJ",
        "NullifiedZeroCNPJ",
        "InvalidCNPJLength",
        "InvalidCNPJDigits",
        "MissingCNPJ",
        "UnparseableCNPJ",
    ]
    warning: str | None = None


def _none_or_blank(raw: Any) -> bool:
    return raw is None or str(raw).strip() == "" or str(raw).strip().upper() in {"NA", "NAN", "NULL"}


def decode_sim_idade(raw: str | int | None) -> DecodedAge:
    if _none_or_blank(raw):
        return DecodedAge(
            age_years=None,
            age_days=None,
            age_unit=None,
            raw_value=None if raw is None else str(raw),
            state="UnknownAge",
            warning="missing_age",
        )

    raw_s = str(raw).strip()
    if not raw_s.isdigit():
        return DecodedAge(
            age_years=None,
            age_days=None,
            age_unit=None,
            raw_value=raw_s,
            state="UnknownAge",
            warning="unparseable_age",
        )

    z = raw_s.zfill(3)
    try:
        code = int(z)
    except ValueError:
        return DecodedAge(
            age_years=None,
            age_days=None,
            age_unit=None,
            raw_value=raw_s,
            state="UnknownAge",
            warning="unparseable_age",
        )

    unit = code // 100
    magnitude = code - 100 * unit

    if unit == 1:
        return DecodedAge(
            age_years=magnitude / (24 * 365.25),
            age_days=magnitude / 24,
            age_unit="hours",
            raw_value=raw_s,
            state="valid",
        )
    if unit == 2:
        return DecodedAge(
            age_years=magnitude / 365.25,
            age_days=float(magnitude),
            age_unit="days",
            raw_value=raw_s,
            state="valid",
        )
    if unit == 3:
        return DecodedAge(
            age_years=magnitude / 12,
            age_days=30.4375 * magnitude,
            age_unit="months",
            raw_value=raw_s,
            state="valid",
        )
    if unit == 4:
        return DecodedAge(
            age_years=float(magnitude),
            age_days=365.25 * magnitude,
            age_unit="years",
            raw_value=raw_s,
            state="valid",
        )
    if unit == 5:
        years = 100 + magnitude
        return DecodedAge(
            age_years=float(years),
            age_days=365.25 * years,
            age_unit="years_100_plus",
            raw_value=raw_s,
            state="valid",
        )

    return DecodedAge(
        age_years=None,
        age_days=None,
        age_unit=None,
        raw_value=raw_s,
        state="UnknownAge",
        warning="unknown_age_unit",
    )


@lru_cache(maxsize=4)
def _sih_age_unit_map(registry_root: str = "config/registries") -> dict[str, dict[str, Any]]:
    path = Path(registry_root) / "composite_decoders.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}
    for entry in payload.get("entries") or []:
        if isinstance(entry, dict) and entry.get("id") == "Decode_SIH_AGE":
            unit_map = entry.get("unit_map")
            if isinstance(unit_map, dict):
                return {str(k): dict(v) for k, v in unit_map.items() if isinstance(v, dict)}
    return {
        "0": {"unit": "ignored", "state": "UnknownAgeUnit", "warning": "age_unit_missing"},
        "1": {"unit": "hours", "years_factor": 1 / (24 * 365.25), "days_factor": 1 / 24},
        "2": {"unit": "days", "years_factor": 1 / 365.25, "days_factor": 1.0},
        "3": {"unit": "months", "years_factor": 1 / 12, "days_factor": 30.4375},
        "4": {"unit": "years", "years_factor": 1.0, "days_factor": 365.25},
        "5": {"unit": "years_100_plus", "years_offset": 100.0, "days_factor": 365.25, "days_offset_years": 100.0},
    }


def decode_sih_age(cod_idade: str | int | None, idade: str | int | None) -> DecodedAge:
    raw_unit = None if cod_idade is None else str(cod_idade).strip()
    raw_age = None if idade is None else str(idade).strip()

    if _none_or_blank(raw_unit):
        return DecodedAge(
            age_years=None,
            age_days=None,
            age_unit=None,
            raw_value=raw_age,
            state="UnknownAgeUnit",
            warning="age_unit_missing",
        )

    if _none_or_blank(raw_age) or not str(raw_age).isdigit():
        return DecodedAge(
            age_years=None,
            age_days=None,
            age_unit=None,
            raw_value=raw_age,
            state="UnknownAge",
            warning="age_missing_or_unparseable",
        )

    m = int(raw_age)
    unit_spec = _sih_age_unit_map().get(raw_unit)
    if unit_spec is not None:
        if unit_spec.get("state"):
            return DecodedAge(
                age_years=None,
                age_days=None,
                age_unit=str(unit_spec.get("unit") or "unknown"),
                raw_value=raw_age,
                state=str(unit_spec["state"]),
                warning=str(unit_spec.get("warning") or "age_unit_missing"),
            )
        years_offset = float(unit_spec.get("years_offset") or 0.0)
        years = years_offset + m * float(unit_spec.get("years_factor", 0.0))
        days_offset_years = float(unit_spec.get("days_offset_years") or 0.0)
        days = 365.25 * days_offset_years + m * float(unit_spec.get("days_factor", 0.0))
        return DecodedAge(
            age_years=years,
            age_days=days,
            age_unit=str(unit_spec.get("unit") or "unknown"),
            raw_value=raw_age,
            state="valid",
        )

    return DecodedAge(
        age_years=None,
        age_days=None,
        age_unit=None,
        raw_value=raw_age,
        state="UnknownAgeUnit",
        warning="age_unit_invalid",
    )


def decode_physical_scalar(
    raw: str | int | float | None,
    *,
    unit: str,
    lower: float,
    upper: float,
    sentinels: set[str],
) -> DecodedScalar:
    if _none_or_blank(raw):
        return DecodedScalar(value=None, unit=unit, raw_value=None if raw is None else str(raw), state="MissingScalar", warning="missing_scalar")

    raw_s = str(raw).strip()
    if raw_s in sentinels:
        return DecodedScalar(value=None, unit=unit, raw_value=raw_s, state="InvalidScalar", warning="sentinel_scalar")

    try:
        value = float(raw_s.replace(",", "."))
    except ValueError:
        return DecodedScalar(value=None, unit=unit, raw_value=raw_s, state="UnparseableScalar", warning="unparseable_scalar")

    if value < lower or value > upper:
        return DecodedScalar(value=None, unit=unit, raw_value=raw_s, state="OutOfRangeScalar", warning="out_of_range_scalar")

    return DecodedScalar(value=value, unit=unit, raw_value=raw_s, state="valid")


def decode_count2(raw: str | int | None, *, sentinels: set[str]) -> DecodedCount:
    if _none_or_blank(raw):
        return DecodedCount(value=None, raw_value=None if raw is None else str(raw), state="UnknownCount", warning="missing_count")

    raw_s = str(raw).strip().zfill(2)
    if raw_s in sentinels:
        return DecodedCount(value=None, raw_value=raw_s, state="UnknownCount", warning="sentinel_count")

    if re.fullmatch(r"\d{1,2}", raw_s):
        return DecodedCount(value=int(raw_s), raw_value=raw_s, state="valid")

    return DecodedCount(value=None, raw_value=raw_s, state="InvalidCount", warning="invalid_count")


def clamp_bool(raw: object) -> DecodedBoolean:
    if raw is None:
        return DecodedBoolean(value=None, raw_value=None, state="MissingFlag", warning="missing_flag")

    raw_s = str(raw).strip()
    norm = raw_s.casefold()

    if norm in {"", "na", "nan", "null"}:
        return DecodedBoolean(value=None, raw_value=raw_s, state="MissingFlag", warning="missing_flag")

    if norm in {"0", "não", "nao", "no", "false", "f"}:
        return DecodedBoolean(value=0, raw_value=raw_s, state="ValidFalse")

    if norm in {"1", "sim", "yes", "true", "t"}:
        return DecodedBoolean(value=1, raw_value=raw_s, state="ValidTrue")

    try:
        f = float(raw_s.replace(",", "."))
    except ValueError:
        return DecodedBoolean(value=None, raw_value=raw_s, state="UnparseableFlag", warning="unparseable_flag")

    # Non-integer floats ("1.5", "0.7") are not valid boolean codes.
    if f != int(f):
        return DecodedBoolean(value=None, raw_value=raw_s, state="UnparseableFlag", warning="unparseable_flag")

    parsed = int(f)
    if parsed not in {0, 1}:
        return DecodedBoolean(value=None, raw_value=raw_s, state="InvalidFlagState", warning="boolean_flag_outlier")

    return DecodedBoolean(value=parsed, raw_value=raw_s, state="ValidTrue" if parsed == 1 else "ValidFalse")


def filter_cnpj(raw: object) -> FilteredCNPJ:
    if raw is None:
        return FilteredCNPJ(cnpj=None, raw_value=None, state="MissingCNPJ", warning="missing_cnpj")

    raw_s = str(raw).strip()
    if raw_s == "" or raw_s.upper() in {"NA", "NAN", "NULL"}:
        return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="MissingCNPJ", warning="missing_cnpj")

    digits = re.sub(r"\D", "", raw_s)

    if digits == "":
        return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="UnparseableCNPJ", warning="unparseable_cnpj")

    if set(digits) == {"0"}:
        return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="NullifiedZeroCNPJ", warning="zero_cnpj_nullified")

    if len(digits) != 14:
        return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="InvalidCNPJLength", warning="invalid_cnpj_length")

    if not digits.isdigit():
        return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="InvalidCNPJDigits", warning="invalid_cnpj_digits")

    if not _valid_cnpj_check_digits(digits):
        return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="InvalidCNPJDigits", warning="invalid_cnpj_digits")

    return FilteredCNPJ(cnpj=digits, raw_value=raw_s, state="ValidCNPJ")


def _valid_cnpj_check_digits(digits: str) -> bool:
    if len(digits) != 14 or not digits.isdigit() or len(set(digits)) == 1:
        return False

    def check_digit(prefix: str, weights: tuple[int, ...]) -> str:
        total = sum(int(digit) * weight for digit, weight in zip(prefix, weights, strict=True))
        remainder = total % 11
        value = 0 if remainder < 2 else 11 - remainder
        return str(value)

    first = check_digit(digits[:12], (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2))
    second = check_digit(digits[:12] + first, (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2))
    return digits[-2:] == first + second


# ---------------------------------------------------------------------------
# Registry-driven single-column decoders (SHE-NORM-01 Path A).
#
# These are dispatched by name from declarative_normalize.py via the source
# field registry. Each returns a DecodedField so the declarative engine extracts
# value/state uniformly. They are the single source of truth for these decodes —
# the vectorized normalizers no longer re-implement them inline (XCUT-02).
# ---------------------------------------------------------------------------

# DATASUS sentinel date encodings that mean "no real date" rather than a value.
_INVALID_DATE_TOKENS = {
    "",
    "0",
    "00000000",
    "0000-00-00",
    "00/00/0000",
    "99999999",
    "9999-99-99",
    "99/99/9999",
}
_DATE_FORMATS = ("%d%m%Y", "%Y%m%d", "%d/%m/%Y", "%Y-%m-%d")


def decode_datasus_date(raw: Any) -> DecodedField:
    """Decode a DATASUS date (typically DDMMYYYY) to an ISO ``YYYY-MM-DD`` string.

    Preserves MSD §2.3 states: ``missing`` for blank/sentinel, ``invalid`` for a
    present-but-unparseable token, ``valid`` otherwise.
    """
    if _none_or_blank(raw):
        return DecodedField(value=None, state="missing", raw_value=None if raw is None else str(raw))
    text = str(raw).strip()
    if text.upper() in _INVALID_DATE_TOKENS:
        return DecodedField(value=None, state="missing", raw_value=text)
    for fmt in _DATE_FORMATS:
        try:
            iso = datetime.strptime(text, fmt).date().isoformat()
            return DecodedField(value=iso, state="valid", raw_value=text)
        except ValueError:
            continue
    return DecodedField(value=None, state="invalid", raw_value=text, warning="unparseable_date")


def decode_datasus_hour(raw: Any) -> DecodedField:
    """Decode a DATASUS hour field (HHMM or HH) to ``HH:MM:00``."""
    if _none_or_blank(raw):
        return DecodedField(value=None, state="missing", raw_value=None if raw is None else str(raw))
    text = str(raw).strip()
    digits = re.sub(r"\D", "", text)
    if digits == "":
        return DecodedField(value=None, state="missing", raw_value=text)
    if len(digits) <= 2:
        hour, minute = int(digits), 0
    else:
        padded = digits.zfill(4)
        hour, minute = int(padded[:2]), int(padded[2:4])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return DecodedField(value=None, state="invalid", raw_value=text, warning="hour_out_of_range")
    return DecodedField(value=f"{hour:02d}:{minute:02d}:00", state="valid", raw_value=text)


def decode_municipality_cod6(raw: Any) -> DecodedField:
    """Decode a municipality code to its 6-digit DATASUS form (cod6).

    A 7-digit IBGE code is truncated to its 6-digit prefix. The cod6→cod7
    crosswalk is applied separately by the assembler (it is a genuine derivation,
    not a single-column decode).
    """
    if _none_or_blank(raw):
        return DecodedField(value=None, state="missing", raw_value=None if raw is None else str(raw))
    text = str(raw).strip()
    digits = re.sub(r"\D", "", text)
    cod6: str | None = None
    if len(digits) == 6:
        cod6 = digits
    elif len(digits) == 7:
        cod6 = digits[:6]
    else:
        return DecodedField(value=None, state="invalid", raw_value=text, warning="municipality_code_length")
    # DATASUS "município ignorado" sentinel: a valid UF prefix followed by 0000
    # (e.g. 270000 for Alagoas). A real municipality sequence is never 0000, so this
    # is a missingness state per MSD §2.3, not a geography stratum.
    if cod6[2:] == "0000":
        return DecodedField(value=None, state="ignored_municipality", raw_value=text, warning="municipality_ignored_sentinel")
    return DecodedField(value=cod6, state="valid", raw_value=text)


def decode_facility_code(raw: Any) -> DecodedField:
    """Decode a CNES facility code, nulling all-zero/blank sentinels."""
    if _none_or_blank(raw):
        return DecodedField(value=None, state="missing", raw_value=None if raw is None else str(raw))
    text = str(raw).strip()
    digits = re.sub(r"\D", "", text)
    if digits == "" or set(digits) == {"0"}:
        return DecodedField(value=None, state="missing", raw_value=text)
    return DecodedField(value=digits, state="valid", raw_value=text)


def decode_race_admin(raw: Any) -> DecodedField:
    """Decode an administrative race/color code (DATASUS RACACOR family).

    Codes 1–5 are valid administrative categories; ``9``/``99`` are the explicit
    "unknown" sentinel (kept distinct from missing per §2.3). The code itself is
    preserved as the value; the state carries the missingness classification.
    """
    if _none_or_blank(raw):
        return DecodedField(value=None, state="missing", raw_value=None if raw is None else str(raw))
    text = str(raw).strip()
    code = re.sub(r"\D", "", text)
    if code in {"9", "99"}:
        return DecodedField(value=code, state="unknown", raw_value=text)
    if code in {"1", "2", "3", "4", "5"}:
        return DecodedField(value=code, state="valid", raw_value=text)
    if code == "":
        return DecodedField(value=None, state="missing", raw_value=text)
    return DecodedField(value=code, state="invalid", raw_value=text, warning="race_code_outlier")


def decode_datasus_sex(raw: Any) -> DecodedField:
    """Decode a DATASUS SEXO code with explicit MSD §2.3 missingness states.

    Single source of truth for sex decoding across SIM-DO / SINASC / SIH-RD:
    1→male, 2→female (valid); 9→unknown; blank→missing; anything else→invalid.
    """
    if _none_or_blank(raw):
        return DecodedField(value=None, state="missing", raw_value=None if raw is None else str(raw))
    text = str(raw).strip()
    code = re.sub(r"\D", "", text)
    if code == "1":
        return DecodedField(value="male", state="valid", raw_value=text)
    if code == "2":
        return DecodedField(value="female", state="valid", raw_value=text)
    if code == "9":
        return DecodedField(value=None, state="unknown", raw_value=text)
    if code == "":
        return DecodedField(value=None, state="missing", raw_value=text)
    return DecodedField(value=None, state="invalid", raw_value=text, warning="sex_code_outlier")


def canonical_scalar_state(decoded: "DecodedScalar") -> str:
    """Map decode_physical_scalar's audit state vocabulary to the canonical MSD §2.3
    states used by the SINASC/SIH/CNES substrate outputs and the EFG legality layer.

    valid→valid; blank→missing; declared sentinel→sentinel; out-of-range or
    unparseable→invalid. This keeps decode_physical_scalar the single source of the
    bounds/sentinel/parse *logic* (SHE-NORM-01 / XCUT-02) while preserving the
    explicit five-state §2.3 contract downstream consumers depend on.
    """
    if decoded.state == "valid":
        return "valid"
    if decoded.warning == "sentinel_scalar":
        return "sentinel"
    if decoded.state == "MissingScalar":
        return "missing"
    return "invalid"


def decode_integer(raw: Any) -> DecodedField:
    """Decode a plain non-negative-or-signed integer mark, preserving sign.

    Non-integer floats (e.g. ``"3.5"``) are ``invalid``; blanks are ``missing``.
    """
    if _none_or_blank(raw):
        return DecodedField(value=None, state="missing", raw_value=None if raw is None else str(raw))
    text = str(raw).strip()
    try:
        f = float(text.replace(",", "."))
    except ValueError:
        return DecodedField(value=None, state="invalid", raw_value=text, warning="unparseable_integer")
    if f != int(f):
        return DecodedField(value=None, state="invalid", raw_value=text, warning="non_integer_value")
    return DecodedField(value=int(f), state="valid", raw_value=text)
