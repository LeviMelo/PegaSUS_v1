from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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

    # microdatasus/SIH codebooks vary by processing layer; keep this registry-replaceable.
    # Conservative defaults: 2=days, 3=months, 4=years, 5=100+ years.
    if raw_unit == "2":
        return DecodedAge(age_years=m / 365.25, age_days=float(m), age_unit="days", raw_value=raw_age, state="valid")
    if raw_unit == "3":
        return DecodedAge(age_years=m / 12, age_days=30.4375 * m, age_unit="months", raw_value=raw_age, state="valid")
    if raw_unit == "4":
        return DecodedAge(age_years=float(m), age_days=365.25 * m, age_unit="years", raw_value=raw_age, state="valid")
    if raw_unit == "5":
        years = 100 + m
        return DecodedAge(age_years=float(years), age_days=365.25 * years, age_unit="years_100_plus", raw_value=raw_age, state="valid")

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
        parsed = int(float(raw_s.replace(",", ".")))
    except ValueError:
        return DecodedBoolean(value=None, raw_value=raw_s, state="UnparseableFlag", warning="unparseable_flag")

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

    return FilteredCNPJ(cnpj=digits, raw_value=raw_s, state="ValidCNPJ")
