from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from pegasus.datasus.contracts import (
    ParsedAge,
    ParsedDate,
    ParsedICD10,
    ParsedMunicipality,
    ParsedValue,
)

_BLANK_STRINGS = {"", " ", "NA", "N/A", "NULL", "NONE", "NAN"}
_INVALID_DATE_PLACEHOLDERS = {
    "00000000",
    "0000-00-00",
    "00/00/0000",
    "99999999",
    "9999-99-99",
    "0",
}

# This is intentionally conservative. A full ICD catalog will later replace
# shape-only validation. Until then, we distinguish shape-valid codes from
# blank/invalid/unparseable states and route selected garbage codes as observer
# process codes.
_ICD10_SHAPE = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](?:\.?[0-9A-Z]{1,2})?$", re.IGNORECASE)

# Initial low-specificity / ill-defined group. This is not a full garbage-code
# registry. It is enough for first-slice SIM-DO observer routing.
_ILL_DEFINED_PREFIXES = (
    "R99",
    "R98",
    "R97",
    "R96",
    "R95",
)


def is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return False


def clean_scalar_string(value: Any) -> str | None:
    if is_missing_scalar(value):
        return None

    text = str(value).strip()

    if text.upper() in _BLANK_STRINGS:
        return ""

    return text


def parse_categorical(
    value: Any,
    *,
    valid_values: set[str] | None = None,
    unknown_values: set[str] | None = None,
    not_applicable_values: set[str] | None = None,
) -> ParsedValue:
    raw_text = clean_scalar_string(value)

    if raw_text is None:
        return ParsedValue(raw=value, normalized=None, state="missing")

    if raw_text == "":
        return ParsedValue(raw=value, normalized=None, state="blank")

    normalized = raw_text.strip()

    if not_applicable_values and normalized in not_applicable_values:
        return ParsedValue(raw=value, normalized=normalized, state="not_applicable")

    if unknown_values and normalized in unknown_values:
        return ParsedValue(raw=value, normalized=normalized, state="unknown")

    if valid_values is not None and normalized not in valid_values:
        return ParsedValue(
            raw=value,
            normalized=normalized,
            state="invalid",
            warnings=[f"categorical_value_not_in_registry:{normalized}"],
        )

    return ParsedValue(raw=value, normalized=normalized, state="valid")


def parse_numeric(value: Any) -> ParsedValue:
    raw_text = clean_scalar_string(value)

    if raw_text is None:
        return ParsedValue(raw=value, normalized=None, state="missing")

    if raw_text == "":
        return ParsedValue(raw=value, normalized=None, state="blank")

    text = raw_text.replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return ParsedValue(
            raw=value,
            normalized=None,
            state="unparseable",
            warnings=[f"numeric_unparseable:{raw_text}"],
        )

    return ParsedValue(raw=value, normalized=number, state="valid")


def parse_date(value: Any) -> ParsedDate:
    raw_text = clean_scalar_string(value)

    if raw_text is None:
        return ParsedDate(raw=value, normalized=None, state="missing")

    if raw_text == "":
        return ParsedDate(raw=value, normalized=None, state="blank")

    text = raw_text.strip()

    if text in _INVALID_DATE_PLACEHOLDERS:
        return ParsedDate(
            raw=value,
            normalized=None,
            state="invalid",
            warnings=[f"invalid_date_placeholder:{text}"],
        )

    formats = (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%Y%m%d",
        "%d%m%Y",
    )

    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue

        if parsed.year <= 0:
            return ParsedDate(
                raw=value,
                normalized=None,
                state="invalid",
                warnings=[f"invalid_date_year:{text}"],
            )

        return ParsedDate(raw=value, normalized=parsed.isoformat(), state="valid")

    return ParsedDate(
        raw=value,
        normalized=None,
        state="unparseable",
        warnings=[f"date_unparseable:{text}"],
    )


def normalize_icd10_text(value: Any) -> tuple[str | None, list[str]]:
    raw_text = clean_scalar_string(value)
    warnings: list[str] = []

    if raw_text is None:
        return None, warnings

    if raw_text == "":
        return "", warnings

    text = raw_text.upper().strip()

    if text.startswith("*"):
        warnings.append("icd_prefixed_asterisk")
        text = text[1:].strip()

    text = text.replace(" ", "")

    if len(text) > 3 and "." not in text:
        text = text[:3] + "." + text[3:]

    return text, warnings


def parse_icd10(value: Any) -> ParsedICD10:
    normalized, warnings = normalize_icd10_text(value)

    if normalized is None:
        return ParsedICD10(raw=value, normalized=None, state="missing", warnings=warnings)

    if normalized == "":
        return ParsedICD10(raw=value, normalized=None, state="blank", warnings=warnings)

    if not _ICD10_SHAPE.match(normalized):
        return ParsedICD10(
            raw=value,
            normalized=normalized,
            state="unparseable",
            warnings=[*warnings, f"icd10_shape_unparseable:{normalized}"],
        )

    compact = normalized.replace(".", "")

    if compact.startswith(_ILL_DEFINED_PREFIXES):
        return ParsedICD10(
            raw=value,
            normalized=normalized,
            state="ill_defined",
            warnings=[*warnings, f"icd10_ill_defined:{normalized}"],
        )

    return ParsedICD10(raw=value, normalized=normalized, state="valid", warnings=warnings)


def parse_datasus_age(value: Any) -> ParsedAge:
    """Decode common DATASUS IDADE format.

    DATASUS age fields often encode the first digit as a unit and the remaining
    digits as the value. Common SIM/SINASC/SIH-like convention:

    1xx = minutes
    2xx = hours
    3xx = months
    4xx = years
    5xx = years over 100 / extended years depending system

    This function is deliberately conservative. It preserves invalid or
    unparseable states instead of inventing an age.
    """
    raw_text = clean_scalar_string(value)

    if raw_text is None:
        return ParsedAge(
            raw=value,
            age_years=None,
            age_days=None,
            state="missing",
            source="datasus_idade",
        )

    if raw_text == "":
        return ParsedAge(
            raw=value,
            age_years=None,
            age_days=None,
            state="blank",
            source="datasus_idade",
        )

    text = re.sub(r"\D", "", raw_text)

    if len(text) < 2:
        return ParsedAge(
            raw=value,
            age_years=None,
            age_days=None,
            state="unparseable",
            source="datasus_idade",
            warnings=[f"age_unparseable:{raw_text}"],
        )

    unit = text[0]
    amount_text = text[1:]

    try:
        amount = int(amount_text)
    except ValueError:
        return ParsedAge(
            raw=value,
            age_years=None,
            age_days=None,
            state="unparseable",
            source="datasus_idade",
            warnings=[f"age_unparseable:{raw_text}"],
        )

    if unit == "1":
        age_days = amount / (24 * 60)
        age_years = age_days / 365.25
    elif unit == "2":
        age_days = amount / 24
        age_years = age_days / 365.25
    elif unit == "3":
        age_days = amount * 30.4375
        age_years = amount / 12
    elif unit == "4":
        age_years = float(amount)
        age_days = age_years * 365.25
    elif unit == "5":
        age_years = float(100 + amount)
        age_days = age_years * 365.25
    else:
        return ParsedAge(
            raw=value,
            age_years=None,
            age_days=None,
            state="invalid",
            source="datasus_idade",
            warnings=[f"age_unit_invalid:{unit}"],
        )

    return ParsedAge(
        raw=value,
        age_years=age_years,
        age_days=age_days,
        state="valid",
        source="datasus_idade",
    )


def normalize_municipality_code(value: Any) -> ParsedMunicipality:
    """Normalize municipality code to cod6/cod7 placeholders.

    This function does not validate a municipality against an official crosswalk.
    It only normalizes shape. The official cod6↔cod7 crosswalk will later turn
    shape-valid codes into verified geography.
    """
    raw_text = clean_scalar_string(value)

    if raw_text is None:
        return ParsedMunicipality(raw=value, cod6=None, cod7=None, state="missing")

    if raw_text == "":
        return ParsedMunicipality(raw=value, cod6=None, cod7=None, state="blank")

    digits = re.sub(r"\D", "", raw_text)

    if len(digits) == 6:
        return ParsedMunicipality(raw=value, cod6=digits, cod7=None, state="valid")

    if len(digits) == 7:
        return ParsedMunicipality(raw=value, cod6=digits[:6], cod7=digits, state="valid")

    return ParsedMunicipality(
        raw=value,
        cod6=None,
        cod7=None,
        state="invalid",
        warnings=[f"municipality_code_invalid_shape:{raw_text}"],
    )