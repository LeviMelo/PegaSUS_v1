from __future__ import annotations

from pegasus.datasus.parsers import (
    normalize_municipality_code,
    parse_categorical,
    parse_datasus_age,
    parse_date,
    parse_icd10,
    parse_numeric,
)


def test_parse_date_accepts_yyyymmdd() -> None:
    parsed = parse_date("20220131")

    assert parsed.state == "valid"
    assert parsed.normalized == "2022-01-31"


def test_parse_date_rejects_invalid_placeholder() -> None:
    parsed = parse_date("00000000")

    assert parsed.state == "invalid"
    assert parsed.normalized is None
    assert "invalid_date_placeholder:00000000" in parsed.warnings


def test_parse_date_does_not_invent_dates() -> None:
    parsed = parse_date("not-a-date")

    assert parsed.state == "unparseable"
    assert parsed.normalized is None


def test_parse_icd10_valid_shape() -> None:
    parsed = parse_icd10("I219")

    assert parsed.state == "valid"
    assert parsed.normalized == "I21.9"


def test_parse_icd10_blank_is_not_r99() -> None:
    parsed = parse_icd10("")

    assert parsed.state == "blank"
    assert parsed.normalized is None


def test_parse_icd10_asterisk_is_field_level_warning() -> None:
    parsed = parse_icd10("*A419")

    assert parsed.state == "valid"
    assert parsed.normalized == "A41.9"
    assert "icd_prefixed_asterisk" in parsed.warnings


def test_parse_icd10_ill_defined_routes_observer() -> None:
    parsed = parse_icd10("R99")

    assert parsed.state == "ill_defined"
    assert parsed.normalized == "R99"
    assert any("icd10_ill_defined" in warning for warning in parsed.warnings)


def test_parse_datasus_age_years() -> None:
    parsed = parse_datasus_age("4035")

    assert parsed.state == "valid"
    assert parsed.age_years == 35


def test_parse_datasus_age_months() -> None:
    parsed = parse_datasus_age("3006")

    assert parsed.state == "valid"
    assert parsed.age_years == 0.5


def test_parse_datasus_age_invalid_unit() -> None:
    parsed = parse_datasus_age("9001")

    assert parsed.state == "invalid"
    assert parsed.age_years is None


def test_municipality_code_cod6() -> None:
    parsed = normalize_municipality_code("270430")

    assert parsed.state == "valid"
    assert parsed.cod6 == "270430"
    assert parsed.cod7 is None


def test_municipality_code_cod7() -> None:
    parsed = normalize_municipality_code("2704302")

    assert parsed.state == "valid"
    assert parsed.cod6 == "270430"
    assert parsed.cod7 == "2704302"


def test_municipality_code_invalid_shape() -> None:
    parsed = normalize_municipality_code("123")

    assert parsed.state == "invalid"


def test_parse_categorical_valid_registry() -> None:
    parsed = parse_categorical("1", valid_values={"1", "2"})

    assert parsed.state == "valid"
    assert parsed.normalized == "1"


def test_parse_categorical_invalid_registry() -> None:
    parsed = parse_categorical("9", valid_values={"1", "2"})

    assert parsed.state == "invalid"
    assert parsed.normalized == "9"


def test_parse_numeric_comma_decimal() -> None:
    parsed = parse_numeric("1,25")

    assert parsed.state == "valid"
    assert parsed.normalized == 1.25