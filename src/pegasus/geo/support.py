from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7


class SupportAlignmentError(ValueError):
    """Raised when numerator/denominator support cannot be safely aligned."""


@dataclass(frozen=True)
class SupportAlignmentResult:
    aligned: bool
    reason: str
    numerator_geography: str | None
    denominator_geography: str | None
    numerator_years: list[int | str]
    denominator_years: list[int | str]
    numerator_municipalities_source: list[str]
    denominator_municipalities_source: list[str]
    numerator_municipalities_ibge_cod7: list[str]
    denominator_municipalities_ibge_cod7: list[str]
    common_municipalities_ibge_cod7: list[str]
    missing_from_denominator_ibge_cod7: list[str]
    missing_from_numerator_ibge_cod7: list[str]
    crosswalk: str

    def model(self) -> dict[str, Any]:
        return asdict(self)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _years(support: dict[str, Any]) -> list[int | str]:
    out = []
    for value in _as_list(support.get("years")):
        if isinstance(value, int):
            out.append(value)
        else:
            text = str(value)
            out.append(int(text) if text.isdigit() else text)
    return sorted(out, key=lambda x: str(x))


def _municipalities(support: dict[str, Any]) -> list[str]:
    return sorted(str(x) for x in _as_list(support.get("municipalities")) if x is not None)


def _municipalities_as_ibge_cod7(
    support: dict[str, Any],
    axes: dict[str, Any],
    *,
    strict: bool,
) -> list[str]:
    geography = axes.get("geography")
    codes = _municipalities(support)

    if not codes:
        return []

    if geography == "mun_residence_cod6":
        return sorted(datasus_cod6_to_ibge_cod7(code, strict=strict) for code in codes)

    if geography in {"municipality", "municipality_ibge_cod7", None}:
        mapped = []
        for code in codes:
            if len(code) == 7 and code.isdigit():
                mapped.append(code)
            else:
                mapped.append(datasus_cod6_to_ibge_cod7(code, strict=strict))
        return sorted(mapped)

    if strict:
        raise SupportAlignmentError(f"Unsupported geography axis for municipality support alignment: {geography!r}")
    return []


def align_municipality_year_support(
    *,
    numerator_support: dict[str, Any],
    numerator_axes: dict[str, Any],
    denominator_support: dict[str, Any],
    denominator_axes: dict[str, Any],
    strict_crosswalk: bool = True,
) -> SupportAlignmentResult:
    numerator_years = _years(numerator_support)
    denominator_years = _years(denominator_support)
    numerator_source = _municipalities(numerator_support)
    denominator_source = _municipalities(denominator_support)

    try:
        numerator_cod7 = _municipalities_as_ibge_cod7(numerator_support, numerator_axes, strict=strict_crosswalk)
        denominator_cod7 = _municipalities_as_ibge_cod7(denominator_support, denominator_axes, strict=strict_crosswalk)
    except Exception as exc:
        return SupportAlignmentResult(
            aligned=False,
            reason=str(exc),
            numerator_geography=numerator_axes.get("geography"),
            denominator_geography=denominator_axes.get("geography"),
            numerator_years=numerator_years,
            denominator_years=denominator_years,
            numerator_municipalities_source=numerator_source,
            denominator_municipalities_source=denominator_source,
            numerator_municipalities_ibge_cod7=[],
            denominator_municipalities_ibge_cod7=[],
            common_municipalities_ibge_cod7=[],
            missing_from_denominator_ibge_cod7=[],
            missing_from_numerator_ibge_cod7=[],
            crosswalk="datasus_cod6_to_ibge_cod7",
        )

    num_year_set = set(numerator_years)
    den_year_set = set(denominator_years)
    num_set = set(numerator_cod7)
    den_set = set(denominator_cod7)

    missing_from_denominator = sorted(num_set - den_set)
    missing_from_numerator = sorted(den_set - num_set)
    common = sorted(num_set & den_set)

    aligned = (
        num_year_set == den_year_set
        and bool(num_set)
        and num_set == den_set
    )
    if num_year_set != den_year_set:
        reason = "time_support_mismatch"
    elif not num_set:
        reason = "empty_numerator_municipality_support"
    elif num_set != den_set:
        reason = "municipality_support_mismatch"
    else:
        reason = "aligned"

    return SupportAlignmentResult(
        aligned=aligned,
        reason=reason,
        numerator_geography=numerator_axes.get("geography"),
        denominator_geography=denominator_axes.get("geography"),
        numerator_years=numerator_years,
        denominator_years=denominator_years,
        numerator_municipalities_source=numerator_source,
        denominator_municipalities_source=denominator_source,
        numerator_municipalities_ibge_cod7=sorted(num_set),
        denominator_municipalities_ibge_cod7=sorted(den_set),
        common_municipalities_ibge_cod7=common,
        missing_from_denominator_ibge_cod7=missing_from_denominator,
        missing_from_numerator_ibge_cod7=missing_from_numerator,
        crosswalk="datasus_cod6_to_ibge_cod7",
    )


def assert_municipality_year_support_aligned(
    *,
    numerator_support: dict[str, Any],
    numerator_axes: dict[str, Any],
    denominator_support: dict[str, Any],
    denominator_axes: dict[str, Any],
) -> SupportAlignmentResult:
    result = align_municipality_year_support(
        numerator_support=numerator_support,
        numerator_axes=numerator_axes,
        denominator_support=denominator_support,
        denominator_axes=denominator_axes,
        strict_crosswalk=True,
    )
    if not result.aligned:
        raise SupportAlignmentError(
            "Illegal denominator attachment: numerator and denominator municipality-year support are not aligned. "
            f"reason={result.reason}; "
            f"numerator_cod7={result.numerator_municipalities_ibge_cod7}; "
            f"denominator_cod7={result.denominator_municipalities_ibge_cod7}; "
            f"numerator_years={result.numerator_years}; denominator_years={result.denominator_years}"
        )
    return result
