from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import polars as pl


AL_UF_PREFIX = "27"


@dataclass(frozen=True)
class MunicipalitySupport:
    municipalities: list[str]
    invalid_cod6: list[str]
    uf_prefix: str

    @property
    def valid_count(self) -> int:
        return len(self.municipalities)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_cod6)

    def as_support_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "municipalities": self.municipalities,
            "municipality_count_valid": self.valid_count,
            "invalid_municipality_cod6_count": self.invalid_count,
            "uf_prefix": self.uf_prefix,
            "municipality_code_policy": "datasus_cod6_valid_municipality_only",
        }
        if self.invalid_cod6:
            payload["invalid_municipality_cod6"] = self.invalid_cod6
            payload["invalid_municipality_code_policy"] = "excluded_from_municipality_support_not_from_raw_audit"
        return payload


def normalize_cod6(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[:6]
    return digits or None


def is_valid_datasus_municipality_cod6(value: Any, *, uf_prefix: str = AL_UF_PREFIX) -> bool:
    code = normalize_cod6(value)
    if code is None:
        return False
    if len(code) != 6:
        return False
    if not code.startswith(str(uf_prefix)):
        return False
    if code.endswith("0000"):
        return False
    if code[-4:] == "0000":
        return False
    return True


def clean_datasus_municipalities(values: Iterable[Any], *, uf_prefix: str = AL_UF_PREFIX) -> tuple[list[str], list[str]]:
    valid: set[str] = set()
    invalid: set[str] = set()
    for value in values:
        code = normalize_cod6(value)
        if code is None:
            continue
        if is_valid_datasus_municipality_cod6(code, uf_prefix=uf_prefix):
            valid.add(code)
        else:
            invalid.add(code)
    return sorted(valid), sorted(invalid)


def municipality_support_from_values(values: Iterable[Any], *, uf_prefix: str = AL_UF_PREFIX) -> MunicipalitySupport:
    valid, invalid = clean_datasus_municipalities(values, uf_prefix=uf_prefix)
    return MunicipalitySupport(valid, invalid, uf_prefix)


def filter_valid_cod6_frame(df: pl.DataFrame, column: str, *, uf_prefix: str = AL_UF_PREFIX) -> pl.DataFrame:
    if column not in df.columns:
        return df
    return df.filter(
        pl.col(column)
        .cast(pl.Utf8)
        .map_elements(lambda value: is_valid_datasus_municipality_cod6(value, uf_prefix=uf_prefix), return_dtype=pl.Boolean)
    )
