from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import polars as pl

from pegasus.geo.uf import VALID_DATASUS_PREFIXES, resolve_uf_code

# Sentinel uf_prefix for national (all-UF) municipality validation: a cod6 is valid if its
# 2-digit prefix is any real Brazilian UF, rather than one specific state.
NATIONAL_UF_PREFIX = "*"


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


@dataclass(frozen=True)
class GeoScope:
    level: str
    uf: str | None
    datasus_uf_prefix: str | None
    ibge_uf_cod2: str | None
    expected_municipality_count: int | None = None
    municipality_cod6_allowlist: frozenset[str] | None = None
    municipality_cod7_allowlist: frozenset[str] | None = None

    @classmethod
    def from_uf(
        cls,
        uf: str,
        *,
        level: str = "municipality",
        expected_municipality_count: int | None = None,
        municipality_cod6_allowlist: Iterable[str] | None = None,
        municipality_cod7_allowlist: Iterable[str] | None = None,
    ) -> "GeoScope":
        resolved = resolve_uf_code(uf)
        return cls(
            level=level,
            uf=resolved.sigla,
            datasus_uf_prefix=resolved.datasus_prefix,
            ibge_uf_cod2=resolved.ibge_cod2,
            expected_municipality_count=expected_municipality_count,
            municipality_cod6_allowlist=(
                frozenset(str(value) for value in municipality_cod6_allowlist)
                if municipality_cod6_allowlist is not None
                else None
            ),
            municipality_cod7_allowlist=(
                frozenset(str(value) for value in municipality_cod7_allowlist)
                if municipality_cod7_allowlist is not None
                else None
            ),
        )

    @classmethod
    def national(cls, *, level: str = "municipality") -> "GeoScope":
        """National scope (SCALE-01): all 27 UFs / ~5,570 municipalities, no single-UF prefix.

        Municipality validation accepts any real Brazilian UF (``NATIONAL_UF_PREFIX``); the
        substrate/compile load all UFs' normalized data rather than one.
        """
        return cls(
            level=level,
            uf=None,
            datasus_uf_prefix=NATIONAL_UF_PREFIX,
            ibge_uf_cod2=None,
            expected_municipality_count=5570,
        )


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


def _require_uf_prefix(uf_prefix: str | None) -> str:
    if uf_prefix is None or str(uf_prefix).strip() == "":
        raise ValueError("DATASUS municipality validation requires an explicit uf_prefix.")
    return str(uf_prefix)


def is_valid_datasus_municipality_cod6(value: Any, *, uf_prefix: str | None) -> bool:
    uf_prefix = _require_uf_prefix(uf_prefix)
    code = normalize_cod6(value)
    if code is None:
        return False
    if len(code) != 6:
        return False
    if code[-4:] == "0000":
        return False
    if uf_prefix == NATIONAL_UF_PREFIX:
        # National scope: accept any municipality of any real Brazilian UF.
        return code[:2] in VALID_DATASUS_PREFIXES
    if not code.startswith(str(uf_prefix)):
        return False
    return True


def clean_datasus_municipalities(values: Iterable[Any], *, uf_prefix: str | None) -> tuple[list[str], list[str]]:
    uf_prefix = _require_uf_prefix(uf_prefix)
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


def municipality_support_from_values(values: Iterable[Any], *, uf_prefix: str | None) -> MunicipalitySupport:
    valid, invalid = clean_datasus_municipalities(values, uf_prefix=uf_prefix)
    return MunicipalitySupport(valid, invalid, _require_uf_prefix(uf_prefix))


def filter_valid_cod6_frame(df: pl.DataFrame, column: str, *, uf_prefix: str | None) -> pl.DataFrame:
    uf_prefix = _require_uf_prefix(uf_prefix)
    if column not in df.columns:
        return df
    return df.filter(
        pl.col(column)
        .cast(pl.Utf8)
        .map_elements(lambda value: is_valid_datasus_municipality_cod6(value, uf_prefix=uf_prefix), return_dtype=pl.Boolean)
    )
