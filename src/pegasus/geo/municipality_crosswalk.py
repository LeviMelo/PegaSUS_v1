from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml


class MunicipalityCrosswalkError(ValueError):
    """Raised when a municipality code cannot be safely crosswalked."""


@dataclass(frozen=True)
class MunicipalityCode:
    source_code: str
    datasus_cod6: str | None
    ibge_cod7: str | None
    state: str
    method: str


DEFAULT_CROSSWALK_REGISTRY = Path("config/registries/municipality_crosswalk_codes.yaml")


@lru_cache(maxsize=8)
def load_municipality_crosswalk(path: str | Path = DEFAULT_CROSSWALK_REGISTRY) -> dict[str, str]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {}
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    entries = payload.get("entries") or []
    out: dict[str, str] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        cod6 = _digits(raw.get("datasus_cod6"))
        cod7 = _digits(raw.get("ibge_cod7"))
        if cod6 is None or cod7 is None:
            continue
        if len(cod6) != 6 or len(cod7) != 7:
            raise MunicipalityCrosswalkError(f"Invalid municipality crosswalk entry: {raw!r}")
        out[cod6] = cod7
    return out


def _digits(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\D", "", str(value).strip())
    return text or None


def normalize_municipality_code(
    value: object,
    *,
    cod6_to_cod7: Mapping[str, str] | None = None,
) -> MunicipalityCode:
    mapping = dict(cod6_to_cod7) if cod6_to_cod7 is not None else load_municipality_crosswalk()
    reverse = {v: k for k, v in mapping.items()}
    digits = _digits(value)
    if digits is None:
        return MunicipalityCode(
            source_code="",
            datasus_cod6=None,
            ibge_cod7=None,
            state="missing",
            method="none",
        )

    if len(digits) == 7:
        return MunicipalityCode(
            source_code=digits,
            datasus_cod6=reverse.get(digits, digits[:6]),
            ibge_cod7=digits,
            state="mapped" if digits in reverse else "assumed_ibge_cod7",
            method="ibge_cod7_identity",
        )

    if len(digits) == 6:
        cod7 = mapping.get(digits)
        return MunicipalityCode(
            source_code=digits,
            datasus_cod6=digits,
            ibge_cod7=cod7,
            state="mapped" if cod7 else "unmapped_datasus_cod6",
            method="registry_cod6_to_cod7" if cod7 else "none",
        )

    return MunicipalityCode(
        source_code=digits,
        datasus_cod6=None,
        ibge_cod7=None,
        state="invalid_length",
        method="none",
    )


def datasus_cod6_to_ibge_cod7(
    value: object,
    *,
    strict: bool = True,
    cod6_to_cod7: Mapping[str, str] | None = None,
) -> str | None:
    code = normalize_municipality_code(value, cod6_to_cod7=cod6_to_cod7)
    if code.ibge_cod7 is not None:
        return code.ibge_cod7
    if strict:
        raise MunicipalityCrosswalkError(
            f"Cannot map DATASUS municipality code {value!r} to IBGE/SIDRA cod7. state={code.state}"
        )
    return None


def ibge_cod7_to_datasus_cod6(
    value: object,
    *,
    strict: bool = True,
    cod6_to_cod7: Mapping[str, str] | None = None,
) -> str | None:
    code = normalize_municipality_code(value, cod6_to_cod7=cod6_to_cod7)
    if code.datasus_cod6 is not None:
        return code.datasus_cod6
    if strict:
        raise MunicipalityCrosswalkError(
            f"Cannot map IBGE/SIDRA municipality code {value!r} to DATASUS cod6. state={code.state}"
        )
    return None
