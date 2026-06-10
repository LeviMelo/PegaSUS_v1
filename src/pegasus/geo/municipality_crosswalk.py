from __future__ import annotations

import re
from dataclasses import dataclass


class MunicipalityCrosswalkError(ValueError):
    """Raised when a municipality code cannot be safely crosswalked."""


@dataclass(frozen=True)
class MunicipalityCode:
    source_code: str
    datasus_cod6: str | None
    ibge_cod7: str | None
    state: str
    method: str


# Minimal explicit AL smoke crosswalk. This is not a national geodata table.
# It exists to prevent accidental string equality between DATASUS six-digit
# municipality support and SIDRA/IBGE seven-digit municipality support.
AL_SMOKE_COD6_TO_COD7: dict[str, str] = {
    "270030": "2700300",  # Arapiraca
    "270430": "2704302",  # Maceió
}
AL_SMOKE_COD7_TO_COD6: dict[str, str] = {v: k for k, v in AL_SMOKE_COD6_TO_COD7.items()}


def _digits(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\D", "", str(value).strip())
    return text or None


def normalize_municipality_code(value: object) -> MunicipalityCode:
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
            datasus_cod6=AL_SMOKE_COD7_TO_COD6.get(digits, digits[:6]),
            ibge_cod7=digits,
            state="mapped" if digits in AL_SMOKE_COD7_TO_COD6 else "assumed_ibge_cod7",
            method="ibge_cod7_identity",
        )

    if len(digits) == 6:
        cod7 = AL_SMOKE_COD6_TO_COD7.get(digits)
        return MunicipalityCode(
            source_code=digits,
            datasus_cod6=digits,
            ibge_cod7=cod7,
            state="mapped" if cod7 else "unmapped_datasus_cod6",
            method="al_smoke_explicit_cod6_to_cod7" if cod7 else "none",
        )

    return MunicipalityCode(
        source_code=digits,
        datasus_cod6=None,
        ibge_cod7=None,
        state="invalid_length",
        method="none",
    )


def datasus_cod6_to_ibge_cod7(value: object, *, strict: bool = True) -> str | None:
    code = normalize_municipality_code(value)
    if code.ibge_cod7 is not None:
        return code.ibge_cod7
    if strict:
        raise MunicipalityCrosswalkError(
            f"Cannot map DATASUS municipality code {value!r} to IBGE/SIDRA cod7. state={code.state}"
        )
    return None


def ibge_cod7_to_datasus_cod6(value: object, *, strict: bool = True) -> str | None:
    code = normalize_municipality_code(value)
    if code.datasus_cod6 is not None:
        return code.datasus_cod6
    if strict:
        raise MunicipalityCrosswalkError(
            f"Cannot map IBGE/SIDRA municipality code {value!r} to DATASUS cod6. state={code.state}"
        )
    return None
