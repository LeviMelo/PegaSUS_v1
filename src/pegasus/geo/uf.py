from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UFCode:
    sigla: str
    ibge_cod2: str

    @property
    def datasus_prefix(self) -> str:
        return self.ibge_cod2


_UF_BY_SIGLA: dict[str, UFCode] = {
    "RO": UFCode("RO", "11"),
    "AC": UFCode("AC", "12"),
    "AM": UFCode("AM", "13"),
    "RR": UFCode("RR", "14"),
    "PA": UFCode("PA", "15"),
    "AP": UFCode("AP", "16"),
    "TO": UFCode("TO", "17"),
    "MA": UFCode("MA", "21"),
    "PI": UFCode("PI", "22"),
    "CE": UFCode("CE", "23"),
    "RN": UFCode("RN", "24"),
    "PB": UFCode("PB", "25"),
    "PE": UFCode("PE", "26"),
    "AL": UFCode("AL", "27"),
    "SE": UFCode("SE", "28"),
    "BA": UFCode("BA", "29"),
    "MG": UFCode("MG", "31"),
    "ES": UFCode("ES", "32"),
    "RJ": UFCode("RJ", "33"),
    "SP": UFCode("SP", "35"),
    "PR": UFCode("PR", "41"),
    "SC": UFCode("SC", "42"),
    "RS": UFCode("RS", "43"),
    "MS": UFCode("MS", "50"),
    "MT": UFCode("MT", "51"),
    "GO": UFCode("GO", "52"),
    "DF": UFCode("DF", "53"),
}

_UF_BY_COD2: dict[str, UFCode] = {value.ibge_cod2: value for value in _UF_BY_SIGLA.values()}

# The 27 federative units — used for national-scale (all-UF) acquisition and validation.
ALL_UF_SIGLAS: tuple[str, ...] = tuple(_UF_BY_SIGLA)
VALID_DATASUS_PREFIXES: frozenset[str] = frozenset(_UF_BY_COD2)


def is_valid_datasus_prefix(prefix: str | None) -> bool:
    """True when ``prefix`` is one of the 27 valid IBGE UF codes (a real Brazilian state)."""
    return prefix is not None and str(prefix).strip() in VALID_DATASUS_PREFIXES


def resolve_uf_code(value: str) -> UFCode:
    key = str(value).strip().upper()
    if key in _UF_BY_SIGLA:
        return _UF_BY_SIGLA[key]
    if key in _UF_BY_COD2:
        return _UF_BY_COD2[key]
    raise ValueError(f"Unknown Brazilian UF code: {value!r}")


def uf_from_datasus_cod6(municipality_cod6: str | None) -> str | None:
    if not municipality_cod6:
        return None
    prefix = str(municipality_cod6).strip()[:2]
    uf = _UF_BY_COD2.get(prefix)
    return uf.sigla if uf is not None else None
