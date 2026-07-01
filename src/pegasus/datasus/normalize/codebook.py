"""In-house DATASUS categorical codebook — the single translation authority.

Loads ``config/registries/datasus/datasus_codebook.yaml`` and resolves a coded DATASUS
value to its PegaSUS canonical value plus an MSD §2.3 state. This is the in-house
replacement for microdatasus's ``process_*()`` categorical translation: the code→
meaning dictionaries live here, deduplicated by *concept* and shared across schemas,
so the eventual in-house FTP processor (and every normalizer today) draws from one
place instead of re-deriving translations per system.

Two consumers:
  * ``translate(concept, code)`` — record-level (used by the SIM record oracle).
  * ``categorical_exprs(concept, code_expr)`` — vectorized (used by ``Cols.categorical``);
    kept here so the dictionary is applied identically in both paths.

State semantics (an improvement over microdatasus's silent 0/9→NA collapse):
  * blank/absent code            → ``missing``
  * code in the concept's values → ``valid``   (carries the canonical value)
  * code in ``unknown`` sentinels → ``unknown`` (value null — genuinely unknown, not absent)
  * anything else                → ``invalid`` (value null)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.config import load_yaml

_BLANK = {"", "NA", "NAN", "NULL", "NONE"}


@lru_cache(maxsize=4)
def load_codebook(registry_root: str = "config/registries") -> dict[str, Any]:
    """Load and cache the codebook YAML (concepts + per-system bindings + reference-
    table lookups)."""
    data = load_yaml(Path(registry_root) / "datasus/datasus_codebook.yaml")
    concepts = data.get("concepts", {})
    bindings = data.get("bindings", {})
    lookups = data.get("lookups", {})
    return {"concepts": concepts, "bindings": bindings, "lookups": lookups}


@lru_cache(maxsize=8)
def load_reference_table(table: str, registry_root: str = "config/registries") -> dict[str, str]:
    """Load a large open code→name reference table (tabCBO/tabNaturalidade/
    tabOcupacao, mirrored 1:1 from microdatasus's shipped data) as a ``{code: name}``
    dict. These are joins (occupation/country/municipality NAME lookups), distinct
    from the small closed `concepts` categorical dictionaries above."""
    path = Path(registry_root) / "datasus" / "reference" / f"{table}.parquet"
    df = pl.read_parquet(path)
    return dict(zip(df["code"].to_list(), df["name"].to_list()))


def lookup_name(table: str, code: Any, *, registry_root: str = "config/registries") -> str | None:
    """Record-level reference-table lookup: raw code → name, or None if absent."""
    text = _clean_code(code)
    if text is None:
        return None
    return load_reference_table(table, registry_root).get(text)


def lookup_expr(table: str, code: pl.Expr, *, registry_root: str = "config/registries") -> pl.Expr:
    """Vectorized twin of :func:`lookup_name`."""
    mapping = load_reference_table(table, registry_root)
    return code.replace_strict(mapping, default=None, return_dtype=pl.Utf8)


def _concept(name: str, registry_root: str = "config/registries") -> dict[str, Any]:
    concept = load_codebook(registry_root)["concepts"].get(name)
    if concept is None:
        raise KeyError(f"unknown codebook concept: {name}")
    return concept


def _strip_leading_zeros(code: str) -> str:
    """Int-normalize a purely-numeric code ("00" -> "0", "01" -> "1").

    Several SIH/CNES raw columns are fixed-width DBC fields (e.g. MARCA_UTI, ESPEC)
    that store a zero-padded code, while the case_match dictionary keys its bare
    digit form ("0"/"1"). Matching falls back to this normalized form so those rows
    don't silently read as 'invalid' despite being valid, common codes. Non-numeric
    codes ("A", "S", "D19"...) are returned unchanged."""
    return str(int(code)) if code.isdigit() else code


def concept_for(system: str, raw_column: str, registry_root: str = "config/registries") -> str | None:
    """Return the concept bound to ``raw_column`` for ``system``, or None."""
    return load_codebook(registry_root)["bindings"].get(system, {}).get(raw_column)


def _clean_code(code: Any) -> str | None:
    if code is None:
        return None
    text = str(code).strip()
    if text == "" or text.upper() in _BLANK:
        return None
    return text


def translate(concept: str, code: Any, *, registry_root: str = "config/registries") -> tuple[Any, str]:
    """Resolve one coded value → ``(canonical_value, state)`` for ``concept``."""
    spec = _concept(concept, registry_root)
    text = _clean_code(code)
    if text is None:
        return None, "missing"
    values = spec.get("values", {})
    unknown = set(spec.get("unknown", []))
    norm = _strip_leading_zeros(text)
    if text in values:
        return values[text], "valid"
    if norm in values:
        return values[norm], "valid"
    if text in unknown or norm in unknown:
        return None, "unknown"
    return None, "invalid"


def categorical_exprs(concept: str, code: pl.Expr, *, registry_root: str = "config/registries") -> tuple[pl.Expr, pl.Expr]:
    """Vectorized twin of :func:`translate`: ``(value_expr, state_expr)`` from a
    cleaned code column ``code`` (Utf8, blanks already nulled)."""
    spec = _concept(concept, registry_root)
    values = spec.get("values", {})
    unknown = list(spec.get("unknown", []))
    is_bool = spec.get("dtype") == "bool"
    return_dtype = pl.Boolean if is_bool else pl.Utf8

    # Zero-padded fixed-width DBC fields ("00"/"01") don't match the bare-digit
    # dictionary keys ("0"/"1") microdatasus's case_match uses -- fall back to the
    # int-normalized code when the exact code doesn't match.
    code_norm = (
        pl.when(code.str.contains(r"^\d+$"))
        .then(code.cast(pl.Int64, strict=False).cast(pl.Utf8))
        .otherwise(code)
    )

    if values:
        exact = code.replace_strict(values, default=None, return_dtype=return_dtype)
        normed = code_norm.replace_strict(values, default=None, return_dtype=return_dtype)
        value = pl.coalesce([exact, normed])
    else:
        value = pl.lit(None, dtype=return_dtype)

    known_codes = list(values.keys())
    state = (
        pl.when(code.is_null()).then(pl.lit("missing"))
        .when(code.is_in(known_codes) | code_norm.is_in(known_codes)).then(pl.lit("valid"))
        .when((code.is_in(unknown) | code_norm.is_in(unknown)) if unknown else pl.lit(False)).then(pl.lit("unknown"))
        .otherwise(pl.lit("invalid"))
    )
    return value, state
