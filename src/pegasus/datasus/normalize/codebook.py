"""In-house DATASUS categorical codebook — the single translation authority.

Loads ``config/registries/datasus_codebook.yaml`` and resolves a coded DATASUS
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
    """Load and cache the codebook YAML (concepts + per-system bindings)."""
    data = load_yaml(Path(registry_root) / "datasus_codebook.yaml")
    concepts = data.get("concepts", {})
    bindings = data.get("bindings", {})
    return {"concepts": concepts, "bindings": bindings}


def _concept(name: str, registry_root: str = "config/registries") -> dict[str, Any]:
    concept = load_codebook(registry_root)["concepts"].get(name)
    if concept is None:
        raise KeyError(f"unknown codebook concept: {name}")
    return concept


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
    if text in values:
        return values[text], "valid"
    if text in set(spec.get("unknown", [])):
        return None, "unknown"
    return None, "invalid"


def categorical_exprs(concept: str, code: pl.Expr, *, registry_root: str = "config/registries") -> tuple[pl.Expr, pl.Expr]:
    """Vectorized twin of :func:`translate`: ``(value_expr, state_expr)`` from a
    cleaned code column ``code`` (Utf8, blanks already nulled)."""
    spec = _concept(concept, registry_root)
    values = spec.get("values", {})
    unknown = list(spec.get("unknown", []))
    is_bool = spec.get("dtype") == "bool"

    if values:
        value = code.replace_strict(values, default=None, return_dtype=pl.Boolean if is_bool else pl.Utf8)
    else:
        value = pl.lit(None, dtype=pl.Boolean if is_bool else pl.Utf8)

    state = (
        pl.when(code.is_null()).then(pl.lit("missing"))
        .when(code.is_in(list(values.keys()))).then(pl.lit("valid"))
        .when(code.is_in(unknown) if unknown else pl.lit(False)).then(pl.lit("unknown"))
        .otherwise(pl.lit("invalid"))
    )
    return value, state
