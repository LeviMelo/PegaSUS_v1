"""Executable-authority callable resolver (MSD-II §II.1, MII-REG-07).

Registries are executable authority: every ``decoder``/``parser``/``adapter``
name declared in a registry MUST resolve to a real Python callable through the
single resolver in this module. Previously resolution was split between a generic
module scan in ``declarative_normalize`` and special-cased dispatch buried inside
``_decode_registered_value`` (so ``decode_sex`` and the compound SIM decoders
looked "unresolved" to any external validator). This module is the one place that
knows the full mapping, so the cross-registry validator can *prove* that no
registry references a callable that does not exist.

Resolution order for a declared name:
1. explicit alias/compound map (``_EXPLICIT``) — synonyms and multi-raw-field
   compound decoders whose implementation lives outside ``decoders.py``;
2. a scan of the registered decoder/parser modules (``_MODULES``), trying the
   name and the common ``decode_``/``parse_`` prefixes.

Callables are resolved lazily (via ``importlib``) so this module has no import
cycle with ``declarative_normalize`` or the decoder modules.
"""

from __future__ import annotations

import importlib
import re
from functools import lru_cache
from typing import Callable


# Modules scanned, in order, for a bare callable name.
_MODULES: tuple[str, ...] = (
    "pegasus.datasus.decoders",
    "pegasus.datasus.icd_parser",
)

# Explicit name → (module, attribute) map for callables that are NOT discoverable
# by a bare module scan: registry synonyms and the compound multi-raw-field
# decoders whose implementation lives in ``declarative_normalize``.
_EXPLICIT: dict[str, tuple[str, str]] = {
    # Synonym: the registry calls it ``decode_sex``; the single authority is
    # ``decode_datasus_sex`` (MSD §2.3 shared sex decoder).
    "decode_sex": ("pegasus.datasus.decoders", "decode_datasus_sex"),
    "decode_datasus_sex": ("pegasus.datasus.decoders", "decode_datasus_sex"),
    # Compound decoders (multiple raw fields → multiple canonical fields).
    "decode_sim_cause_chain": ("pegasus.datasus.declarative_normalize", "_sim_cause_chain_decode"),
    "decode_sim_associated": ("pegasus.datasus.declarative_normalize", "_sim_associated_decode"),
}


def normalize_callable_name(name: object) -> str | None:
    """Normalize a raw registry callable string to a canonical identifier token."""
    if name is None:
        return None
    text = str(name).strip()
    if not text or text.lower() in {"none", "null", "na"}:
        return None
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


@lru_cache(maxsize=512)
def resolve_callable(name: str | None) -> Callable | None:
    """Resolve a registry-declared callable name to a Python callable, or ``None``.

    This is the single resolver mandated by MSD-II §II.1. It covers every name a
    registry may reference: explicit synonyms/compound decoders first, then a scan
    of the registered decoder/parser modules.
    """
    canonical = normalize_callable_name(name)
    if not canonical:
        return None

    explicit = _EXPLICIT.get(canonical)
    if explicit is not None:
        module_name, attr = explicit
        try:
            fn = getattr(importlib.import_module(module_name), attr, None)
        except Exception:
            fn = None
        if callable(fn):
            return fn

    candidates = (
        canonical,
        canonical.lower(),
        canonical.upper(),
        f"decode_{canonical}",
        f"decode_{canonical.lower()}",
        f"parse_{canonical}",
        f"parse_{canonical.lower()}",
    )
    for module_name in _MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for candidate in candidates:
            fn = getattr(module, candidate, None)
            if callable(fn):
                return fn
    return None


def resolve_callable_strict(name: str | None) -> Callable:
    """Resolve a callable or raise ``LookupError`` — the fail-loud form."""
    fn = resolve_callable(name)
    if fn is None:
        raise LookupError(f"unresolved_registry_callable:{name}")
    return fn


__all__ = ["resolve_callable", "resolve_callable_strict", "normalize_callable_name"]
