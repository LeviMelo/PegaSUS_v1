"""Deterministic ICD/CID hierarchy adapter (MSD-II §II.13.2).

Wraps ``simple-icd-10`` (WHO ICD-10) as the hierarchy backbone behind clean, cached,
deterministic callables — closure, ancestors/descendants, nearest-common-ancestor,
chapter/block resolution, dot normalization. The **Brazilian CID-10 remains the
authoritative code system for DATASUS data**; the WHO hierarchy is used for closure,
labels, and projection only. A code that exists in DATASUS but not in the WHO tree is
typed ``source_system_specific`` and is NEVER silently coerced to a WHO subcode
(§II.13.2 guard). This is the deterministic substrate the DiseaseGraph (§II.13.3) and the
Disease Concept Registry (§II.13.1) build on; it does not touch ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import simple_icd_10 as _icd


CodeSystemStatus = Literal["who_icd10", "source_system_specific", "malformed"]

# Canonical ICD-10 / CID-10 chapter ranges (roman numeral -> inclusive 3-char interval).
# Authoritative for chapter resolution of EVERY DATASUS code, including codes present in
# Brazilian CID-10 but absent from the WHO tree that ``simple-icd-10`` backs (e.g. A90/A91
# dengue, central to Brazilian arbovirus epidemiology). CID-10 is authoritative for DATASUS
# data (§II.13.2); the WHO hierarchy is used only where the code exists in it.
_CID10_CHAPTERS: tuple[tuple[str, str, str], ...] = (
    ("I", "A00", "B99"), ("II", "C00", "D48"), ("III", "D50", "D89"), ("IV", "E00", "E90"),
    ("V", "F00", "F99"), ("VI", "G00", "G99"), ("VII", "H00", "H59"), ("VIII", "H60", "H95"),
    ("IX", "I00", "I99"), ("X", "J00", "J99"), ("XI", "K00", "K93"), ("XII", "L00", "L99"),
    ("XIII", "M00", "M99"), ("XIV", "N00", "N99"), ("XV", "O00", "O99"), ("XVI", "P00", "P96"),
    ("XVII", "Q00", "Q99"), ("XVIII", "R00", "R99"), ("XIX", "S00", "T98"), ("XX", "V01", "Y98"),
    ("XXI", "Z00", "Z99"), ("XXII", "U00", "U99"),
)


def cid10_chapter(code: str) -> str | None:
    """Roman-numeral chapter of a code by CID-10 category range (works for WHO-absent codes)."""
    cat = _clean(code).replace(".", "")[:3]
    if len(cat) < 3 or not cat[0].isalpha():
        return None
    for chapter, lo, hi in _CID10_CHAPTERS:
        if lo <= cat <= hi:
            return chapter
    return None


@dataclass(frozen=True)
class ICDCodeInfo:
    """Provenance-typed view of a single code against the WHO ICD-10 hierarchy."""

    raw: str
    normalized: str | None            # dot form, e.g. "I21.9"; None if unparseable
    status: CodeSystemStatus
    is_leaf: bool
    chapter: str | None               # e.g. "IX"
    block: str | None                 # e.g. "I20-I25"
    category: str | None              # 3-char, e.g. "I21"


def _clean(code: str) -> str:
    return str(code).strip().upper().replace(" ", "")


def add_dot(code: str) -> str:
    """``"I219" -> "I21.9"`` (WHO normalization). Falls back to the cleaned input when the
    code is not a WHO item (e.g. a block range or a CID-10-only code)."""
    cleaned = _clean(code)
    try:
        if _icd.is_valid_item(cleaned):
            return _icd.add_dot(cleaned)
        dotted = _icd.add_dot(cleaned)
        return dotted if _icd.is_valid_item(dotted) else cleaned
    except Exception:
        return cleaned


def remove_dot(code: str) -> str:
    cleaned = _clean(code)
    try:
        return _icd.remove_dot(cleaned)
    except Exception:
        return cleaned.replace(".", "")


@lru_cache(maxsize=100_000)
def is_valid_who(code: str) -> bool:
    """True when the (dot-normalized) code is a node in the WHO ICD-10 tree."""
    try:
        return bool(_icd.is_valid_item(add_dot(code)))
    except Exception:
        return False


@lru_cache(maxsize=100_000)
def ancestors(code: str) -> tuple[str, ...]:
    """Ancestor chain from nearest parent up to chapter, dot-normalized. Empty if unknown."""
    if not is_valid_who(code):
        return ()
    try:
        return tuple(_icd.get_ancestors(add_dot(code)))
    except Exception:
        return ()


@lru_cache(maxsize=100_000)
def descendants(code: str) -> tuple[str, ...]:
    if not is_valid_who(code):
        return ()
    try:
        return tuple(_icd.get_descendants(add_dot(code)))
    except Exception:
        return ()


@lru_cache(maxsize=100_000)
def parent(code: str) -> str | None:
    if not is_valid_who(code):
        return None
    try:
        value = _icd.get_parent(add_dot(code))
        return value or None
    except Exception:
        return None


@lru_cache(maxsize=100_000)
def children(code: str) -> tuple[str, ...]:
    if not is_valid_who(code):
        return ()
    try:
        return tuple(_icd.get_children(add_dot(code)))
    except Exception:
        return ()


@lru_cache(maxsize=100_000)
def is_leaf(code: str) -> bool:
    if not is_valid_who(code):
        return False
    try:
        return bool(_icd.is_leaf(add_dot(code)))
    except Exception:
        return False


def nearest_common_ancestor(a: str, b: str) -> str | None:
    """Deepest node that is an ancestor-or-self of both ``a`` and ``b`` in the WHO tree
    (needed for disease-graph distance, §II.13.3). None when they share no ancestor or
    either is not a WHO code."""
    da = _clean_or_none(a)
    db = _clean_or_none(b)
    if da is None or db is None:
        return None
    # ancestor-or-self chains, nearest first
    chain_a = (add_dot(a),) + ancestors(a)
    set_b = set((add_dot(b),) + ancestors(b))
    for node in chain_a:
        if node in set_b:
            return node
    return None


def _clean_or_none(code: str) -> str | None:
    return add_dot(code) if is_valid_who(code) else None


@lru_cache(maxsize=100_000)
def code_info(code: str) -> ICDCodeInfo:
    """Full provenance-typed view. ``status='source_system_specific'`` for a well-formed
    code absent from the WHO tree (e.g. a Brazilian CID-10-only subcode) — never coerced."""
    cleaned = _clean(code)
    if not cleaned:
        return ICDCodeInfo(raw=code, normalized=None, status="malformed", is_leaf=False,
                           chapter=None, block=None, category=None)
    if not is_valid_who(cleaned):
        # Well-formed letter+digits shape but not a WHO node => source-system-specific
        # (a Brazilian CID-10 code absent from the WHO tree, e.g. A90 dengue). CID-10 is
        # authoritative here: resolve the chapter from the canonical CID-10 range table so
        # the code is still usable, but keep status source_system_specific (WHO closure/leaf
        # ops are unavailable) and DO NOT coerce it to a WHO subcode.
        looks_icd = len(cleaned) >= 3 and cleaned[0].isalpha() and cleaned[1].isdigit()
        status: CodeSystemStatus = "source_system_specific" if looks_icd else "malformed"
        return ICDCodeInfo(raw=code, normalized=(add_dot(cleaned) if looks_icd else None),
                           status=status, is_leaf=False,
                           chapter=(cid10_chapter(cleaned) if looks_icd else None), block=None,
                           category=(cleaned[:3] if looks_icd else None))
    normalized = add_dot(cleaned)
    chain = ancestors(cleaned)
    chapter = chain[-1] if chain else None
    block = next((c for c in chain if "-" in c), None)
    category = next((c for c in ([normalized, *chain]) if len(c.replace(".", "")) == 3 and "-" not in c), None)
    return ICDCodeInfo(raw=code, normalized=normalized, status="who_icd10",
                       is_leaf=is_leaf(cleaned), chapter=chapter, block=block, category=category)


__all__ = [
    "ICDCodeInfo",
    "CodeSystemStatus",
    "add_dot",
    "remove_dot",
    "is_valid_who",
    "ancestors",
    "descendants",
    "parent",
    "children",
    "is_leaf",
    "nearest_common_ancestor",
    "code_info",
]
