"""Canonical demographic-axis category maps (MSD §2.12.2 / §3.7.4).

Single source of truth for aligning source-specific category codes (DATASUS sex codes,
SIDRA classification categories) onto shared canonical axis categories, so a stratified
numerator (deaths by sex) and denominator (population by sex) can be divided on the same
axis. Used by both the demographic population tensor loader and the count stratifier — no
hardcoded sex/race mappings in the engine.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml


REGISTRY_FILE = "demographic/demographic_axis_maps.yaml"

TOTAL = "__total__"
UNKNOWN = "__unknown__"

# Axes that are shared demographic strata: a stratified numerator on one of these axes
# must be divided by a denominator stratified on the SAME axis (not a marginal total).
DEMOGRAPHIC_AXES: frozenset[str] = frozenset({"sex", "age_group", "race"})


@lru_cache(maxsize=8)
def _axis_maps(root: str) -> dict[str, Any]:
    path = Path(root) / REGISTRY_FILE
    if not path.exists():
        return {}
    payload = load_yaml(path) or {}
    return dict(payload.get("axes", {}) or {})


@lru_cache(maxsize=8)
def _classification_axis(root: str) -> dict[str, str]:
    path = Path(root) / REGISTRY_FILE
    if not path.exists():
        return {}
    payload = load_yaml(path) or {}
    return {str(k): str(v) for k, v in (payload.get("classification_axis", {}) or {}).items()}


def axis_for_classification(
    classification_id: str, *, registry_root: str | Path = "config/registries"
) -> str | None:
    """Map a SIDRA classification id (e.g. '2' Sexo) to a canonical axis ('sex')."""
    return _classification_axis(str(registry_root)).get(str(classification_id))


def canonical_categories(axis: str, *, registry_root: str | Path = "config/registries") -> tuple[str, ...]:
    spec = _axis_maps(str(registry_root)).get(axis, {})
    return tuple(str(c) for c in (spec.get("canonical_categories") or []))


@lru_cache(maxsize=256)
def _source_category_map_cached(axis: str, source_system: str, root: str) -> dict[str, str]:
    spec = _axis_maps(root).get(axis, {})
    source_maps = spec.get("source_maps", {}) or {}
    return {str(k): str(v) for k, v in (source_maps.get(source_system) or {}).items()}


def source_category_map(
    axis: str, source_system: str, *, registry_root: str | Path = "config/registries"
) -> dict[str, str]:
    # Cached: this is called O(n_records) times during a tensor build and rebuilt the same small dict
    # every call (~4-8s of pure dict-comprehension at state scale). The returned dict is read-only for
    # every caller (map_category does .get); do NOT mutate it.
    return _source_category_map_cached(axis, source_system, str(registry_root))


def source_column(
    axis: str, source_system: str, *, registry_root: str | Path = "config/registries"
) -> str | None:
    """Normalized event column carrying ``axis`` for ``source_system`` (e.g. sex->'sex')."""
    spec = _axis_maps(str(registry_root)).get(axis, {})
    columns = spec.get("source_columns", {}) or {}
    value = columns.get(source_system)
    return str(value) if value else None


def map_category(
    axis: str,
    source_system: str,
    code: Any,
    *,
    registry_root: str | Path = "config/registries",
) -> str:
    """Map a raw source category code to its canonical axis category.

    Unknown codes map to ``__unknown__`` so they are excluded from stratified rates rather
    than silently merged into a real stratum.
    """
    mapping = source_category_map(axis, source_system, registry_root=registry_root)
    return mapping.get(str(code), UNKNOWN)


def age_group_sort_key(age_group: str) -> tuple[int, int]:
    """Numeric ordering key for canonical age_group labels.

    ``tuple(sorted(...))`` on the raw strings is alphabetical, not chronological:
    "age_10" < "age_2" lexically, which would silently scramble the tensor's age
    axis for any run with more than ~11 single-year categories present -- fatal
    for the aging loss (MSD §2.8.4), which assumes index ``a`` transitions into
    index ``a+1`` as one calendar year older. Sort with this key instead.
    """
    if age_group == "age_100_plus":
        return (0, 100)
    if age_group.startswith("age_"):
        try:
            return (0, int(age_group[len("age_"):]))
        except ValueError:
            pass
    # TOTAL / UNKNOWN / anything unrecognized: keep stable, ordered after real ages.
    return (1, 0)


def age_group_for_years(age_years: Any) -> str:
    """Bucket a single-year age into the canonical ``age_group`` category.

    Unlike ``sex``/``race`` this axis is a direct arithmetic bucketing of an
    integer field (SIM/SINASC/SIH already carry ``age_years``), not a source
    category-code crosswalk, so it needs no registry lookup -- the canonical
    labels themselves (``age_0``..``age_99``, ``age_100_plus``) are the SIDRA
    9606 single-year basis declared in ``demographic/demographic_axis_maps.yaml``.
    """
    if age_years is None:
        return TOTAL
    try:
        years = int(age_years)
    except (TypeError, ValueError):
        return UNKNOWN
    if years < 0:
        return UNKNOWN
    return "age_100_plus" if years >= 100 else f"age_{years}"


__all__ = [
    "DEMOGRAPHIC_AXES",
    "TOTAL",
    "UNKNOWN",
    "REGISTRY_FILE",
    "axis_for_classification",
    "canonical_categories",
    "source_category_map",
    "source_column",
    "map_category",
    "age_group_for_years",
    "age_group_sort_key",
]
