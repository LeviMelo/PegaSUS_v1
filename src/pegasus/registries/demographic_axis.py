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


REGISTRY_FILE = "demographic_axis_maps.yaml"

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


def source_category_map(
    axis: str, source_system: str, *, registry_root: str | Path = "config/registries"
) -> dict[str, str]:
    spec = _axis_maps(str(registry_root)).get(axis, {})
    source_maps = spec.get("source_maps", {}) or {}
    return {str(k): str(v) for k, v in (source_maps.get(source_system) or {}).items()}


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
]
