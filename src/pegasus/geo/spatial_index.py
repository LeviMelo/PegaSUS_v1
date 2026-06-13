"""Deterministic support index used by sparse geospatial modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialIndex:
    ids: tuple[str, ...]
    index_by_id: dict[str, int]


def build_spatial_index(ids) -> SpatialIndex:
    ordered = tuple(sorted(dict.fromkeys(str(value) for value in ids)))
    if not ordered:
        raise ValueError("spatial index requires at least one support ID")
    return SpatialIndex(ordered, {value: index for index, value in enumerate(ordered)})
