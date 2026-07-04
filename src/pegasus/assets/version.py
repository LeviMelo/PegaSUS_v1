"""Immutable, versioned foundational artifacts (MSD-III §VI.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NATIONAL_FULL_HISTORY = "national_full_history"


@dataclass(frozen=True)
class AssetVersion:
    """One immutable version of a foundational asset, with its input manifest.

    Example: ``population_tensor@v2026.1 ← {census: 2000,2010,2022; estimapop:…2025;
    code:<hash>; certification:verified}``. ``build_scope`` MUST be
    ``national_full_history`` for a foundational asset (§VI.1) — the invariant that keeps
    a query from amputating out-of-window anchors like the 2000/2010 censuses.
    """

    name: str
    version: str
    build_scope: str
    input_manifest: dict[str, Any] = field(default_factory=dict)
    certification: str = "unverified"
    payload: Any = None

    @property
    def is_national_full_history(self) -> bool:
        return self.build_scope == NATIONAL_FULL_HISTORY


__all__ = ["AssetVersion", "NATIONAL_FULL_HISTORY"]
