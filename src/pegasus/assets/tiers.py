"""Asset tiers + the scope-invariance invariant (MSD-III §VI.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegasus.assets.store import AssetStore
from pegasus.assets.version import NATIONAL_FULL_HISTORY, AssetVersion

# Scope-invariant assets: built once at national + full-history scope, sliced by queries.
FOUNDATIONAL: frozenset[str] = frozenset({
    "substrate",
    "population_tensor",
    "exposures",
    "spatial_graph",
    "disease_graph",
    "registry",
    "skeleton",
})


class ScopeInvarianceError(ValueError):
    """§VI.1 abort: a foundational asset was built at a reduced (query) scope.

    Building a scope-invariant asset at reduced scope is a *correctness* bug — it drops
    out-of-window anchors (e.g. the 2000/2010 censuses) and breaks national constraints
    (migration enclosure) — not merely an inefficiency.
    """


@dataclass(frozen=True)
class AssetView:
    """A query's view over a foundational asset: a slice, not a rebuild."""

    source_version: AssetVersion
    query_scope: dict[str, Any]

    @property
    def build_scope(self) -> str:
        return self.source_version.build_scope

    @property
    def input_manifest(self) -> dict[str, Any]:
        return self.source_version.input_manifest


def resolve_asset(store: AssetStore, name: str, query_scope: dict[str, Any]) -> AssetView:
    """Serve a foundational asset as a *view* sliced to ``query_scope`` — never a rebuild.

    Enforces the invariant: a foundational asset MUST have been built at
    ``national_full_history`` scope. A query selects a view; it does not (and must not)
    trigger a reduced-scope rebuild.
    """
    art = store.latest(name)
    if name in FOUNDATIONAL and art.build_scope != NATIONAL_FULL_HISTORY:
        raise ScopeInvarianceError(
            f"foundational asset {name!r} was built at scope {art.build_scope!r}, "
            f"not {NATIONAL_FULL_HISTORY!r} — a query must slice it, never rebuild it."
        )
    return AssetView(source_version=art, query_scope=dict(query_scope))


__all__ = ["FOUNDATIONAL", "ScopeInvarianceError", "AssetView", "resolve_asset"]
