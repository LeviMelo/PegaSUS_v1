"""The Foundational Asset Layer & lifecycle (MSD-III Part VI).

Fixes the class of error where a query's scope silently amputates a scope-invariant
asset (the "one-census collapse"). Foundational assets — the harmonized substrate, the
denominator-asset family, the spatial graph, the disease registry/graph, context cubes,
and the discovered link skeleton — are built ONCE at national + full-history scope,
versioned, and refreshed on cadence. A query selects a *view* over them; it MUST NOT
trigger a rebuild at reduced scope. Every query pins the foundation versions it consumed,
so a result is exactly reproducible from ``(foundation versions, query)``.
"""

from pegasus.assets.store import AssetStore
from pegasus.assets.tiers import (
    FOUNDATIONAL,
    AssetView,
    ScopeInvarianceError,
    resolve_asset,
)
from pegasus.assets.version import AssetVersion

__all__ = [
    "AssetVersion",
    "AssetStore",
    "FOUNDATIONAL",
    "AssetView",
    "ScopeInvarianceError",
    "resolve_asset",
]
