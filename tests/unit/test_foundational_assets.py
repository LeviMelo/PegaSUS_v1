"""FAL-01 — foundational asset tiers + scope-invariance + versioning (MSD-III §VI)."""

from __future__ import annotations

import pytest

from pegasus.assets import (
    AssetStore,
    AssetVersion,
    ScopeInvarianceError,
    resolve_asset,
)


def test_query_slices_national_asset_never_rebuilds() -> None:
    store = AssetStore()
    store.put(AssetVersion(
        "population_tensor", "v2026.1", "national_full_history",
        input_manifest={"census": [2000, 2010, 2022], "estimapop": 2025},
        certification="verified",
    ))
    view = resolve_asset(store, "population_tensor", {"uf": "AL", "years": [2015, 2016]})

    # a state-scoped query SLICES the national asset — which still carries all 3 censuses,
    # including the 2000/2010 anchors outside the query window (the one-census-collapse fix)
    assert view.input_manifest["census"] == [2000, 2010, 2022]
    assert view.build_scope == "national_full_history"
    assert view.query_scope["uf"] == "AL"


def test_foundational_asset_built_at_query_scope_is_rejected() -> None:
    store = AssetStore()
    store.put(AssetVersion("population_tensor", "vbad", "state_AL_2015_2016"))
    with pytest.raises(ScopeInvarianceError):
        resolve_asset(store, "population_tensor", {"uf": "AL"})


def test_immutable_versioning_and_query_pinning() -> None:
    store = AssetStore()
    store.put(AssetVersion("population_tensor", "v1", "national_full_history"))
    store.put(AssetVersion("spatial_graph", "v1", "national_full_history"))
    store.put(AssetVersion("population_tensor", "v2", "national_full_history"))   # new vintage

    assert store.latest("population_tensor").version == "v2"
    assert store.get("population_tensor", "v1").version == "v1"      # prior vintage still reproducible
    assert store.pin(["population_tensor", "spatial_graph"]) == {
        "population_tensor": "v2", "spatial_graph": "v1",
    }
    with pytest.raises(ValueError):                                  # versions are immutable
        store.put(AssetVersion("population_tensor", "v2", "national_full_history"))


def test_non_foundational_artifact_may_be_query_scoped() -> None:
    store = AssetStore()
    store.put(AssetVersion("my_query_result", "v1", "state_AL"))     # a query artifact, not foundational
    view = resolve_asset(store, "my_query_result", {"uf": "AL"})
    assert view.build_scope == "state_AL"                            # no scope-invariance guard applies
