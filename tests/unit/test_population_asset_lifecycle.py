"""FAL-POP-VER: population tensor as a build-once, versioned, sliced foundational asset (§VI).

The tensor is built once at national + full-history scope, stored immutably, and served to every query
as a SLICE — never rebuilt at reduced scope. These tests pin: build-once idempotence (same inputs =>
reuse, not rebuild), the query slice, the scope-invariance guard, and reduced-scope builds NOT being
stored as foundational.
"""

from pathlib import Path

import polars as pl
import pytest

from pegasus.assets import (
    NATIONAL_FULL_HISTORY,
    PersistentAssetStore,
    ScopeInvarianceError,
    population_tensor_input_identity,
    resolve_asset,
    resolve_or_build_population_tensor,
    slice_population_tensor,
)


class _FakeBuild:
    def __init__(self, path: Path):
        self.path = path

    def as_manifest(self):
        return {"output_path": str(self.path), "periods": ["2000", "2010", "2022"]}


def _write_tensor(path: Path, years=(2000, 2010, 2015, 2022, 2024)):
    rows = [{"year": y, "municipality_cod6": "270430", "age_group": "total",
             "sex": "total", "race": "total", "value": 1000.0 + y} for y in years]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _build_fn_factory(counter: list[int]):
    def build_fn(out_path: Path):
        counter[0] += 1
        _write_tensor(out_path)
        return _FakeBuild(out_path)
    return build_fn


def test_slice_population_tensor_filters_window(tmp_path: Path):
    src = tmp_path / "full.parquet"
    _write_tensor(src)
    out = slice_population_tensor(src, 2010, 2022, tmp_path / "sliced.parquet")
    years = sorted(pl.read_parquet(out)["year"].unique().to_list())
    assert years == [2010, 2015, 2022]  # 2000 and 2024 excluded


def test_build_once_then_reuse(tmp_path: Path):
    store = PersistentAssetStore(tmp_path / "assets")
    identity = population_tensor_input_identity(input_hashes={"strata": "h1", "totals": "h2"}, mode="independent_denominator")
    counter = [0]
    common = dict(
        store=store, input_identity=identity, build_scope=NATIONAL_FULL_HISTORY,
        build_fn=_build_fn_factory(counter), input_manifest={"census": [2000, 2010, 2022]},
        max_year=2024, query_window=(2010, 2022), run_dir=tmp_path / "run1", solver_mode="independent_denominator",
    )
    first = resolve_or_build_population_tensor(**common)
    assert first["reused"] is False and counter[0] == 1
    assert first["version"] == "v2024.1"

    # second run, identical inputs -> reuse the stored version, no rebuild
    common2 = {**common, "run_dir": tmp_path / "run2"}
    second = resolve_or_build_population_tensor(**common2)
    assert second["reused"] is True and counter[0] == 1  # build_fn NOT called again
    assert second["version"] == "v2024.1"
    # the slice is produced fresh in the new run dir and honors the window
    years = sorted(pl.read_parquet(second["sliced_path"])["year"].unique().to_list())
    assert min(years) >= 2010 and max(years) <= 2022


def test_persistence_across_store_instances(tmp_path: Path):
    root = tmp_path / "assets"
    identity = population_tensor_input_identity(input_hashes={"strata": "h1"}, mode="independent_denominator")
    counter = [0]
    resolve_or_build_population_tensor(
        store=PersistentAssetStore(root), input_identity=identity, build_scope=NATIONAL_FULL_HISTORY,
        build_fn=_build_fn_factory(counter), input_manifest={}, max_year=2024,
        query_window=(None, None), run_dir=tmp_path / "run1", solver_mode="independent_denominator",
    )
    # a fresh store instance loads the on-disk index and reuses the version
    reopened = PersistentAssetStore(root)
    assert reopened.has("population_tensor")
    view = resolve_asset(reopened, "population_tensor", {"start_year": 2021, "end_year": 2022})
    assert view.build_scope == NATIONAL_FULL_HISTORY
    second = resolve_or_build_population_tensor(
        store=reopened, input_identity=identity, build_scope=NATIONAL_FULL_HISTORY,
        build_fn=_build_fn_factory(counter), input_manifest={}, max_year=2024,
        query_window=(None, None), run_dir=tmp_path / "run2", solver_mode="independent_denominator",
    )
    assert second["reused"] is True and counter[0] == 1


def test_reduced_scope_build_not_stored_as_foundational(tmp_path: Path):
    """A state/dev build must NOT be stored as a foundational version (else resolve_asset's scope
    guard would be violated by a query-scope artifact)."""
    store = PersistentAssetStore(tmp_path / "assets")
    identity = population_tensor_input_identity(input_hashes={"strata": "hAL"}, mode="independent_denominator")
    counter = [0]
    result = resolve_or_build_population_tensor(
        store=store, input_identity=identity, build_scope="state_full_history",
        build_fn=_build_fn_factory(counter), input_manifest={}, max_year=2022,
        query_window=(2021, 2022), run_dir=tmp_path / "run", solver_mode="independent_denominator",
    )
    assert result["reused"] is False and result["version"] == "run_local"
    assert not store.has("population_tensor")  # nothing foundational stored
    # and resolve_asset must refuse to serve a non-existent foundational asset
    with pytest.raises(KeyError):
        resolve_asset(store, "population_tensor", {"start_year": 2021, "end_year": 2022})


def test_scope_guard_rejects_reduced_scope_foundational(tmp_path: Path):
    """If a reduced-scope version were ever mislabeled foundational, resolve_asset aborts (§VI.1)."""
    from pegasus.assets import AssetVersion
    store = PersistentAssetStore(tmp_path / "assets")
    store.put(AssetVersion(name="population_tensor", version="v2022.1", build_scope="state_full_history"))
    with pytest.raises(ScopeInvarianceError):
        resolve_asset(store, "population_tensor", {"start_year": 2021})
