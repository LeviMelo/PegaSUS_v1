"""Output query — count/raw_field kinds (P3c shared field-tensor reader).

Pins the capability that was stubbed "not yet wired": resolve a quantity to a materialized field and
read its long-form tensor, with the denominator-principle unit flag and typed refusals (no silent
fallbacks).
"""
from __future__ import annotations

import polars as pl
import pytest

from pegasus.output.query.engine import QueryError, materialize_query
from pegasus.output.query.field_tensor import FieldResolutionError, resolve_field
from pegasus.output.query.spec import QuerySpec

FID = "a" * 64
FID2 = "b" * 64


def _make_bundle(tmp_path, *, unit="counts"):
    b = tmp_path / "bundle"
    (b / "Tables" / "efg_tensors").mkdir(parents=True)
    pl.DataFrame({
        "field_id": [FID, FID, FID],
        "technical_name": ["SIM.deaths", "SIM.deaths", "SIM.deaths"],
        "display_name": ["Deaths", "Deaths", "Deaths"],
        "carrier": ["Deaths", "Deaths", "Deaths"],
        "unit": [unit, unit, unit],
    }).write_parquet(b / "VariableDictionary.parquet")
    pl.DataFrame({
        "year": [2018, 2019, 2020],
        "municipality_cod6": ["270430", "270430", "270430"],
        "value": [100.0, 110.0, 120.0],
        "field_id": [FID, FID, FID],
        "field_name": ["SIM.deaths"] * 3,
        "operator": ["she_substrate_materialization"] * 3,
    }).write_parquet(b / "Tables" / "efg_tensors" / f"{FID}.parquet")
    return b


def test_raw_field_reads_materialized_values(tmp_path):
    b = _make_bundle(tmp_path)
    ds = materialize_query(b, QuerySpec(quantity=FID, kind="raw_field"))
    assert ds.kind == "raw_field"
    assert ds.frame.height == 3
    assert ds.frame["value"].to_list() == [100.0, 110.0, 120.0]
    assert ds.provenance["cell_dimensions"] == ["year", "municipality_cod6"]
    assert ds.provenance["field_id"] == FID


def test_resolves_by_technical_name(tmp_path):
    b = _make_bundle(tmp_path)
    ds = materialize_query(b, QuerySpec(quantity="SIM.deaths", kind="raw_field"))
    assert ds.provenance["field_id"] == FID


def test_count_flags_non_count_unit(tmp_path):
    counts = _make_bundle(tmp_path / "c", unit="counts")
    rate = _make_bundle(tmp_path / "r", unit="per_100k")
    assert materialize_query(counts, QuerySpec(quantity=FID, kind="count")).warnings == ()
    warned = materialize_query(rate, QuerySpec(quantity=FID, kind="count")).warnings
    assert any("not_a_count" in w for w in warned)


def test_filter_pushdown_and_missing_filter_column(tmp_path):
    b = _make_bundle(tmp_path)
    ds = materialize_query(b, QuerySpec(quantity=FID, kind="raw_field", filters={"year": 2019}))
    assert ds.frame.height == 1 and ds.frame["value"].to_list() == [110.0]
    with pytest.raises(QueryError, match="not in field tensor columns"):
        materialize_query(b, QuerySpec(quantity=FID, kind="raw_field", filters={"nonexistent": 1}))


def test_unknown_quantity_is_refused(tmp_path):
    b = _make_bundle(tmp_path)
    with pytest.raises(QueryError, match="not found"):
        materialize_query(b, QuerySpec(quantity="NOPE", kind="count"))


def test_ambiguous_quantity_is_refused(tmp_path):
    b = tmp_path / "bundle"
    (b / "Tables" / "efg_tensors").mkdir(parents=True)
    pl.DataFrame({  # two distinct fields sharing a technical_name
        "field_id": [FID, FID2],
        "technical_name": ["SIM.deaths", "SIM.deaths"],
        "display_name": ["Deaths A", "Deaths B"],
        "carrier": ["Deaths", "Deaths"],
        "unit": ["counts", "counts"],
    }).write_parquet(b / "VariableDictionary.parquet")
    with pytest.raises(FieldResolutionError, match="ambiguous"):
        resolve_field(b, "SIM.deaths")
