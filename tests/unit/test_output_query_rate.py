"""Output query — the `rate` kind (P3d). A rate is a SELECTION of the EFG's already-materialized RN
ratio field (carrier num/denom), never a query-time recompute (spec §2), so it cannot drift from the
EFG's canonical rate. standardized_rate stays refused (age-standardization is its own engine).
"""
from __future__ import annotations

import polars as pl
import pytest

from pegasus.output.query.engine import QueryError, materialize_query
from pegasus.output.query.spec import QuerySpec

NUM = "a" * 64          # numerator field_id (carrier "Deaths")
RATE = "b" * 64         # RN rate field_id (carrier "Deaths/Population")


def _make_bundle(tmp_path, *, with_rate=True):
    b = tmp_path / "bundle"
    (b / "Tables" / "efg_tensors").mkdir(parents=True)
    rows = [
        {"field_id": NUM, "technical_name": "SIM.deaths", "display_name": "Deaths",
         "carrier": "Deaths", "unit": "counts"},
    ]
    if with_rate:
        rows.append({"field_id": RATE, "technical_name": "deaths_rate", "display_name": "Death rate",
                     "carrier": "Deaths/Population", "unit": "rate"})
    pl.DataFrame(rows).write_parquet(b / "VariableDictionary.parquet")
    pl.DataFrame({
        "year": [2019, 2020], "municipality_cod6": ["270430", "270430"],
        "value": [50.0, 60.0], "field_id": [NUM, NUM],
        "field_name": ["SIM.deaths"] * 2, "operator": ["she_substrate_materialization"] * 2,
    }).write_parquet(b / "Tables" / "efg_tensors" / f"{NUM}.parquet")
    if with_rate:
        pl.DataFrame({
            "year": [2019, 2020], "municipality_cod6": ["270430", "270430"],
            "value": [0.005, 0.006], "field_id": [RATE, RATE],
            "field_name": ["deaths_rate"] * 2, "operator": ["rn_ratio"] * 2,
        }).write_parquet(b / "Tables" / "efg_tensors" / f"{RATE}.parquet")
    return b


def test_rate_selects_materialized_rn_field(tmp_path):
    b = _make_bundle(tmp_path)
    ds = materialize_query(b, QuerySpec(quantity=NUM, kind="rate", denominator="resident_population"))
    assert ds.kind == "rate"
    assert ds.frame["value"].to_list() == [0.005, 0.006]            # the RN rate, not the count
    assert ds.provenance["rate_carrier"] == "Deaths/Population"
    assert ds.provenance["denominator"] == {"id": "resident_population", "carrier": "Population"}
    assert "query-time division" in ds.provenance["note"]
    assert ds.warnings == ()                                        # unit is "rate"


def test_rate_defaults_to_resident_population(tmp_path):
    b = _make_bundle(tmp_path)
    ds = materialize_query(b, QuerySpec(quantity=NUM, kind="rate"))  # no denominator -> resident pop
    assert ds.provenance["rate_carrier"] == "Deaths/Population"


def test_rate_filters_pushdown(tmp_path):
    b = _make_bundle(tmp_path)
    ds = materialize_query(b, QuerySpec(quantity=NUM, kind="rate", filters={"year": 2020}))
    assert ds.frame.height == 1 and ds.frame["value"].to_list() == [0.006]


def test_rate_refuses_when_no_materialized_rn_field(tmp_path):
    b = _make_bundle(tmp_path, with_rate=False)   # numerator present, no RN rate field
    with pytest.raises(QueryError, match="no materialized RN rate field"):
        materialize_query(b, QuerySpec(quantity=NUM, kind="rate", denominator="resident_population"))


def test_standardized_rate_is_refused_to_the_standardization_engine(tmp_path):
    b = _make_bundle(tmp_path)
    with pytest.raises(QueryError, match="age_standardization engine"):
        materialize_query(b, QuerySpec(quantity="mortality_all_cause", kind="standardized_rate"))
