from __future__ import annotations

import polars as pl

from pegasus.efg.executor import _shift_year, _temporal_lag_of


class _Field:
    def __init__(self, support):
        self.support = support


def test_temporal_lag_of_reads_support_and_params() -> None:
    assert _temporal_lag_of(_Field({"temporal_lag": 1})) == 1
    assert _temporal_lag_of(_Field({"operator_params": {"temporal_lag": 2}})) == 2
    assert _temporal_lag_of(_Field({})) == 0
    assert _temporal_lag_of(_Field({"temporal_lag": 0})) == 0
    assert _temporal_lag_of(_Field({"temporal_lag": -3})) == 0


def test_shift_year_aligns_left_to_prior_year() -> None:
    df = pl.DataFrame({"year": [2015, 2016], "municipality_cod6": ["270010", "270010"], "value": [1.0, 2.0]})
    shifted = _shift_year(df, 1)
    # left(2015) is relabeled to 2016 so an inner join on year pairs it with right(2016):
    # the divergence output cell at year t then carries left(t-1).
    assert shifted["year"].to_list() == [2016, 2017]
    assert shifted["value"].to_list() == [1.0, 2.0]


def test_shift_year_noop_without_year_column_or_zero_lag() -> None:
    df = pl.DataFrame({"municipality_cod6": ["270010"], "value": [1.0]})
    assert _shift_year(df, 1).equals(df)
    df2 = pl.DataFrame({"year": [2016], "value": [1.0]})
    assert _shift_year(df2, 0).equals(df2)
