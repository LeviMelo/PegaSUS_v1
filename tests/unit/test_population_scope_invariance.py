"""FAL-POP increment 1: the population tensor's census anchors are scope-invariant (§II.4/§VI.1).

MSD-III §II.4: "All three censuses (2000, 2010, 2022) MUST be ingested regardless of a query's time
window; a query slices, never re-scopes, the tensor." This pins that the census-anchor selection
does not depend on the intent window — the bug that left a 2021-22 run anchored to 2022 alone.
"""

from __future__ import annotations

from types import SimpleNamespace

from pegasus.workflows.pipeline import SIDRA_POPULATION_TABLE, _all_census_periods


def _meta(periods) -> SimpleNamespace:
    return SimpleNamespace(tables={SIDRA_POPULATION_TABLE: SimpleNamespace(periods=set(periods))})


def test_all_census_periods_returns_every_census_sorted() -> None:
    meta = _meta({"2022", "2010"})
    assert _all_census_periods(meta) == ["2010", "2022"]


def test_all_census_periods_is_independent_of_any_query_window() -> None:
    # The full national+full-history census set is returned no matter what — there is no intent
    # argument, by construction. A 2021-2022 query gets 2010 AND 2022 as anchors, not just 2022.
    meta = _meta({"2000", "2010", "2022"})
    assert _all_census_periods(meta) == ["2000", "2010", "2022"]


def test_all_census_periods_ignores_non_numeric_and_missing_table() -> None:
    assert _all_census_periods(_meta({"2010", "acervo", "2022"})) == ["2010", "2022"]
    assert _all_census_periods(SimpleNamespace(tables={})) == []
