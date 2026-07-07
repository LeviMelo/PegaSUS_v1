"""O7 / §I.2 / §III.5 — the count-with-exposure margin routes ONLY extensive-count numerators.

The RN kernel emits a MeasuredQuantity sidecar, but the count-with-exposure (Poisson-offset)
margin is valid only when the NUMERATOR is a genuine extensive count (deaths, births). An RN
whose numerator is intensive (a rate ÷ rate, a density, a continuous index) must NOT advertise
a measured_quantity_ref — otherwise the LDO would model an intensive quantity as an extensive
Poisson count (a denominator-principle violation). This pins the emission gate.
"""

from __future__ import annotations

import polars as pl

from pegasus.core.enums import FieldState, MaterializationState
from pegasus.efg.executor.kernels import _compute_rn_ratio
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node


def _parent(kind, name, values, tmp_path):
    df = pl.DataFrame({
        "municipality_cod6": ["270430", "270630"],
        "year": [2020, 2020],
        "value": values,
    })
    p = tmp_path / f"{name}.parquet"
    df.write_parquet(p)
    return make_field_node(
        name=name, kind=kind, carrier=name, unit="counts",
        support={}, axes={"geography": "municipality", "time": "year"},
        aggregation="additive" if kind == "extensive_measure" else "weighted_mean",
        role=["fixture"], source=["fixture"],
        operator="COUNT_MEASURE" if kind == "extensive_measure" else "RN",
        provenance=["official"], state=FieldState.verified, warnings=[],
        lineage=make_lineage(parent_ids=[], operator_type="fixture", operator_params={"n": name}),
        materialization_state=MaterializationState.materialized, path=str(p),
    )


def _rn_field(num, den):
    return make_field_node(
        name="rate", kind="intensive_density", carrier="Rate", unit="per_capita",
        support={}, axes={"geography": "municipality", "time": "year"},
        aggregation="weighted_mean", role=["fixture"], source=["fixture"], operator="RN",
        provenance=["official"], state=FieldState.verified, warnings=[],
        lineage=make_lineage(parent_ids=[num.id, den.id], operator_type="RN", operator_params={}),
        materialization_state=MaterializationState.metadata_only,
    )


def test_extensive_count_numerator_advertises_exposure_ref(tmp_path):
    num = _parent("extensive_measure", "deaths", [12.0, 6.0], tmp_path)      # a genuine count
    den = _parent("extensive_measure", "population", [1200.0, 600.0], tmp_path)
    _, _, support = _compute_rn_ratio(_rn_field(num, den), {num.id: num, den.id: den}, tmp_path)
    assert support["measured_quantity_extensive"] is True
    assert "measured_quantity_ref" in support                # → count-with-exposure margin


def test_intensive_numerator_is_denied_the_exposure_ref(tmp_path):
    # numerator is itself a density/rate (intensive) — modelling it as an extensive Poisson
    # count with an offset would be a denominator-principle violation.
    num = _parent("intensive_density", "gdp_per_worker", [3.2, 4.1], tmp_path)
    den = _parent("extensive_measure", "population", [1200.0, 600.0], tmp_path)
    _, _, support = _compute_rn_ratio(_rn_field(num, den), {num.id: num, den.id: den}, tmp_path)
    assert support["measured_quantity_extensive"] is False
    assert "measured_quantity_ref" not in support            # → stays on the rank-PIT margin
    assert support["numerator_kind"] == "intensive_density"
