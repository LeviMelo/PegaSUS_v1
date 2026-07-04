"""EFG-OUT-01 — MeasuredQuantity: the EFG emits count+exposure, not a rate (MSD-III §II.3)."""

from __future__ import annotations

import polars as pl

from pegasus.efg.measured_quantity import (
    measured_quantity_from_rn_join,
    write_measured_quantity,
)


def test_measured_quantity_is_count_plus_exposure() -> None:
    joined = pl.DataFrame({
        "municipality_cod6": ["270430", "270630"],
        "icd_chapter": ["IX", "IX"],                 # a σ_C stratifier survives into the structure
        "value_numerator": [12.0, 6.0],
        "value_denominator": [1200.0, 600.0],
    })
    mq = measured_quantity_from_rn_join(
        joined, keys=["municipality_cod6"], strata=["icd_chapter"],
        field_id="Deaths_over_Population", provenance=["official"], denom_fragility=0.1,
    )
    # the terminal object carries the COUNT and the EXPOSURE — never a pre-divided rate
    assert mq.numerator_count.to_list() == [12.0, 6.0]
    assert mq.exposure.to_list() == [1200.0, 600.0]
    assert mq.offset_semantics == "log_exposure"
    # the rate is only the DERIVED view (count / exposure)
    assert mq.implied_rate().to_list() == [0.01, 0.01]
    # structure, provenance and uncertainty are preserved for the model
    assert mq.structure["strata"] == ["icd_chapter"]
    assert mq.provenance["operator"] == "RN"
    assert mq.uncertainty["denom_fragility"] == 0.1


def test_measured_quantity_roundtrips_to_sidecar(tmp_path) -> None:
    joined = pl.DataFrame({
        "municipality_cod6": ["270430"], "value_numerator": [5.0], "value_denominator": [500.0],
    })
    mq = measured_quantity_from_rn_join(
        joined, keys=["municipality_cod6"], strata=[], field_id="f", provenance=[],
    )
    back = pl.read_parquet(write_measured_quantity(mq, tmp_path / "f.measured_quantity.parquet"))
    assert set(back.columns) == {"municipality_cod6", "numerator_count", "exposure"}
    assert back["numerator_count"].to_list() == [5.0] and back["exposure"].to_list() == [500.0]
