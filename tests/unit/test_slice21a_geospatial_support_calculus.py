from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pegasus.geo.adjacency import load_adjacency
from pegasus.geo.amc import contract_to_amc
from pegasus.geo.geneallocation import geneallocate_measure
from pegasus.geo.geodata import GeoArtifactError
from pegasus.geo.spatial_index import build_spatial_index


def test_slice21a_amc_contracts_additive_mass_and_rejects_rates(tmp_path: Path) -> None:
    crosswalk = tmp_path / "amc.csv"
    pl.DataFrame({
        "municipality_id": ["a", "b"], "year": [2020, 2020], "amc_id": ["x", "x"],
    }).write_csv(crosswalk)
    frame = pl.DataFrame({"municipality_id": ["a", "b"], "year": [2020, 2020], "value": [2.0, 3.0]})
    result = contract_to_amc(frame, crosswalk_path=crosswalk, value_column="value")
    assert result.conserved is True
    assert result.frame["value"].to_list() == [5.0]
    with pytest.raises(GeoArtifactError, match="rates/intensive"):
        contract_to_amc(frame, crosswalk_path=crosswalk, value_column="value", aggregation="weighted_mean")


def test_slice21a_geneallocation_requires_normalized_weights_and_conserves_mass(tmp_path: Path) -> None:
    weights = tmp_path / "weights.csv"
    pl.DataFrame({
        "source_id": ["old", "old"], "year": [2020, 2020],
        "target_id": ["new-a", "new-b"], "weight": [0.25, 0.75],
    }).write_csv(weights)
    frame = pl.DataFrame({"source_id": ["old"], "year": [2020], "value": [100.0]})
    result = geneallocate_measure(frame, weights_path=weights, value_column="value")
    assert result.conserved is True
    assert result.frame["value"].to_list() == [25.0, 75.0]
    with pytest.raises(GeoArtifactError, match="forbidden"):
        geneallocate_measure(frame, weights_path=weights, value_column="value", aggregation="weighted_mean")


def test_slice21a_adjacency_and_spatial_index_are_explicit_and_deterministic(tmp_path: Path) -> None:
    adjacency = tmp_path / "adjacency.csv"
    pl.DataFrame({"left_id": ["a", "b"], "right_id": ["b", "a"]}).write_csv(adjacency)
    assert load_adjacency(adjacency) == {"a": ("b",), "b": ("a",)}
    index = build_spatial_index(["b", "a", "b"])
    assert index.ids == ("a", "b")
    assert index.index_by_id == {"a": 0, "b": 1}


def test_slice21a_missing_external_artifact_fails_explicitly(tmp_path: Path) -> None:
    frame = pl.DataFrame({"municipality_id": ["a"], "year": [2020], "value": [1.0]})
    with pytest.raises(GeoArtifactError, match="missing"):
        contract_to_amc(frame, crosswalk_path=tmp_path / "missing.csv", value_column="value")
