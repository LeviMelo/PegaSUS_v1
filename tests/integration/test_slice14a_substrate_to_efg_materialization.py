
from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.efg.materialize import materialize_substrate_bundle, materialized_field_nodes
from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle


def test_slice14a_materializes_real_substrate_bundle_without_promoting_zero_variance(tmp_path: Path) -> None:
    artifact = tmp_path / "sim.parquet"
    pl.DataFrame(
        {
            "year": [2020, 2021, 2021],
            "age_years": [50, 51, 52],
            "race_color_admin": ["1", "4", ""],
            "underlying_icd_norm": ["I10", "J18", "R99"],
            "constant_col": [1, 1, 1],
            "all_missing_col": [None, None, None],
        }
    ).write_parquet(artifact)

    bundle = build_substrate_bundle(
        artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
    )
    assert bundle.admissible_candidate_count > 0
    assert bundle.excluded_field_count > 0

    result = materialize_substrate_bundle(bundle)
    nodes = materialized_field_nodes(bundle)
    candidate_columns = {candidate.column for candidate in bundle.candidates}
    excluded_columns = {exclusion.column for exclusion in bundle.exclusions}
    materialized_columns = {field.support["column"] for field in nodes}

    assert result.field_count == len(bundle.candidates)
    assert result.excluded_field_count == len(bundle.exclusions)
    assert materialized_columns == candidate_columns
    assert materialized_columns.isdisjoint(excluded_columns)
    assert "constant_col" in excluded_columns
    assert "all_missing_col" in excluded_columns
    assert all(field.materialization_state == "metadata_only" for field in nodes)
    assert any(field.unit == "ICD10" and field.kind == "observer_proxy" for field in nodes)
