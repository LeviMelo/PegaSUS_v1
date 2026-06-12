from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.she.substrate import build_substrate_bundle
from pegasus.she.source_registry import resolve_source_fields


def test_slice13a_source_registry_resolves_sim_semantics() -> None:
    resolution = resolve_source_fields(
        source_system="SIM-DO",
        columns=["underlying_icd_norm", "age_years", "raw_json"],
    )
    specs = {spec.column: spec for spec in resolution.specs}
    assert specs["underlying_icd_norm"].unit == "ICD10"
    assert "diagnostic_topology" in specs["underlying_icd_norm"].role
    assert specs["age_years"].carrier == "deaths"
    assert specs["raw_json"].admissible_by_registry is False


def test_slice13a_substrate_bundle_admits_only_registry_and_variance_valid_fields(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({
        "event_id": ["a", "b", "c"],
        "year": [2020, 2020, 2020],
        "age_years": [10.0, 30.0, 50.0],
        "underlying_icd_norm": ["I10", "J18", "I10"],
        "constant_marker": [1, 1, 1],
        "all_missing_marker": [None, None, None],
    }).write_parquet(path)

    bundle = build_substrate_bundle(artifacts=[{
        "path": str(path),
        "source_system": "SIM-DO",
        "artifact_role": "processed_events",
        "provenance_mode": "fixture",
        "source_manifest_hash": "fixture_manifest_hash",
    }])

    candidate_columns = {c.column for c in bundle.candidates}
    exclusion_reasons = {e.column: e.reason for e in bundle.exclusions}

    assert "age_years" in candidate_columns
    assert "underlying_icd_norm" in candidate_columns
    assert exclusion_reasons["event_id"] == "structural_or_audit_only"
    assert exclusion_reasons["year"] == "zero_variance_constant"
    assert exclusion_reasons["constant_marker"] == "zero_variance_constant"
    assert exclusion_reasons["all_missing_marker"] == "all_missing"
    assert bundle.zero_variance_exclusion_count >= 2
    assert bundle.all_missing_exclusion_count == 1
