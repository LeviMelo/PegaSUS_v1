from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest


def test_slice13b_substrate_uses_registry_semantics_for_candidates_and_exclusions(tmp_path: Path) -> None:
    artifact = tmp_path / "sim.parquet"
    pl.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "year": [2020, 2020, 2021],
            "race_color_admin": ["1", "4", ""],
            "underlying_icd_norm": ["I10", "J18", "R99"],
            "underlying_icd_parse_state": ["valid", "valid", "ill-defined"],
            "constant_col": [1, 1, 1],
            "all_missing_col": [None, None, None],
        }
    ).write_parquet(artifact)

    bundle = build_substrate_bundle(
        artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
    )
    candidates = {candidate.column_name: candidate for candidate in bundle.candidate_fields}
    exclusions = {exclusion.column_name: exclusion for exclusion in bundle.excluded_fields}

    assert candidates["year"].carrier == "Deaths"
    assert candidates["race_color_admin"].axes["race_axis_type"] == "administrative_death_declaration"
    assert candidates["underlying_icd_norm"].quality_role == "diagnostic_code"
    assert exclusions["event_id"].reason in {"registry_not_admissible", "structural_or_audit_only"}
    assert exclusions["underlying_icd_parse_state"].reason in {"registry_not_admissible", "structural_or_audit_only"}
    assert exclusions["constant_col"].reason == "zero_variance_constant"
    assert exclusions["all_missing_col"].reason == "all_missing"

    manifest_path = write_substrate_bundle_manifest(bundle, tmp_path / "substrate.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] >= 3
    assert payload["excluded_count"] >= 4
    assert payload["registry_backed"] is True
    assert payload["source_reality_mode"] == "fixture_only"
