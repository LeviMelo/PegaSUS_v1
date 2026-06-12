from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.she.zero_variance import profile_table_variance


def test_slice13a_zero_variance_gate_excludes_all_missing_and_constant(tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    pl.DataFrame({
        "event_id": ["a", "b", "c"],
        "all_missing": [None, None, None],
        "constant_cost": [0, 0, 0],
        "variable_cost": [1, 2, 3],
        "status_state": ["valid", "valid", "invalid"],
    }).write_parquet(path)

    profile = profile_table_variance(path)
    reasons = {p.column: p.exclusion_reason for p in profile.profiles}

    assert reasons["all_missing"] == "all_missing"
    assert reasons["constant_cost"] == "zero_variance_constant"
    assert reasons["event_id"] == "structural_or_audit_only"
    assert reasons["status_state"] == "structural_or_audit_only"
    assert "variable_cost" in profile.admissible_columns
    assert "constant_cost" not in profile.admissible_columns
