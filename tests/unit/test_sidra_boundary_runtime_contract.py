from __future__ import annotations

from pathlib import Path


def test_generic_sidra_anchor_has_no_al_specific_policy() -> None:
    source = Path("src/pegasus/she/population/sidra_anchor.py").read_text(encoding="utf-8")
    assert "_aggregate_total_9606_candidates" not in source
    assert "AL N6" not in source
    assert "cod7 prefix 27" not in source
    assert "Expected exactly one total SIDRA 9606 population anchor fact" in source


def test_al_specific_sidra_aggregation_lives_in_audit_adapter() -> None:
    source = Path("scripts/dev/audits/actual_state_panel_runtime.py").read_text(encoding="utf-8")
    assert "SIDRA AL N6 filter expected 102 municipalities" in source
    assert "population_2022_al_n6_municipal_sidecar.parquet" in source
    assert "\"locality_level\": \"N3\"" in source
    assert "\"derived_from\":" in source
