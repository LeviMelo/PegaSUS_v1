
from __future__ import annotations

from pegasus.acceptance.contracts import (
    CANONICAL_ACCEPTANCE_SURFACES,
    FORBIDDEN_DASHBOARD_COMPUTE_STAGES,
    acceptance_plan,
    required_first_class_key_paths,
)
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES


def test_slice11a_acceptance_plan_names_all_output_keys_and_surfaces() -> None:
    plan = acceptance_plan()
    assert plan["slice"] == "11A"
    assert set(plan["required_output_keys"]) == set(OUTPUT_BUNDLE_FILES)
    assert set(plan["required_output_paths"]) == set(required_first_class_key_paths())
    assert "compile_smoke" in CANONICAL_ACCEPTANCE_SURFACES
    assert "dashboard_read_only" in CANONICAL_ACCEPTANCE_SURFACES


def test_slice11a_forbidden_dashboard_compute_stages_cover_pipeline() -> None:
    forbidden = set(FORBIDDEN_DASHBOARD_COMPUTE_STAGES)
    for stage in [
        "datasus_acquire",
        "sidra_fetch",
        "she_build",
        "efg_build",
        "population_solver",
        "race_bridge",
        "stdfm",
        "pirs_model",
        "pirs_hsic",
    ]:
        assert stage in forbidden
