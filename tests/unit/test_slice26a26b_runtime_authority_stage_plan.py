from __future__ import annotations

from pegasus.workflows.compile import _compiler_architecture_metadata
from pegasus.workflows.stage_plan import build_compile_stage_plan, validate_compiler_stage_plan


def test_slice26a_architecture_quarantines_legacy_runtime_authority() -> None:
    metadata = _compiler_architecture_metadata()
    assert metadata["graph_authority"] == "autonomous_efg_core"
    assert metadata["numerical_materialization"] == "autonomous_compiler_services"
    assert metadata["legacy_bootstrap_status"] == "quarantined_fixture_only"
    assert metadata["legacy_graph_authority"] is False
    assert metadata["numerical_materialization"] != "legacy_bootstrap"
    assert metadata["legacy_bootstrap_builder"] is None
    assert metadata["legacy_bootstrap_status"] == "quarantined_fixture_only"
    assert "compatibility_materializer" not in str(metadata)
    assert "pegasus.workflows.efg.run_build_sim_fixture" not in str(metadata)


def test_slice26b_unrequested_optional_stages_have_skip_proofs() -> None:
    plan = build_compile_stage_plan(
        intent={"budget": "fast", "geo_mode": "native", "population_mode": "official_sidra_anchor", "context_policy": []},
        population_tensor_mode=None,
    )
    manifest = plan.as_manifest()
    for stage_id in ("population_solver", "stdfm", "pirs_model", "pirs_hsic"):
        spec = manifest["by_stage"][stage_id]
        assert spec["requested"] is False
        assert spec["skip_allowed"] is True
        assert spec["skip_reason"]
    errors = validate_compiler_stage_plan(
        stage_plan=manifest,
        stage_status={
            "geo_support": "success",
            "population_solver": "skipped",
            "stdfm": "skipped",
            "pirs_model": "skipped",
            "pirs_hsic": "skipped",
        },
    )
    assert errors == ()


def test_slice26b_requested_stage_cannot_be_silently_skipped() -> None:
    plan = build_compile_stage_plan(
        intent={
            "budget": "fast",
            "geo_mode": "native",
            "population_mode": "independent_population_tensor",
            "context_policy": ["include_population_tensor", "hsic"],
        },
        population_tensor_mode="independent_denominator",
    )
    errors = validate_compiler_stage_plan(
        stage_plan=plan.as_manifest(),
        stage_status={
            "geo_support": "success",
            "population_solver": "skipped",
            "stdfm": "skipped",
            "pirs_model": "skipped",
            "pirs_hsic": "skipped",
        },
    )
    assert "requested_stage_skipped:population_solver" in errors
    assert "requested_stage_skipped:pirs_hsic" in errors
