import pytest

from pegasus.dashboard.contracts import (
    DashboardContractError,
    assert_no_compute_trigger,
    assert_read_only_operation,
    dashboard_policy_manifest,
)


def test_dashboard_policy_is_strictly_read_only() -> None:
    manifest = dashboard_policy_manifest()
    assert manifest["read_only"] is True
    assert manifest["computation_triggers_forbidden"] is True
    assert manifest["external_fetch_forbidden"] is True
    assert manifest["mutation_forbidden"] is True
    assert "inspect_run" in manifest["allowed_operations"]
    assert "compile" in manifest["forbidden_operations"]
    assert "pirs_hsic" in manifest["forbidden_operations"]


def test_dashboard_blocks_computation_fetch_and_mutation_operations() -> None:
    for operation in ["datasus_fetch", "sidra_fetch", "she_build", "efg_build", "pirs_model", "pirs_hsic", "stdfm_solve", "population_solver", "race_bridge", "compile", "mutate_run"]:
        with pytest.raises(DashboardContractError):
            assert_no_compute_trigger(operation)


def test_dashboard_rejects_unregistered_read_operation() -> None:
    with pytest.raises(DashboardContractError):
        assert_read_only_operation("surprise_read_path")

