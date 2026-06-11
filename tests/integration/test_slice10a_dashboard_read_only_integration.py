from pathlib import Path

import pytest

from pegasus.dashboard.contracts import DashboardContractError, assert_no_compute_trigger
from pegasus.dashboard.read_only import inspect_run, read_table_head, variable_dictionary
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.dashboard import run_dashboard_inspect_run, run_dashboard_table_head


def test_slice10a_dashboard_inspects_valid_run_without_computation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    payload = inspect_run(run_dir=run_dir)
    assert payload["read_only"] is True
    assert payload["validation_ok"] is True
    assert not payload["first_class_keys_missing"]
    names = {row["name"] for row in payload["tables"]["tables"]}
    assert "V_fields" in names
    assert "Q_tensor" in names
    assert "VariableDictionary" in names


def test_slice10a_dashboard_table_head_is_local_and_bounded(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    head = read_table_head(run_dir=run_dir, table_name="V_fields", limit=1)
    assert head["table"] == "V_fields"
    assert head["rows_returned"] == 1
    assert "field_id" in head["columns"]
    vd = variable_dictionary(run_dir=run_dir, limit=2)
    assert vd["table"] == "VariableDictionary"
    assert vd["rows_returned"] == 1
    with pytest.raises(ValueError):
        read_table_head(run_dir=run_dir, table_name="Tables", limit=1)
    with pytest.raises(ValueError):
        read_table_head(run_dir=run_dir, table_name="V_fields", limit=501)


def test_slice10a_workflow_wrappers_remain_read_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    inspected = run_dashboard_inspect_run(run_dir=run_dir)
    assert inspected["validation_ok"] is True
    head = run_dashboard_table_head(run_dir=run_dir, table_name="Warnings", limit=5)
    assert head["table"] == "Warnings"
    with pytest.raises(DashboardContractError):
        assert_no_compute_trigger("sidra_extract")

