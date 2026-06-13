
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pegasus.cli import app
import pegasus.workflows.efg_apply as apply_workflow
import pegasus.workflows.efg_promotion as promotion_workflow


runner = CliRunner()


def test_slice15d_plan_promotion_cli_delegates_to_workflow(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    materialization = tmp_path / "efg_materialization.json"
    materialization.write_text(json.dumps({"summary": {"field_count": 1}}), encoding="utf-8")

    def fake_plan(*, run_dir, materialization_manifest, output=None):
        assert Path(run_dir).name == "run"
        assert Path(materialization_manifest) == materialization
        assert output is None
        return {"summary": {"status": "evaluated", "planned_promotion_count": 1}}

    monkeypatch.setattr(promotion_workflow, "run_plan_efg_promotion", fake_plan)
    result = runner.invoke(
        app,
        [
            "efg",
            "plan-promotion",
            "--run-dir",
            str(run_dir),
            "--materialization-manifest",
            str(materialization),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["summary"]["planned_promotion_count"] == 1


def test_slice15d_attach_promotion_plan_cli_delegates_to_workflow(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"

    def fake_attach(*, run_dir, materialization_manifest=None):
        assert Path(run_dir).name == "run"
        assert materialization_manifest is None
        return {"status": "evaluated", "planned_promotion_count": 2}

    monkeypatch.setattr(promotion_workflow, "run_attach_efg_promotion_plan_to_run", fake_attach)
    result = runner.invoke(app, ["efg", "attach-promotion-plan", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["planned_promotion_count"] == 2


def test_slice15d_apply_promotion_plan_cli_delegates_to_workflow(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    plan = tmp_path / "efg_promotion_plan.json"
    plan.write_text(json.dumps({"summary": {"planned_promotion_count": 1}}), encoding="utf-8")

    def fake_apply(*, run_dir, promotion_plan=None, validate=True):
        assert Path(run_dir).name == "run"
        assert Path(promotion_plan) == plan
        assert validate is False
        return {"gate": "efg_promotion_apply", "promoted_field_count": 1}

    monkeypatch.setattr(apply_workflow, "run_apply_efg_promotion_plan", fake_apply)
    result = runner.invoke(
        app,
        [
            "efg",
            "apply-promotion-plan",
            "--run-dir",
            str(run_dir),
            "--promotion-plan",
            str(plan),
            "--no-validate",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["promoted_field_count"] == 1


def test_slice15d_inspect_promotion_plan_cli_is_read_only(tmp_path: Path) -> None:
    plan = tmp_path / "efg_promotion_plan.json"
    plan.write_text(
        json.dumps({"summary": {"status": "evaluated", "planned_promotion_count": 3}}),
        encoding="utf-8",
    )
    before = plan.read_text(encoding="utf-8")
    result = runner.invoke(app, ["efg", "inspect-promotion-plan", "--plan", str(plan)])
    after = plan.read_text(encoding="utf-8")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["planned_promotion_count"] == 3
    assert after == before
