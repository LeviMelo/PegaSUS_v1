from __future__ import annotations

import ast
import shutil
import textwrap
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def ensure_clean_parse(path: str) -> None:
    ast.parse(read(path), filename=path)


def patch_cli() -> None:
    path = "src/pegasus/cli.py"
    text = read(path)
    if "acceptance_app = typer.Typer" not in text:
        anchor = "pirs_app = typer.Typer(no_args_is_help=True)"
        if anchor not in text:
            raise RuntimeError("cli.py: pirs_app anchor not found")
        text = text.replace(anchor, anchor + "\nacceptance_app = typer.Typer(no_args_is_help=True)", 1)
    if 'app.add_typer(acceptance_app, name="acceptance")' not in text:
        anchor = 'app.add_typer(pirs_app, name="pirs")'
        if anchor not in text:
            raise RuntimeError("cli.py: pirs add_typer anchor not found")
        text = text.replace(anchor, anchor + '\napp.add_typer(acceptance_app, name="acceptance")', 1)
    block = '''

# Slice 11A acceptance hardening commands
@acceptance_app.command("plan")
def acceptance_plan() -> None:
    from pegasus.workflows.report.acceptance import run_acceptance_plan

    typer.echo(json.dumps(run_acceptance_plan(), indent=2, sort_keys=True))


@acceptance_app.command("check-run")
def acceptance_check_run(
    run: Path = typer.Option(..., "--run"),
    require_non_scaffold: bool = typer.Option(False, "--require-non-scaffold"),
) -> None:
    from pegasus.workflows.report.acceptance import run_acceptance_check_run

    result = run_acceptance_check_run(run_dir=run, require_non_scaffold=require_non_scaffold)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("ok", False):
        raise typer.Exit(1)
'''
    if 'acceptance_check_run' not in text:
        text = text.rstrip() + block
    # json import exists after Slice 10A repair; add defensively if absent.
    if "import json" not in text:
        text = text.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nimport json\n", 1)
    write(path, text)


ACCEPTANCE_INIT = '''
"""Acceptance and milestone-contract checks for PegaSUS."""
'''

ACCEPTANCE_CONTRACTS = r'''
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.validate import validate_output_bundle

FORBIDDEN_DASHBOARD_COMPUTE_STAGES: tuple[str, ...] = (
    "datasus_acquire",
    "datasus_decode",
    "sidra_fetch",
    "sidra_extract",
    "sidra_normalize",
    "she_build",
    "efg_build",
    "population_solver",
    "race_bridge",
    "stdfm",
    "pirs_model",
    "pirs_hsic",
)

CANONICAL_ACCEPTANCE_SURFACES: tuple[str, ...] = (
    "compile_smoke",
    "race_bridge_compile",
    "cnes_sih_compile",
    "population_tensor_compile",
    "sidra_stdfm_standalone",
    "pirs_parametric_standalone",
    "hsic_standalone",
    "dashboard_read_only",
)

REQUIRED_OUTPUT_TABLES: tuple[str, ...] = (
    "V_fields",
    "E_DAG",
    "Q_tensor",
    "Warnings",
    "ModelAssociations",
    "ResidualAssociations",
    "Hypotheses",
    "VariableDictionary",
    "FailedBranches",
    "QuarantinedFields",
    "ForcedFields",
)


@dataclass(frozen=True)
class AcceptanceRunSummary:
    run_dir: str
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    first_class_keys: tuple[str, ...]
    field_count: int
    q_count: int
    warning_count: int
    failed_branch_count: int
    model_association_count: int
    residual_association_count: int
    hypothesis_count: int
    dashboard_safe_values: tuple[str, ...]
    telemetry_stage_status: dict[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "first_class_keys": list(self.first_class_keys),
            "field_count": self.field_count,
            "q_count": self.q_count,
            "warning_count": self.warning_count,
            "failed_branch_count": self.failed_branch_count,
            "model_association_count": self.model_association_count,
            "residual_association_count": self.residual_association_count,
            "hypothesis_count": self.hypothesis_count,
            "dashboard_safe_values": list(self.dashboard_safe_values),
            "telemetry_stage_status": self.telemetry_stage_status,
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _count_parquet(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pq.read_table(path).num_rows)


def _string_values(path: Path, column: str) -> tuple[str, ...]:
    if not path.exists():
        return ()
    table = pq.read_table(path, columns=[column])
    values = sorted({str(v.as_py()) for v in table[column] if v.as_py() is not None})
    return tuple(values)


def exact_first_class_keys(run_dir: str | Path) -> tuple[str, ...]:
    root = Path(run_dir)
    return tuple(sorted(p.name if p.is_dir() else p.name for p in root.iterdir()))


def required_first_class_key_paths() -> tuple[str, ...]:
    return tuple(sorted(OUTPUT_BUNDLE_FILES.values()))


def acceptance_plan() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "slice": "11A",
        "purpose": "milestone acceptance hardening and canonical output-contract drift detection",
        "surfaces": list(CANONICAL_ACCEPTANCE_SURFACES),
        "required_output_keys": list(OUTPUT_BUNDLE_FILES.keys()),
        "required_output_paths": list(required_first_class_key_paths()),
        "required_tables": list(REQUIRED_OUTPUT_TABLES),
        "forbidden_dashboard_compute_stages": list(FORBIDDEN_DASHBOARD_COMPUTE_STAGES),
    }


def summarize_run(run_dir: str | Path, *, require_non_scaffold: bool = False) -> AcceptanceRunSummary:
    root = Path(run_dir)
    validation = validate_output_bundle(run_dir=str(root))
    errors = list(validation.errors)
    manifest = _load_json(root / "ReproducibilityManifest.json")
    telemetry = manifest.get("telemetry", {}) if isinstance(manifest.get("telemetry", {}), dict) else {}
    stage_status = telemetry.get("stage_status", {}) if isinstance(telemetry.get("stage_status", {}), dict) else {}

    v_path = root / "V_fields.parquet"
    q_path = root / "Q_tensor.parquet"
    fields = _count_parquet(v_path)
    q_rows = _count_parquet(q_path)
    if fields != q_rows:
        errors.append(f"V_fields/Q_tensor row-count mismatch: {fields} != {q_rows}")
    if require_non_scaffold and fields <= 1:
        errors.append("acceptance check required a non-scaffold run but V_fields has <=1 row")
    dashboard_values = _string_values(v_path, "dashboard_safe")
    if not dashboard_values:
        errors.append("V_fields.dashboard_safe has no values")
    bad_dashboard = sorted(v for v in dashboard_values if v not in {"True", "False", "warning"})
    if bad_dashboard:
        errors.append(f"invalid dashboard_safe values: {bad_dashboard}")

    return AcceptanceRunSummary(
        run_dir=str(root),
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(validation.warnings),
        first_class_keys=exact_first_class_keys(root),
        field_count=fields,
        q_count=q_rows,
        warning_count=_count_parquet(root / "Warnings.parquet"),
        failed_branch_count=_count_parquet(root / "FailedBranches.parquet"),
        model_association_count=_count_parquet(root / "ModelAssociations.parquet"),
        residual_association_count=_count_parquet(root / "ResidualAssociations.parquet"),
        hypothesis_count=_count_parquet(root / "Hypotheses.parquet"),
        dashboard_safe_values=dashboard_values,
        telemetry_stage_status=dict(stage_status),
    )


def assert_dashboard_did_not_compute(*, before: AcceptanceRunSummary, after: AcceptanceRunSummary) -> None:
    if before.as_manifest() != after.as_manifest():
        raise AssertionError("dashboard/read-only inspection changed the run acceptance summary")
'''

ACCEPTANCE_WORKFLOW = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.acceptance.contracts import acceptance_plan, summarize_run


def run_acceptance_plan() -> dict[str, Any]:
    return acceptance_plan()


def run_acceptance_check_run(*, run_dir: str | Path, require_non_scaffold: bool = False) -> dict[str, Any]:
    return summarize_run(run_dir, require_non_scaffold=require_non_scaffold).as_manifest()
'''

UNIT_TEST = r'''
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
'''

INTEGRATION_TEST = r'''
from __future__ import annotations

from pathlib import Path

from pegasus.acceptance.contracts import assert_dashboard_did_not_compute, summarize_run
from pegasus.dashboard.read_only import inspect_run
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.report.acceptance import run_acceptance_check_run, run_acceptance_plan


def test_slice11a_acceptance_checks_valid_17_key_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    summary = summarize_run(run_dir)
    assert summary.ok, summary.errors
    assert summary.field_count == 1
    assert summary.q_count == 1
    assert "V_fields.parquet" in summary.first_class_keys
    assert "RunConfig.json" in summary.first_class_keys


def test_slice11a_acceptance_can_require_non_scaffold_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    result = run_acceptance_check_run(run_dir=run_dir, require_non_scaffold=True)
    assert result["ok"] is False
    assert any("non-scaffold" in error for error in result["errors"])


def test_slice11a_dashboard_inspection_does_not_change_acceptance_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    before = summarize_run(run_dir)
    inspected = inspect_run(run_dir)
    after = summarize_run(run_dir)
    assert inspected.validation_ok is True
    assert_dashboard_did_not_compute(before=before, after=after)


def test_slice11a_workflow_plan_exposes_acceptance_surfaces() -> None:
    plan = run_acceptance_plan()
    assert "compile_smoke" in plan["surfaces"]
    assert "dashboard_read_only" in plan["surfaces"]
'''

AUDIT = r'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pegasus.acceptance.contracts import acceptance_plan, summarize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Slice 11A milestone acceptance contract surfaces.")
    parser.add_argument("--run", action="append", default=[], help="Run directory to validate. May be repeated.")
    parser.add_argument("--require-non-scaffold", action="store_true")
    args = parser.parse_args()

    plan = acceptance_plan()
    print(json.dumps({"acceptance_plan": plan}, indent=2, sort_keys=True))
    errors: list[str] = []
    for run in args.run:
        summary = summarize_run(Path(run), require_non_scaffold=args.require_non_scaffold)
        print(json.dumps(summary.as_manifest(), indent=2, sort_keys=True))
        if not summary.ok:
            errors.extend(f"{run}: {error}" for error in summary.errors)
    if errors:
        raise SystemExit("AUDIT FAILED: " + "; ".join(errors))
    print("AUDIT PASSED: Slice 11A milestone acceptance contract and run-bundle summaries validated.")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    write("src/pegasus/acceptance/__init__.py", ACCEPTANCE_INIT)
    write("src/pegasus/acceptance/contracts.py", ACCEPTANCE_CONTRACTS)
    write("src/pegasus/workflows/acceptance.py", ACCEPTANCE_WORKFLOW)
    write("tests/unit/test_acceptance_contract.py", UNIT_TEST)
    write("tests/integration/test_slice11a_acceptance_integration.py", INTEGRATION_TEST)
    write("scripts/dev/audits/audit_slice11a_acceptance.py", AUDIT)
    patch_cli()
    for path in [
        "src/pegasus/acceptance/__init__.py",
        "src/pegasus/acceptance/contracts.py",
        "src/pegasus/workflows/acceptance.py",
        "src/pegasus/cli.py",
        "tests/unit/test_acceptance_contract.py",
        "tests/integration/test_slice11a_acceptance_integration.py",
        "scripts/dev/audits/audit_slice11a_acceptance.py",
    ]:
        ensure_clean_parse(path)
    print("Slice 11A updater applied: milestone acceptance contract, run-bundle summary checks, CLI, tests, and audit added.")


if __name__ == "__main__":
    main()
