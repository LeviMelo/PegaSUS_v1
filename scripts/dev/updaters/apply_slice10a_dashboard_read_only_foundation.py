from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip() + "\n", encoding="utf-8", newline="\n")


def patch_cli() -> None:
    path = ROOT / "src/pegasus/cli.py"
    text = path.read_text(encoding="utf-8")
    if "def dashboard_inspect_run" in text:
        return
    block = r'''


# -----------------------------------------------------------------------------
# Slice 10A: read-only dashboard inspection commands
# -----------------------------------------------------------------------------
dashboard_app = typer.Typer(help="Read-only inspection of completed PegaSUS run bundles.")
app.add_typer(dashboard_app, name="dashboard")


@dashboard_app.command("assert-read-only")
def dashboard_assert_read_only() -> None:
    from pegasus.dashboard.contracts import dashboard_policy_manifest

    typer.echo(json.dumps(dashboard_policy_manifest(), indent=2, sort_keys=True))


@dashboard_app.command("inspect-run")
def dashboard_inspect_run(
    run: Path = typer.Option(..., "--run"),
) -> None:
    from pegasus.workflows.report.dashboard import run_dashboard_inspect_run

    typer.echo(json.dumps(run_dashboard_inspect_run(run_dir=run), indent=2, sort_keys=True))


@dashboard_app.command("table-head")
def dashboard_table_head(
    run: Path = typer.Option(..., "--run"),
    table: str = typer.Option(..., "--table"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    from pegasus.workflows.report.dashboard import run_dashboard_table_head

    typer.echo(json.dumps(run_dashboard_table_head(run_dir=run, table_name=table, limit=limit), indent=2, sort_keys=True))
'''
    marker = '\nif __name__ == "__main__":'
    if marker in text:
        text = text.replace(marker, block + marker)
    else:
        text = text.rstrip() + block + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/dashboard/contracts.py", r'''
        """Read-only dashboard contract for completed PegaSUS run bundles.

        The dashboard layer is allowed to inspect already-materialized output bundles.
        It is forbidden to fetch sources, invoke SHE/EFG/PIRS/ST-DFM/HSIC/population
        solvers, mutate run artifacts, or create new analytic fields.
        """

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any


        class DashboardContractError(PermissionError):
            """Raised when a dashboard request attempts a forbidden computation."""


        READ_ONLY_OPERATIONS: frozenset[str] = frozenset({
            "inspect_run",
            "list_tables",
            "table_head",
            "variable_dictionary",
            "hypotheses",
            "warnings",
            "q_tensor",
            "bundle_summary",
            "audit_policy",
        })

        FORBIDDEN_OPERATIONS: frozenset[str] = frozenset({
            "datasus_fetch",
            "datasus_ingest",
            "sidra_fetch",
            "sidra_extract",
            "she_build",
            "efg_build",
            "pirs_model",
            "pirs_hsic",
            "stdfm_solve",
            "population_solver",
            "race_bridge",
            "bridge_r",
            "compile",
            "write_bundle",
            "mutate_run",
        })


        @dataclass(frozen=True)
        class DashboardPolicy:
            read_only: bool
            allowed_operations: tuple[str, ...]
            forbidden_operations: tuple[str, ...]
            computation_triggers_forbidden: bool
            external_fetch_forbidden: bool
            mutation_forbidden: bool

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "read_only": self.read_only,
                    "allowed_operations": list(self.allowed_operations),
                    "forbidden_operations": list(self.forbidden_operations),
                    "computation_triggers_forbidden": self.computation_triggers_forbidden,
                    "external_fetch_forbidden": self.external_fetch_forbidden,
                    "mutation_forbidden": self.mutation_forbidden,
                }


        def dashboard_policy() -> DashboardPolicy:
            return DashboardPolicy(
                read_only=True,
                allowed_operations=tuple(sorted(READ_ONLY_OPERATIONS)),
                forbidden_operations=tuple(sorted(FORBIDDEN_OPERATIONS)),
                computation_triggers_forbidden=True,
                external_fetch_forbidden=True,
                mutation_forbidden=True,
            )


        def dashboard_policy_manifest() -> dict[str, Any]:
            return dashboard_policy().as_manifest()


        def assert_read_only_operation(operation: str) -> None:
            if operation in FORBIDDEN_OPERATIONS:
                raise DashboardContractError(
                    f"Dashboard operation {operation!r} is forbidden because it would trigger computation, fetch, or mutation."
                )
            if operation not in READ_ONLY_OPERATIONS:
                raise DashboardContractError(
                    f"Dashboard operation {operation!r} is not in the explicit read-only allowlist."
                )


        def assert_no_compute_trigger(operation: str) -> None:
            assert_read_only_operation(operation)
    ''')

    write("src/pegasus/dashboard/read_only.py", r'''
        """Read-only run-bundle inspection services for the PegaSUS dashboard.

        These functions only read local files from completed run directories. They do
        not fetch DATASUS/SIDRA, do not call compiler workflows, and do not mutate runs.
        """

        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Any

        import pyarrow.parquet as pq

        from pegasus.dashboard.contracts import assert_no_compute_trigger
        from pegasus.output.validate import validate_output_bundle


        FIRST_CLASS_KEYS: tuple[str, ...] = (
            "V_fields",
            "E_DAG",
            "Q_tensor",
            "P_vector",
            "UserIntent",
            "Warnings",
            "ModelAssociations",
            "ResidualAssociations",
            "Hypotheses",
            "Tables",
            "Maps",
            "VariableDictionary",
            "FailedBranches",
            "QuarantinedFields",
            "ForcedFields",
            "RunConfig",
            "ReproducibilityManifest",
        )

        TABLE_FILES: dict[str, str] = {
            "V_fields": "V_fields.parquet",
            "E_DAG": "E_DAG.parquet",
            "Q_tensor": "Q_tensor.parquet",
            "Warnings": "Warnings.parquet",
            "ModelAssociations": "ModelAssociations.parquet",
            "ResidualAssociations": "ResidualAssociations.parquet",
            "Hypotheses": "Hypotheses.parquet",
            "VariableDictionary": "VariableDictionary.parquet",
            "FailedBranches": "FailedBranches.parquet",
            "QuarantinedFields": "QuarantinedFields.parquet",
            "ForcedFields": "ForcedFields.parquet",
        }


        def _json_file(path: Path) -> dict[str, Any]:
            return json.loads(path.read_text(encoding="utf-8"))


        def _run_dir(path: str | Path) -> Path:
            run_dir = Path(path)
            if not run_dir.exists() or not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
            return run_dir


        def _table_path(run_dir: Path, table_name: str) -> Path:
            if table_name not in TABLE_FILES:
                raise ValueError(f"Unsupported dashboard table: {table_name!r}")
            path = run_dir / TABLE_FILES[table_name]
            if not path.exists():
                raise FileNotFoundError(f"Dashboard table missing: {path}")
            return path


        def parquet_row_count(path: Path) -> int:
            return int(pq.ParquetFile(path).metadata.num_rows)


        def list_tables(*, run_dir: str | Path) -> dict[str, Any]:
            assert_no_compute_trigger("list_tables")
            root = _run_dir(run_dir)
            tables = []
            for key, rel in TABLE_FILES.items():
                path = root / rel
                tables.append({
                    "name": key,
                    "path": rel,
                    "exists": path.exists(),
                    "rows": parquet_row_count(path) if path.exists() else None,
                })
            extra_tables = []
            tables_dir = root / "Tables"
            if tables_dir.exists():
                for path in sorted(tables_dir.glob("*.parquet")):
                    extra_tables.append({
                        "name": path.stem,
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "rows": parquet_row_count(path),
                    })
            return {"tables": tables, "artifact_tables": extra_tables}


        def read_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:
            assert_no_compute_trigger("table_head")
            if limit < 0 or limit > 500:
                raise ValueError("Dashboard table head limit must be between 0 and 500.")
            root = _run_dir(run_dir)
            path = _table_path(root, table_name)
            table = pq.read_table(path)
            rows = table.to_pylist()[:limit]
            return {
                "table": table_name,
                "path": TABLE_FILES[table_name],
                "rows_returned": len(rows),
                "row_count": parquet_row_count(path),
                "columns": list(table.column_names),
                "rows": rows,
            }


        def inspect_run(*, run_dir: str | Path, validate: bool = True) -> dict[str, Any]:
            assert_no_compute_trigger("inspect_run")
            root = _run_dir(run_dir)
            present = []
            missing = []
            for key in FIRST_CLASS_KEYS:
                if key == "Tables":
                    exists = (root / "Tables").exists()
                elif key == "Maps":
                    exists = (root / "Maps").exists()
                elif key in {"RunConfig", "UserIntent", "P_vector", "ReproducibilityManifest"}:
                    filename = "ReproducibilityManifest.json" if key == "ReproducibilityManifest" else f"{key}.json"
                    exists = (root / filename).exists()
                else:
                    exists = (root / f"{key}.parquet").exists()
                (present if exists else missing).append(key)
            validation = validate_output_bundle(run_dir=str(root)) if validate else None
            run_config = _json_file(root / "RunConfig.json") if (root / "RunConfig.json").exists() else {}
            manifest = _json_file(root / "ReproducibilityManifest.json") if (root / "ReproducibilityManifest.json").exists() else {}
            table_summary = list_tables(run_dir=root)
            return {
                "run_dir": str(root),
                "read_only": True,
                "first_class_keys_present": present,
                "first_class_keys_missing": missing,
                "validation_ok": validation.ok if validation is not None else None,
                "validation_errors": validation.errors if validation is not None else [],
                "validation_warnings": validation.warnings if validation is not None else [],
                "run_config_slice": run_config.get("slice"),
                "workflow_mode": run_config.get("workflow_mode") or manifest.get("workflow_mode"),
                "telemetry_stage_status": (manifest.get("telemetry") or {}).get("stage_status", {}),
                "tables": table_summary,
            }


        def variable_dictionary(*, run_dir: str | Path, limit: int = 100) -> dict[str, Any]:
            assert_no_compute_trigger("variable_dictionary")
            return read_table_head(run_dir=run_dir, table_name="VariableDictionary", limit=limit)


        def hypotheses(*, run_dir: str | Path, limit: int = 100) -> dict[str, Any]:
            assert_no_compute_trigger("hypotheses")
            return read_table_head(run_dir=run_dir, table_name="Hypotheses", limit=limit)


        def warnings_summary(*, run_dir: str | Path, limit: int = 100) -> dict[str, Any]:
            assert_no_compute_trigger("warnings")
            return read_table_head(run_dir=run_dir, table_name="Warnings", limit=limit)
    ''')

    write("src/pegasus/workflows/dashboard.py", r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.dashboard.read_only import inspect_run, read_table_head


        def run_dashboard_inspect_run(*, run_dir: str | Path) -> dict[str, Any]:
            return inspect_run(run_dir=run_dir, validate=True)


        def run_dashboard_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:
            return read_table_head(run_dir=run_dir, table_name=table_name, limit=limit)
    ''')

    write("tests/unit/test_dashboard_read_only_contract.py", r'''
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
    ''')

    write("tests/integration/test_slice10a_dashboard_read_only_integration.py", r'''
        from pathlib import Path

        import pytest

        from pegasus.dashboard.contracts import DashboardContractError, assert_no_compute_trigger
        from pegasus.dashboard.read_only import inspect_run, read_table_head, variable_dictionary
        from pegasus.output.bundle import create_empty_output_bundle
        from pegasus.workflows.report.dashboard import run_dashboard_inspect_run, run_dashboard_table_head


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
    ''')

    write("scripts/dev/audits/audit_slice10a_dashboard_read_only.py", r'''
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        from pegasus.dashboard.contracts import DashboardContractError, assert_no_compute_trigger, dashboard_policy_manifest
        from pegasus.dashboard.read_only import inspect_run, read_table_head


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--run", required=True)
            args = parser.parse_args()
            run_dir = Path(args.run)
            policy = dashboard_policy_manifest()
            if not policy.get("read_only"):
                raise SystemExit("dashboard policy is not read-only")
            for forbidden in ["datasus_fetch", "sidra_fetch", "she_build", "efg_build", "pirs_model", "pirs_hsic", "stdfm_solve", "population_solver", "race_bridge", "compile", "mutate_run"]:
                try:
                    assert_no_compute_trigger(forbidden)
                except DashboardContractError:
                    pass
                else:
                    raise SystemExit(f"dashboard allowed forbidden operation: {forbidden}")
            summary = inspect_run(run_dir=run_dir, validate=True)
            if not summary["validation_ok"]:
                raise SystemExit("dashboard inspected run does not validate: " + json.dumps(summary["validation_errors"]))
            if summary["first_class_keys_missing"]:
                raise SystemExit("dashboard did not see all 17 keys")
            v_head = read_table_head(run_dir=run_dir, table_name="V_fields", limit=1)
            q_head = read_table_head(run_dir=run_dir, table_name="Q_tensor", limit=1)
            if "field_id" not in v_head["columns"] or "field_id" not in q_head["columns"]:
                raise SystemExit("dashboard table heads do not expose schema columns")
            print("AUDIT PASSED: Slice 10A read-only dashboard contract, local run inspection, table-head view, and computation-trigger forbiddance validated.")


        if __name__ == "__main__":
            main()
    ''')

    patch_cli()
    print("Slice 10A updater applied: read-only dashboard contract, local run inspection services, CLI commands, tests, and audit added.")


if __name__ == "__main__":
    main()
