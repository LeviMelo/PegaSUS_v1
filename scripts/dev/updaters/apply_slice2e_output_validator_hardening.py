from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path.cwd()

EXPECTED_REPLACE = [
    "src/pegasus/output/validate.py",
]

EXPECTED_CREATE = [
    "tests/integration/test_slice2e_output_validator_contract.py",
    "scripts/dev/audits/audit_slice2e_output_validator_contract.py",
]


def rel(path: str) -> Path:
    return ROOT / path


def read(path: str) -> str:
    return rel(path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = rel(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def preflight() -> None:
    required = [
        "pyproject.toml",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/schemas.py",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/cli.py",
        "config/intents/alagoas_smoke.json",
        "tests/integration/test_slice2d_compile_smoke.py",
    ]
    missing = [p for p in required if not rel(p).exists()]
    if missing:
        raise RuntimeError(f"Slice 2E preflight failed; missing expected files: {missing}")

    validate_py = read("src/pegasus/output/validate.py")
    if "def validate_output_bundle" not in validate_py:
        raise RuntimeError("Slice 2E preflight failed: validate_output_bundle not found.")
    if "OutputSchemaRegistry" not in validate_py or "OUTPUT_BUNDLE_FILES" not in validate_py:
        raise RuntimeError("Slice 2E preflight failed: validator does not import expected output schema objects.")

    reproducibility_py = read("src/pegasus/output/reproducibility.py")
    if "COMPILE_TELEMETRY_STAGES" not in reproducibility_py or "RunTelemetry" not in reproducibility_py:
        raise RuntimeError("Slice 2E preflight failed: Slice 2D telemetry objects not found.")
    if "def block(" in reproducibility_py and "self.flush()" not in reproducibility_py:
        raise RuntimeError("Slice 2E preflight failed: RunTelemetry.block() does not appear to flush after Slice 2D repair.")

    compile_py = read("src/pegasus/workflows/compile.py")
    if "def run_compile" not in compile_py:
        raise RuntimeError("Slice 2E preflight failed: run_compile not found.")


def main() -> None:
    preflight()

    print("Slice 2E updater preflight passed.")
    print("REPLACE:")
    for p in EXPECTED_REPLACE:
        print(f"  {p}")
    print("CREATE:")
    for p in EXPECTED_CREATE:
        print(f"  {p}")

    write("src/pegasus/output/validate.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    import pyarrow.parquet as pq

    from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES, TERMINAL_STAGE_STATUSES
    from pegasus.output.schemas import OUTPUT_BUNDLE_FILES, OutputSchemaRegistry, OutputValidationResult

    REQUIRED_V_FIELDS_COLUMNS = {
        "field_id",
        "name",
        "kind",
        "carrier",
        "unit",
        "aggregation",
        "role",
        "source",
        "support_json",
        "axes_json",
        "operator",
        "provenance",
        "state",
        "dashboard_safe",
        "warnings",
        "lineage_hash",
        "registry_hash",
        "materialization_state",
        "path",
    }

    REQUIRED_E_DAG_COLUMNS = {
        "edge_id",
        "parent_field_id",
        "child_field_id",
        "operator",
        "operator_params_json",
        "registry_versions_json",
        "created_at",
    }

    REQUIRED_Q_TENSOR_COLUMNS = {
        "field_id",
        "n_events",
        "n_denom",
        "n_eff",
        "cov_S",
        "cov_T",
        "missingness",
        "zero_inflation",
        "denom_fragility",
        "provenance_risk",
        "state",
        "dashboard_safe",
        "warnings",
        "computed_at",
        "q_schema_version",
    }

    REQUIRED_VARIABLE_DICTIONARY_COLUMNS = {
        "field_id",
        "display_name",
        "technical_name",
        "definition",
        "estimand_label",
        "source_systems",
        "carrier",
        "unit",
        "support_description",
        "axis_description",
        "provenance_description",
        "state",
        "dashboard_safe",
        "interpretation_warning",
    }

    FIELD_REFERENCE_COLUMNS = {
        "field_id",
        "parent_field_id",
        "child_field_id",
        "outcome_field_id",
        "covariate_field_id",
        "residual_field_id",
    }


    def _read(path: Path):
        return pq.read_table(path)


    def _column_values(table, column: str) -> list[Any]:
        if column not in table.column_names:
            return []
        return table.column(column).to_pylist()


    def _nonnull(values: list[Any]) -> set[Any]:
        return {value for value in values if value is not None and value != ""}


    def _load_json_file(path: Path, *, errors: list[str], name: str) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid {name}: {exc}")
            return None


    def _load_json_cell(value: Any, *, errors: list[str], context: str) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except Exception as exc:
            errors.append(f"invalid JSON cell at {context}: {exc}")
            return None


    def _require_columns(*, table_name: str, actual: set[str], required: set[str], errors: list[str]) -> None:
        missing = sorted(required - actual)
        if missing:
            errors.append(f"{table_name} missing required columns: {missing}")


    def _validate_first_class_keys(root: Path, schema_registry: OutputSchemaRegistry, errors: list[str]) -> None:
        expected_names = {OUTPUT_BUNDLE_FILES[key] for key in schema_registry.required_keys}
        found_names = {p.name for p in root.iterdir()}

        missing = expected_names - found_names
        extra = found_names - expected_names

        for name in sorted(missing):
            errors.append(f"missing first-class artifact: {name}")
        for name in sorted(extra):
            errors.append(f"extra first-class artifact: {name}")

        for key, name in OUTPUT_BUNDLE_FILES.items():
            if key not in schema_registry.required_keys:
                continue
            path = root / name
            if not path.exists():
                continue
            if key in {"Tables", "Maps"}:
                if not path.is_dir():
                    errors.append(f"first-class artifact is not a directory: {name}")
            elif not path.is_file():
                errors.append(f"first-class artifact is not a file: {name}")


    def _validate_manifest_and_config(
        *,
        root: Path,
        errors: list[str],
        warnings: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        user_intent = _load_json_file(root / "UserIntent.json", errors=errors, name="UserIntent.json")
        run_config = _load_json_file(root / "RunConfig.json", errors=errors, name="RunConfig.json")
        manifest = _load_json_file(root / "ReproducibilityManifest.json", errors=errors, name="ReproducibilityManifest.json")
        p_vector = _load_json_file(root / "P_vector.json", errors=errors, name="P_vector.json")

        if not isinstance(user_intent, dict):
            errors.append("UserIntent.json is not a frozen JSON object")
            user_intent = {}
        if not isinstance(run_config, dict):
            errors.append("RunConfig.json is not a frozen JSON object")
            run_config = {}
        if not isinstance(manifest, dict):
            errors.append("ReproducibilityManifest.json is not a JSON object")
            manifest = {}
        if not isinstance(p_vector, (dict, list)):
            errors.append("P_vector.json must be a JSON object or list")

        compile_mode = run_config.get("compile_mode") or manifest.get("compile_mode")
        is_compile_run = bool(compile_mode)

        source_hashes = manifest.get("source_hashes")
        registry_hashes = manifest.get("registry_hashes")

        if is_compile_run:
            if not isinstance(source_hashes, dict) or not source_hashes:
                errors.append("ReproducibilityManifest.json missing nonempty source_hashes for compile run")
            if not isinstance(registry_hashes, dict) or not registry_hashes:
                errors.append("ReproducibilityManifest.json missing nonempty registry_hashes for compile run")
            if not isinstance(run_config.get("source_hashes"), dict) or not run_config.get("source_hashes"):
                errors.append("RunConfig.json missing nonempty source_hashes for compile run")
            if not isinstance(run_config.get("registry_hashes"), dict) or not run_config.get("registry_hashes"):
                errors.append("RunConfig.json missing nonempty registry_hashes for compile run")
        else:
            if not isinstance(source_hashes, dict) or not source_hashes:
                warnings.append("ReproducibilityManifest.json has empty or missing source_hashes on non-compile run")
            if not isinstance(registry_hashes, dict) or not registry_hashes:
                warnings.append("ReproducibilityManifest.json has empty or missing registry_hashes on non-compile run")

        _validate_telemetry(manifest=manifest, is_compile_run=is_compile_run, errors=errors)
        return user_intent, run_config, manifest


    def _validate_telemetry(*, manifest: dict[str, Any], is_compile_run: bool, errors: list[str]) -> None:
        telemetry = manifest.get("telemetry")
        if not isinstance(telemetry, dict):
            errors.append("ReproducibilityManifest.json missing global telemetry object")
            return

        total_wall_seconds = telemetry.get("total_wall_seconds")
        if not isinstance(total_wall_seconds, (int, float)) or total_wall_seconds < 0:
            errors.append("telemetry.total_wall_seconds missing or negative")

        stage_status = telemetry.get("stage_status")
        stage_wall_seconds = telemetry.get("stage_wall_seconds")
        if not isinstance(stage_status, dict):
            errors.append("telemetry.stage_status missing or not a mapping")
            stage_status = {}
        if not isinstance(stage_wall_seconds, dict):
            errors.append("telemetry.stage_wall_seconds missing or not a mapping")
            stage_wall_seconds = {}

        if is_compile_run:
            missing_status = sorted(set(COMPILE_TELEMETRY_STAGES) - set(stage_status))
            missing_duration = sorted(set(COMPILE_TELEMETRY_STAGES) - set(stage_wall_seconds))
            if missing_status:
                errors.append(f"telemetry.stage_status missing compile stages: {missing_status}")
            if missing_duration:
                errors.append(f"telemetry.stage_wall_seconds missing compile stages: {missing_duration}")

        for stage, status in stage_status.items():
            if status not in TERMINAL_STAGE_STATUSES:
                errors.append(f"invalid telemetry stage status: {stage}={status}")

        for stage, duration in stage_wall_seconds.items():
            if not isinstance(duration, (int, float)) or duration < 0:
                errors.append(f"invalid telemetry stage duration: {stage}={duration}")


    def _validate_parquet_contracts(
        *,
        root: Path,
        errors: list[str],
    ) -> None:
        try:
            v = _read(root / "V_fields.parquet")
            q = _read(root / "Q_tensor.parquet")
            vd = _read(root / "VariableDictionary.parquet")
            edges = _read(root / "E_DAG.parquet")
            warnings_table = _read(root / "Warnings.parquet")
            failed_branches = _read(root / "FailedBranches.parquet")
            quarantined = _read(root / "QuarantinedFields.parquet")
            forced = _read(root / "ForcedFields.parquet")
            model_assoc = _read(root / "ModelAssociations.parquet")
            residual_assoc = _read(root / "ResidualAssociations.parquet")
            hypotheses = _read(root / "Hypotheses.parquet")
        except Exception as exc:
            errors.append(f"parquet read failure: {exc}")
            return

        _require_columns(table_name="V_fields", actual=set(v.column_names), required=REQUIRED_V_FIELDS_COLUMNS, errors=errors)
        _require_columns(table_name="E_DAG", actual=set(edges.column_names), required=REQUIRED_E_DAG_COLUMNS, errors=errors)
        _require_columns(table_name="Q_tensor", actual=set(q.column_names), required=REQUIRED_Q_TENSOR_COLUMNS, errors=errors)
        _require_columns(table_name="VariableDictionary", actual=set(vd.column_names), required=REQUIRED_VARIABLE_DICTIONARY_COLUMNS, errors=errors)

        if q.num_rows == 0:
            errors.append("Q_tensor is empty")

        v_ids = _nonnull(_column_values(v, "field_id"))
        q_ids = _nonnull(_column_values(q, "field_id"))
        vd_ids = _nonnull(_column_values(vd, "field_id"))

        if not v_ids:
            errors.append("V_fields has no field_id values")
        if not v_ids.issubset(vd_ids):
            errors.append(f"VariableDictionary does not cover all V_fields: {sorted(v_ids - vd_ids)}")
        if not v_ids.issubset(q_ids):
            errors.append(f"Q_tensor does not cover all V_fields: {sorted(v_ids - q_ids)}")

        if edges.num_rows:
            for col in ["parent_field_id", "child_field_id"]:
                bad = _nonnull(_column_values(edges, col)) - v_ids
                if bad:
                    errors.append(f"E_DAG {col} contains IDs absent from V_fields: {sorted(bad)}")

        if warnings_table.num_rows and "field_id" in warnings_table.column_names:
            bad_warnings = {
                x
                for x in warnings_table.column("field_id").to_pylist()
                if x is not None and x not in {"", "run"} and x not in v_ids
            }
            if bad_warnings:
                errors.append(f"Warnings link to invalid field IDs: {sorted(bad_warnings)}")

        if failed_branches.num_rows and "parent_field_ids" in failed_branches.column_names:
            for idx, raw in enumerate(failed_branches.column("parent_field_ids").to_pylist()):
                parents = _load_json_cell(raw, errors=errors, context=f"FailedBranches.parent_field_ids[{idx}]")
                if parents is None:
                    continue
                if not isinstance(parents, list):
                    errors.append(f"FailedBranches.parent_field_ids[{idx}] is not a JSON list")
                    continue
                bad = {x for x in parents if x not in v_ids}
                if bad:
                    errors.append(f"FailedBranches parent IDs absent from V_fields at row {idx}: {sorted(bad)}")

        illegal_ids = set()
        if "state" in v.column_names and "field_id" in v.column_names:
            field_ids = v.column("field_id").to_pylist()
            states = v.column("state").to_pylist()
            illegal_ids = {fid for fid, state in zip(field_ids, states, strict=False) if state == "illegal_excluded"}

        for table_name, table in [
            ("ModelAssociations", model_assoc),
            ("ResidualAssociations", residual_assoc),
            ("Hypotheses", hypotheses),
        ]:
            if not illegal_ids:
                break
            for col in FIELD_REFERENCE_COLUMNS & set(table.column_names):
                bad = _nonnull(_column_values(table, col)) & illegal_ids
                if bad:
                    errors.append(f"{table_name}.{col} references illegal_excluded fields: {sorted(bad)}")

        for table_name, table in [("QuarantinedFields", quarantined), ("ForcedFields", forced)]:
            for col in FIELD_REFERENCE_COLUMNS & set(table.column_names):
                bad = _nonnull(_column_values(table, col)) - v_ids
                if bad:
                    errors.append(f"{table_name}.{col} contains IDs absent from V_fields: {sorted(bad)}")


    def validate_output_bundle(
        *,
        run_dir: str,
        schema_registry: OutputSchemaRegistry | None = None,
    ) -> OutputValidationResult:
        """Validate exact 17-key output bundle and mandatory cross-references."""
        schema_registry = schema_registry or OutputSchemaRegistry()
        root = Path(run_dir)
        errors: list[str] = []
        warnings: list[str] = []

        if not root.exists():
            return OutputValidationResult(ok=False, errors=[f"run_dir does not exist: {root}"], warnings=[])
        if not root.is_dir():
            return OutputValidationResult(ok=False, errors=[f"run_dir is not a directory: {root}"], warnings=[])

        _validate_first_class_keys(root, schema_registry, errors)
        if errors:
            return OutputValidationResult(ok=False, errors=errors, warnings=warnings)

        _validate_manifest_and_config(root=root, errors=errors, warnings=warnings)
        _validate_parquet_contracts(root=root, errors=errors)

        return OutputValidationResult(ok=not errors, errors=errors, warnings=warnings)
    ''')

    write("tests/integration/test_slice2e_output_validator_contract.py", r'''
    import json
    import shutil
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.compile import run_compile


    def _compile_run(tmp_path: Path) -> Path:
        run_dir = tmp_path / "compile_run"
        result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=run_dir,
            data_root=tmp_path / "data",
        )
        assert result["status"] == "success"
        assert validate_output_bundle(run_dir=str(run_dir)).ok
        return run_dir


    def _copy_run(src: Path, dst: Path) -> Path:
        shutil.copytree(src, dst)
        return dst


    def test_slice2e_validator_accepts_compile_smoke_bundle(tmp_path: Path):
        run = _compile_run(tmp_path)
        result = validate_output_bundle(run_dir=str(run))
        assert result.ok, result.errors


    def test_slice2e_validator_rejects_extra_first_class_artifact(tmp_path: Path):
        run = _copy_run(_compile_run(tmp_path), tmp_path / "with_extra")
        (run / "leak.txt").write_text("not a first-class bundle key", encoding="utf-8")

        result = validate_output_bundle(run_dir=str(run))

        assert not result.ok
        assert any("extra first-class artifact" in error and "leak.txt" in error for error in result.errors)


    def test_slice2e_validator_rejects_missing_q_tensor_coverage(tmp_path: Path):
        run = _copy_run(_compile_run(tmp_path), tmp_path / "missing_q")
        q_path = run / "Q_tensor.parquet"
        q = pl.read_parquet(q_path)
        v = pl.read_parquet(run / "V_fields.parquet")
        removed_field = v["field_id"].to_list()[0]
        q = q.filter(pl.col("field_id") != removed_field)
        q.write_parquet(q_path)

        result = validate_output_bundle(run_dir=str(run))

        assert not result.ok
        assert any("Q_tensor does not cover all V_fields" in error for error in result.errors)


    def test_slice2e_validator_rejects_invalid_compile_telemetry(tmp_path: Path):
        run = _copy_run(_compile_run(tmp_path), tmp_path / "bad_telemetry")
        manifest_path = run / "ReproducibilityManifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["telemetry"]["stage_status"]["pirs_model"] = "not_terminal"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        result = validate_output_bundle(run_dir=str(run))

        assert not result.ok
        assert any("invalid telemetry stage status" in error for error in result.errors)


    def test_slice2e_validator_rejects_missing_compile_source_hashes(tmp_path: Path):
        run = _copy_run(_compile_run(tmp_path), tmp_path / "missing_hashes")
        manifest_path = run / "ReproducibilityManifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_hashes"] = {}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        result = validate_output_bundle(run_dir=str(run))

        assert not result.ok
        assert any("source_hashes" in error for error in result.errors)
    ''')

    write("scripts/dev/audits/audit_slice2e_output_validator_contract.py", r'''
    from __future__ import annotations

    import argparse
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    from pegasus.output.validate import validate_output_bundle


    def fail(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        sys.exit(1)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", required=True)
        args = parser.parse_args()

        run = Path(args.run)
        failures: list[dict] = []

        positive = validate_output_bundle(run_dir=str(run))
        if not positive.ok:
            failures.append({"kind": "positive_validation", "errors": positive.errors})

        with tempfile.TemporaryDirectory(prefix="pegasus_slice2e_validator_") as tmp:
            tmp_root = Path(tmp)

            extra = tmp_root / "extra"
            shutil.copytree(run, extra)
            (extra / "extra_artifact.txt").write_text("poison", encoding="utf-8")
            result = validate_output_bundle(run_dir=str(extra))
            if result.ok or not any("extra first-class artifact" in error for error in result.errors):
                failures.append({"kind": "negative_extra_artifact", "ok": result.ok, "errors": result.errors})

            bad_manifest = tmp_root / "bad_manifest"
            shutil.copytree(run, bad_manifest)
            manifest_path = bad_manifest / "ReproducibilityManifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["telemetry"]["total_wall_seconds"] = -1
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
            result = validate_output_bundle(run_dir=str(bad_manifest))
            if result.ok or not any("total_wall_seconds" in error for error in result.errors):
                failures.append({"kind": "negative_bad_telemetry", "ok": result.ok, "errors": result.errors})

        if failures:
            fail({"status": "failed", "failures": failures})

        print("AUDIT PASSED: Slice 2E output validator accepts valid compile bundle and rejects poisoned bundles.")


    if __name__ == "__main__":
        main()
    ''')

    print("Applied Slice 2E output validator hardening.")
    print("Run the validation commands supplied by the assistant.")


if __name__ == "__main__":
    main()
