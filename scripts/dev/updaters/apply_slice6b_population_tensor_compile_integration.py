
from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path.cwd()


def _path(rel: str) -> Path:
    return ROOT / rel


def read(rel: str) -> str:
    return _path(rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = _path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n").rstrip() + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Cannot patch {label}: expected anchor not found.")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Cannot patch {label}: expected exactly one anchor, found {count}.")
    return text.replace(old, new)


def preflight() -> None:
    required = [
        "src/pegasus/workflows/compile.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/population_tensor_bundle.py",
        "src/pegasus/she/population/solvers.py",
        "src/pegasus/workflows/population.py",
        "config/intents/alagoas_smoke.json",
    ]
    missing = [rel for rel in required if not _path(rel).exists()]
    if missing:
        raise RuntimeError(f"Slice 6B preflight failed; missing files: {missing}")

    compile_text = read("src/pegasus/workflows/compile.py")
    if 'telemetry.block("population_solver", reason="official SIDRA anchor smoke path; tensor solver scaffold remains blocked")' not in compile_text:
        raise RuntimeError("Slice 6B preflight failed: compile.py does not contain the current Slice 6A population_solver block anchor.")
    if 'include_cnes_sih = _context_policy_enabled(intent, "include_cnes_sih")' not in compile_text:
        raise RuntimeError("Slice 6B preflight failed: compile.py does not look like the committed Slice 5B/6A compile flow.")
    if "race_bridge_plan = resolve_race_bridge_plan" not in compile_text:
        raise RuntimeError("Slice 6B preflight failed: compile.py lacks the committed Slice 4B race bridge plan.")

    validate_text = read("src/pegasus/output/validate.py")
    if "def _validate_cnes_sih_contract" not in validate_text or "def _validate_race_bridge_contract" not in validate_text:
        raise RuntimeError("Slice 6B preflight failed: validate.py lacks current race bridge and CNES/SIH validators.")

    print("Slice 6B population tensor compile integration preflight passed.")


def write_intents() -> None:
    base = json.loads(_path("config/intents/alagoas_smoke.json").read_text(encoding="utf-8"))

    independent = dict(base)
    independent["population_mode"] = "independent_population_tensor"
    independent["context_policy"] = sorted(set(independent.get("context_policy", [])) | {"include_population_tensor"})
    _path("config/intents/alagoas_smoke_population_tensor.json").write_text(
        json.dumps(independent, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    sim_informed = dict(base)
    sim_informed["population_mode"] = "sim_informed_population_tensor"
    sim_informed["context_policy"] = sorted(set(sim_informed.get("context_policy", [])) | {"include_population_tensor"})
    _path("config/intents/alagoas_smoke_population_tensor_sim_informed.json").write_text(
        json.dumps(sim_informed, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


POPULATION_TENSOR_COMPILE_ATTACH = r'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from pegasus.core.hashing import sha256_file
from pegasus.output.population_tensor_bundle import (
    _failed_dense_branch,
    _field,
    _q_row,
    _vd_row,
    _warning_rows,
)
from pegasus.she.population.solvers import solve_population_tensor_from_sidra_anchor

POPULATION_TENSOR_FIELD_PREFIX = "population_tensor_"
POPULATION_TENSOR_WARNING_PREFIX = "population_tensor_"
POPULATION_TENSOR_FAILED_BRANCH_IDS = {"failed_dense_national_population_tensor_above_threshold"}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pq.read_table(path).schema
    shaped = [{name: row.get(name) for name in schema.names} for row in rows]
    if shaped:
        table = pa.Table.from_pylist(shaped, schema=schema)
    else:
        table = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    pq.write_table(table, path)


def _append_replace(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    id_column: str,
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
) -> None:
    remove_ids = remove_ids or set()
    existing = _read_rows(path)

    def keep(row: dict[str, Any]) -> bool:
        value = str(row.get(id_column, ""))
        if value in remove_ids:
            return False
        return not any(value.startswith(prefix) for prefix in remove_prefixes)

    _write_rows_like(path, [row for row in existing if keep(row)] + rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\\n", encoding="utf-8")


def _metadata(*, result, field: dict[str, Any], sidra_facts_path: Path, mode: str) -> dict[str, Any]:
    manifest = result.as_manifest()
    manifest.update(
        {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "population_solver",
            "field_id": field["field_id"],
            "field_name": field["name"],
            "requested_population_mode": mode,
            "sidra_facts_path": str(sidra_facts_path),
            "source_hashes": {"sidra_facts": sha256_file(sidra_facts_path)},
            "independent_denominator_mode": result.mode == "independent_denominator",
            "sim_feedback_warning": bool(result.denominator_feedback_warning),
            "dashboard_safe": field.get("dashboard_safe"),
            "materialization_state": field.get("materialization_state"),
            "table_paths": {"diagnostics": "Tables/population_tensor_diagnostics.parquet"},
        }
    )
    return manifest


def attach_population_tensor_compile_fields(
    *,
    run_dir: str | Path,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    sidra_facts_path = Path(sidra_facts_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not sidra_facts_path.exists():
        raise FileNotFoundError(f"sidra_facts_path does not exist: {sidra_facts_path}")

    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=sidra_facts_path, mode=mode)
    field = _field(result)
    q_row = _q_row(field, result)
    vd_row = _vd_row(field, result)
    warning_rows = _warning_rows(field, result)
    failed_row = _failed_dense_branch(field)
    meta = _metadata(result=result, field=field, sidra_facts_path=sidra_facts_path, mode=mode)

    _append_replace(run_dir / "V_fields.parquet", [field], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "Q_tensor.parquet", [q_row], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "VariableDictionary.parquet", [vd_row], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "Warnings.parquet", warning_rows, id_column="warning_id", remove_prefixes=(POPULATION_TENSOR_WARNING_PREFIX,))
    _append_replace(run_dir / "FailedBranches.parquet", [failed_row], id_column="failed_branch_id", remove_ids=POPULATION_TENSOR_FAILED_BRANCH_IDS)

    (run_dir / "Tables").mkdir(exist_ok=True)
    pl.DataFrame([meta]).write_parquet(run_dir / "Tables" / "population_tensor_diagnostics.parquet")

    for json_name in ["RunConfig.json", "ReproducibilityManifest.json", "P_vector.json"]:
        path = run_dir / json_name
        payload = _load_json(path)
        payload["population_tensor"] = meta
        if json_name == "P_vector.json":
            payload.setdefault("source_systems", [])
            if "SIDRA" not in payload["source_systems"]:
                payload["source_systems"].append("SIDRA")
        _write_json(path, payload)

    return meta
'''


def write_population_tensor_compile_attach() -> None:
    write("src/pegasus/output/population_tensor_compile_attach.py", POPULATION_TENSOR_COMPILE_ATTACH)


def patch_compile() -> None:
    rel = "src/pegasus/workflows/compile.py"
    text = read(rel)

    if "from pegasus.output.population_tensor_compile_attach import attach_population_tensor_compile_fields" not in text:
        text = replace_once(
            text,
            "from pegasus.output.maternal_child_compile_attach import attach_maternal_child_compile_fields\n",
            "from pegasus.output.maternal_child_compile_attach import attach_maternal_child_compile_fields\n"
            "from pegasus.output.population_tensor_compile_attach import attach_population_tensor_compile_fields\n",
            label="compile.py import insertion",
        )

    if "def _compile_population_tensor_mode(" not in text:
        marker = "\ndef run_compile("
        if marker not in text:
            raise RuntimeError("Cannot patch compile.py population tensor mode helper: run_compile marker not found.")
        helper_block = (
            'def _compile_population_tensor_mode(intent: UserIntent) -> str | None:\n'
            '    if intent.population_mode == "independent_population_tensor":\n'
            '        return "independent_denominator"\n'
            '    if intent.population_mode == "sim_informed_population_tensor":\n'
            '        return "sim_informed_denominator"\n'
            '    return None\n\n'
        )
        text = text.replace(marker, "\n" + helper_block + marker.lstrip("\n"), 1)

    if 'population_tensor_mode = _compile_population_tensor_mode(intent)' not in text:
        text = replace_once(
            text,
            '    include_cnes_sih = _context_policy_enabled(intent, "include_cnes_sih")\n',
            '    include_cnes_sih = _context_policy_enabled(intent, "include_cnes_sih")\n'
            '    population_tensor_mode = _compile_population_tensor_mode(intent)\n',
            label="compile.py population mode variable",
        )

    if 'population_tensor_metadata: dict[str, Any] | None = None' not in text:
        text = replace_once(
            text,
            '    cnes_sih_metadata: dict[str, Any] | None = None\n    with telemetry.stage("she_build"):\n',
            '    cnes_sih_metadata: dict[str, Any] | None = None\n'
            '    population_tensor_metadata: dict[str, Any] | None = None\n'
            '    with telemetry.stage("she_build"):\n',
            label="compile.py population metadata variable",
        )

    if 'attach_population_tensor_compile_fields(' not in text:
        text = replace_once(
            text,
            '''            source_hashes.update(cnes_sih_metadata.get("source_hashes", {}))

    race_bridge_metadata: dict[str, Any] | None = None
''',
            '''            source_hashes.update(cnes_sih_metadata.get("source_hashes", {}))

    if population_tensor_mode is not None:
        with telemetry.stage("population_solver"):
            population_tensor_metadata = attach_population_tensor_compile_fields(
                run_dir=run_dir,
                sidra_facts_path=sidra_facts_path,
                mode=population_tensor_mode,
            )
            source_hashes.update(
                {
                    f"population_tensor_{key}": value
                    for key, value in population_tensor_metadata.get("source_hashes", {}).items()
                }
            )
            diagnostics_path = Path(run_dir) / "Tables" / "population_tensor_diagnostics.parquet"
            if diagnostics_path.exists():
                source_hashes["population_tensor_diagnostics"] = sha256_file(diagnostics_path)

    race_bridge_metadata: dict[str, Any] | None = None
''',
            label="compile.py population tensor attach block",
        )

    if 'if population_tensor_metadata is None:\n        telemetry.block("population_solver"' not in text:
        text = replace_once(
            text,
            '    telemetry.block("population_solver", reason="official SIDRA anchor smoke path; tensor solver scaffold remains blocked")\n',
            '    if population_tensor_metadata is None:\n'
            '        telemetry.block("population_solver", reason="official SIDRA anchor smoke path; tensor solver scaffold remains blocked")\n',
            label="compile.py population telemetry conditional block",
        )

    if '        if population_tensor_metadata is None and existing_run_config.get("population_tensor"):' not in text:
        text = replace_once(
            text,
            '''        if cnes_sih_metadata is None and existing_run_config.get("cnes_sih"):
            cnes_sih_metadata = existing_run_config["cnes_sih"]
        if cnes_sih_metadata is not None:
            run_config_payload["cnes_sih"] = cnes_sih_metadata
''',
            '''        if cnes_sih_metadata is None and existing_run_config.get("cnes_sih"):
            cnes_sih_metadata = existing_run_config["cnes_sih"]
        if cnes_sih_metadata is not None:
            run_config_payload["cnes_sih"] = cnes_sih_metadata
        if population_tensor_metadata is None and existing_run_config.get("population_tensor"):
            population_tensor_metadata = existing_run_config["population_tensor"]
        if population_tensor_metadata is not None:
            run_config_payload["population_tensor"] = population_tensor_metadata
''',
            label="compile.py run config population tensor metadata",
        )

    if '        if population_tensor_metadata is not None:\n            manifest_extras["population_tensor"] = population_tensor_metadata' not in text:
        text = replace_once(
            text,
            '''        if cnes_sih_metadata is not None:
            manifest_extras["cnes_sih"] = cnes_sih_metadata
        write_reproducibility_manifest(
''',
            '''        if cnes_sih_metadata is not None:
            manifest_extras["cnes_sih"] = cnes_sih_metadata
        if population_tensor_metadata is not None:
            manifest_extras["population_tensor"] = population_tensor_metadata
        write_reproducibility_manifest(
''',
            label="compile.py manifest extras population tensor first write",
        )

    if '    if population_tensor_metadata is not None:\n        final_extras["population_tensor"] = population_tensor_metadata' not in text:
        text = replace_once(
            text,
            '''    if cnes_sih_metadata is not None:
        final_extras["cnes_sih"] = cnes_sih_metadata
    write_reproducibility_manifest(
''',
            '''    if cnes_sih_metadata is not None:
        final_extras["cnes_sih"] = cnes_sih_metadata
    if population_tensor_metadata is not None:
        final_extras["population_tensor"] = population_tensor_metadata
    write_reproducibility_manifest(
''',
            label="compile.py final extras population tensor",
        )

    if '        "population_tensor": population_tensor_metadata,' not in text:
        text = replace_once(
            text,
            '        "cnes_sih": cnes_sih_metadata,\n',
            '        "cnes_sih": cnes_sih_metadata,\n'
            '        "population_tensor": population_tensor_metadata,\n',
            label="compile.py return population tensor",
        )

    write(rel, text)


def patch_validate() -> None:
    rel = "src/pegasus/output/validate.py"
    text = read(rel)

    if "POPULATION_TENSOR_SUPPORT_KEYS" not in text:
        text = replace_once(
            text,
            'RUN_CONFIG_CNES_SIH_KEYS = {"schema_version", "source_systems", "attach_stage", "cnes", "sih"}\n',
            'RUN_CONFIG_CNES_SIH_KEYS = {"schema_version", "source_systems", "attach_stage", "cnes", "sih"}\n'
            'POPULATION_TENSOR_SUPPORT_KEYS = {"PopulationTensorMode", "SolverBackend", "SolverID", "SparseJacobian", "DenominatorFeedbackWarning", "population_tensor_diagnostics"}\n'
            'POPULATION_TENSOR_AXIS_KEYS = {"geography_axis", "time_axis", "population_strata_axis", "population_tensor_mode"}\n'
            'RUN_CONFIG_POPULATION_TENSOR_KEYS = {"schema_version", "source_systems", "attach_stage", "field_id", "tensor_id", "mode", "solver_id", "solver_backend", "denominator_feedback_warning", "independent_denominator_mode", "sim_feedback_warning", "source_hashes"}\n',
            label="validate.py population tensor constants",
        )

    if "def _validate_population_tensor_contract" not in text:
        function = r'''
def _validate_population_tensor_contract(*, root: Path, v, q, run_config: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
    rows = v.to_pylist()
    q_rows = {str(row.get("field_id")): row for row in q.to_pylist() if row.get("field_id") is not None}
    population_rows = [row for row in rows if str(row.get("field_id", "")).startswith("population_tensor_")]
    if not population_rows and "population_tensor" not in run_config and "population_tensor" not in manifest:
        return

    run_meta = run_config.get("population_tensor")
    manifest_meta = manifest.get("population_tensor")
    if not isinstance(run_meta, dict):
        errors.append("RunConfig.json missing population_tensor metadata while population tensor is present")
        run_meta = {}
    if not isinstance(manifest_meta, dict):
        errors.append("ReproducibilityManifest.json missing population_tensor metadata while population tensor is present")
        manifest_meta = {}

    missing_run_keys = sorted(RUN_CONFIG_POPULATION_TENSOR_KEYS - set(run_meta))
    if missing_run_keys:
        errors.append(f"RunConfig.population_tensor missing keys: {missing_run_keys}")
    if run_meta.get("source_systems") != ["SIDRA"]:
        errors.append("RunConfig.population_tensor.source_systems must equal ['SIDRA']")
    if run_meta.get("attach_stage") not in {"population_solver", "standalone_population_tensor"}:
        errors.append("RunConfig.population_tensor.attach_stage must be population_solver or standalone_population_tensor")
    if manifest_meta.get("field_id") and run_meta.get("field_id") and manifest_meta.get("field_id") != run_meta.get("field_id"):
        errors.append("RunConfig.population_tensor.field_id and ReproducibilityManifest.population_tensor.field_id disagree")
    if not (root / "Tables" / "population_tensor_diagnostics.parquet").exists():
        errors.append("population tensor metadata exists but Tables/population_tensor_diagnostics.parquet is missing")

    for row in population_rows:
        fid = str(row.get("field_id"))
        support = _load_json_cell(row.get("support_json"), errors=errors, context=f"V_fields.support_json[{fid}]")
        axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
        provenance = _load_json_cell(row.get("provenance_json"), errors=errors, context=f"V_fields.provenance_json[{fid}]")
        warnings_cell = _load_json_cell(row.get("warnings_json"), errors=errors, context=f"V_fields.warnings_json[{fid}]")

        if not isinstance(support, dict):
            errors.append(f"population tensor field support_json is not an object: {fid}")
            support = {}
        if not isinstance(axes, dict):
            errors.append(f"population tensor field axes_json is not an object: {fid}")
            axes = {}
        if not isinstance(provenance, list) or "population_tensor" not in provenance:
            errors.append(f"population tensor field provenance must include population_tensor: {fid}")

        missing_support = sorted(POPULATION_TENSOR_SUPPORT_KEYS - set(support))
        if missing_support:
            errors.append(f"population tensor field missing support metadata: {fid} {missing_support}")
        missing_axes = sorted(POPULATION_TENSOR_AXIS_KEYS - set(axes))
        if missing_axes:
            errors.append(f"population tensor field missing axes metadata: {fid} {missing_axes}")
        if axes.get("population_tensor_mode") != support.get("PopulationTensorMode"):
            errors.append(f"population tensor support/axes mode mismatch: {fid}")
        if q_rows.get(fid) is None:
            errors.append(f"population tensor field missing Q_tensor row: {fid}")

        sim_feedback = bool(support.get("DenominatorFeedbackWarning"))
        if sim_feedback:
            if row.get("dashboard_safe") in {True, "true", "True"}:
                errors.append(f"SIM-informed population tensor field cannot be dashboard_safe=true: {fid}")
            if row.get("state") not in {"fragile", "experimental", "blocked", "quarantined"}:
                errors.append(f"SIM-informed population tensor field must be fragile/experimental or worse: {fid}")
            if isinstance(warnings_cell, list) and "sim_informed_population_feedback_risk" not in warnings_cell:
                errors.append(f"SIM-informed population tensor field missing feedback-risk warning: {fid}")
'''.rstrip() + "\n\n"
        text = replace_once(
            text,
            "\ndef _validate_parquet_contracts(",
            "\n" + function + "\ndef _validate_parquet_contracts(",
            label="validate.py population tensor validator insertion",
        )

    if "_validate_population_tensor_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)" not in text:
        text = replace_once(
            text,
            "    _validate_cnes_sih_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)\n    _validate_race_bridge_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)\n",
            "    _validate_cnes_sih_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)\n"
            "    _validate_population_tensor_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)\n"
            "    _validate_race_bridge_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)\n",
            label="validate.py population tensor validator call",
        )

    write(rel, text)


def patch_population_bundle_metadata() -> None:
    rel = "src/pegasus/output/population_tensor_bundle.py"
    text = read(rel)
    if '"attach_stage": "standalone_population_tensor"' not in text:
        text = replace_once(
            text,
            '''    population_manifest = result.as_manifest()
    user_intent = {
''',
            '''    population_manifest = result.as_manifest()
    population_manifest.update(
        {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "standalone_population_tensor",
            "field_id": field["field_id"],
            "field_name": field["name"],
            "source_hashes": {"sidra_facts": sha256_file(sidra_facts_path)},
            "independent_denominator_mode": result.mode == "independent_denominator",
            "sim_feedback_warning": bool(result.denominator_feedback_warning),
            "dashboard_safe": field.get("dashboard_safe"),
            "materialization_state": field.get("materialization_state"),
            "table_paths": {"diagnostics": "Tables/population_tensor_diagnostics.parquet"},
        }
    )
    user_intent = {
''',
            label="population_tensor_bundle.py stronger metadata",
        )
    write(rel, text)


def patch_workflows_population() -> None:
    rel = "src/pegasus/workflows/population.py"
    text = read(rel)
    if "run_attach_population_tensor_compile_fields" in text:
        return
    if "from pegasus.output.population_tensor_compile_attach import attach_population_tensor_compile_fields" not in text:
        text = text.replace(
            "from pegasus.output.population_tensor_bundle import write_population_tensor_fixture_bundle\n",
            "from pegasus.output.population_tensor_bundle import write_population_tensor_fixture_bundle\n"
            "from pegasus.output.population_tensor_compile_attach import attach_population_tensor_compile_fields\n",
        )
    text += r'''

def run_attach_population_tensor_compile_fields(
    *,
    run_dir: str | Path,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
) -> dict[str, object]:
    metadata = attach_population_tensor_compile_fields(
        run_dir=run_dir,
        sidra_facts_path=sidra_facts_path,
        mode=mode,
    )
    return {"status": "success", "population_tensor": metadata, "run_dir": str(run_dir)}
'''
    write(rel, text)


def write_tests_and_audit() -> None:
    write(
        "tests/integration/test_slice6b_compile_population_tensor_integration.py",
        r'''
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile


def _field_ids(run_dir: Path) -> set[str]:
    return set(pl.read_parquet(run_dir / "V_fields.parquet")["field_id"].to_list())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_slice6b_compile_attaches_independent_population_tensor(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_population_tensor"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_population_tensor.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors

    ids = _field_ids(run_dir)
    assert "population_tensor_independent_denominator" in ids

    run_config = _load_json(run_dir / "RunConfig.json")
    manifest = _load_json(run_dir / "ReproducibilityManifest.json")
    assert run_config["population_tensor"]["mode"] == "independent_denominator"
    assert run_config["population_tensor"]["attach_stage"] == "population_solver"
    assert manifest["population_tensor"]["mode"] == "independent_denominator"
    assert manifest["telemetry"]["stage_status"]["population_solver"] == "success"
    assert (run_dir / "Tables" / "population_tensor_diagnostics.parquet").exists()

    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    assert "population_tensor_independent_denominator" in set(q["field_id"].to_list())

    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors


def test_slice6b_compile_baseline_remains_official_anchor_without_population_tensor(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_baseline"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors
    assert not any(field_id.startswith("population_tensor_") for field_id in _field_ids(run_dir))

    manifest = _load_json(run_dir / "ReproducibilityManifest.json")
    assert manifest["telemetry"]["stage_status"]["population_solver"] == "blocked"


def test_slice6b_compile_sim_informed_population_tensor_is_warning_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_sim_informed_population_tensor"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_population_tensor_sim_informed.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors

    ids = _field_ids(run_dir)
    assert "population_tensor_sim_informed_denominator" in ids

    fields = pl.read_parquet(run_dir / "V_fields.parquet").to_dicts()
    field = next(row for row in fields if row["field_id"] == "population_tensor_sim_informed_denominator")
    assert field["dashboard_safe"] != "true"
    assert field["state"] in {"fragile", "experimental", "blocked", "quarantined"}

    warnings = pl.read_parquet(run_dir / "Warnings.parquet").to_dicts()
    assert any(row.get("code") == "sim_informed_population_feedback_risk" for row in warnings)

    run_config = _load_json(run_dir / "RunConfig.json")
    assert run_config["population_tensor"]["sim_feedback_warning"] is True
    assert validate_output_bundle(run_dir=str(run_dir)).ok


def test_slice6b_validator_rejects_population_tensor_metadata_erasure(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_population_tensor_invalid"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_population_tensor.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors

    run_config_path = run_dir / "RunConfig.json"
    run_config = _load_json(run_config_path)
    del run_config["population_tensor"]["solver_id"]
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    validation = validate_output_bundle(run_dir=str(run_dir))
    assert not validation.ok
    assert any("RunConfig.population_tensor missing keys" in error for error in validation.errors)
''',
    )

    write(
        "scripts/dev/audits/audit_slice6b_compile_population_tensor.py",
        r'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Slice 6B compile-integrated Population Tensor contract.")
    parser.add_argument("--run", required=True, help="Compile run directory emitted from alagoas_smoke_population_tensor.json")
    args = parser.parse_args()

    run_dir = Path(args.run)
    errors: list[str] = []

    validation = validate_output_bundle(run_dir=str(run_dir))
    if not validation.ok:
        errors.extend(validation.errors)

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    warnings = pl.read_parquet(run_dir / "Warnings.parquet")
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

    ids = set(v["field_id"].to_list())
    if "population_tensor_independent_denominator" not in ids:
        errors.append("population_tensor_independent_denominator field missing from V_fields")

    q_ids = set(q["field_id"].to_list())
    if "population_tensor_independent_denominator" not in q_ids:
        errors.append("population_tensor_independent_denominator missing from Q_tensor")

    meta = run_config.get("population_tensor")
    if not isinstance(meta, dict):
        errors.append("RunConfig.population_tensor missing")
        meta = {}
    if meta.get("mode") != "independent_denominator":
        errors.append(f"unexpected population tensor mode: {meta.get('mode')!r}")
    if meta.get("independent_denominator_mode") is not True:
        errors.append("independent_denominator_mode must be true for independent tensor compile run")
    if meta.get("attach_stage") != "population_solver":
        errors.append("population tensor attach_stage must be population_solver")

    manifest_meta = manifest.get("population_tensor")
    if not isinstance(manifest_meta, dict):
        errors.append("ReproducibilityManifest.population_tensor missing")
    elif manifest_meta.get("field_id") != meta.get("field_id"):
        errors.append("RunConfig/ReproducibilityManifest population_tensor field_id mismatch")

    if manifest.get("telemetry", {}).get("stage_status", {}).get("population_solver") != "success":
        errors.append("telemetry.population_solver must be success")

    if not (run_dir / "Tables" / "population_tensor_diagnostics.parquet").exists():
        errors.append("population tensor diagnostics table missing")

    warning_codes = set(warnings["code"].to_list()) if warnings.height and "code" in warnings.columns else set()
    if "independent_population_denominator_mode" not in warning_codes:
        errors.append("independent population denominator warning code missing")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("AUDIT PASSED: Slice 6B compile-integrated Population Tensor metadata, independent denominator mode, telemetry, Q coverage, and validator enforcement validated.")


if __name__ == "__main__":
    main()
''',
    )


def main() -> None:
    preflight()
    print("CREATE/REPLACE:")
    for rel in [
        "config/intents/alagoas_smoke_population_tensor.json",
        "config/intents/alagoas_smoke_population_tensor_sim_informed.json",
        "src/pegasus/output/population_tensor_compile_attach.py",
        "tests/integration/test_slice6b_compile_population_tensor_integration.py",
        "scripts/dev/audits/audit_slice6b_compile_population_tensor.py",
    ]:
        print(f"  {rel}")
    print("PATCH:")
    for rel in [
        "src/pegasus/workflows/compile.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/population_tensor_bundle.py",
        "src/pegasus/workflows/population.py",
    ]:
        print(f"  {rel}")

    write_intents()
    write_population_tensor_compile_attach()
    patch_compile()
    patch_validate()
    patch_population_bundle_metadata()
    patch_workflows_population()
    write_tests_and_audit()

    print("Slice 6B population tensor compile integration updater applied.")


if __name__ == "__main__":
    main()
