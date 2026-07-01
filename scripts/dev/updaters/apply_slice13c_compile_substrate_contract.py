from __future__ import annotations

import ast
import inspect
import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "13C"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/acceptance/contracts.py",
    "tests/unit/test_slice13c_public_api_stability.py",
    "tests/integration/test_slice13c_compile_substrate_contract.py",
    "scripts/dev/audits/audit_slice13c_compile_substrate_contract.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/efg/",
    "src/pegasus/pirs/",
    "src/pegasus/dashboard/",
    "src/pegasus/she/stdfm/",
    "src/pegasus/she/population/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
)


def fail(message: str) -> None:
    raise SystemExit(f"[slice13c] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if path.startswith(FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_function_nodes(text: str, name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(text)
    return [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]


def preflight() -> None:
    for path in ("src/pegasus/workflows/compile.py", "src/pegasus/she/substrate.py", "src/pegasus/she/source_registry.py"):
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    sr = read("src/pegasus/she/source_registry.py")
    if "class CarrierId" not in sr:
        fail("source_registry.py does not contain CarrierId; apply/commit Slice 13B final repairs before Slice 13C")
    if "allow_heuristic" not in sr:
        fail("source_registry.py resolve_source_fields does not expose allow_heuristic")
    if "source_registry_manifest" not in sr or "source_field_registry_summary" not in sr:
        fail("source_registry.py missing source registry manifest helpers")
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from pegasus.she.source_registry import source_registry_manifest as _slice13c_source_registry_manifest
    _slice13c_manifest = _slice13c_source_registry_manifest()
    if int(_slice13c_manifest.get("entry_count", 0) or 0) < 40:
        fail("source_registry_manifest runtime result does not expose top-level entry_count >= 40")
    sub = read("src/pegasus/she/substrate.py")
    for token in ("build_substrate_bundle", "attach_substrate_summary_to_run", "load_source_artifacts_from_manifest"):
        if token not in sub:
            fail(f"substrate.py missing required API: {token}")
    compile_text = read("src/pegasus/workflows/compile.py")
    defs = top_level_function_nodes(compile_text, "run_compile")
    if len(defs) not in {1, 2}:
        fail(f"compile.py has unsupported run_compile count before consolidation: {len(defs)}")


def strip_existing_slice13c(text: str) -> str:
    marker = "# ---- Slice 13C compile/substrate contract consolidation ----"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip() + "\n"


def consolidate_compile() -> None:
    path = "src/pegasus/workflows/compile.py"
    text = strip_existing_slice13c(read(path))
    lines = text.splitlines()
    tree = ast.parse(text)
    run_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_compile"]

    if len(run_defs) == 2:
        first, second = run_defs
        line = lines[first.lineno - 1]
        if "def run_compile" not in line:
            fail("cannot locate first run_compile definition line for rename")
        lines[first.lineno - 1] = line.replace("def run_compile", "def _run_compile_impl", 1)
        text = "\n".join(lines) + "\n"
        marker_start = text.find("# ---- Slice 13A SHE substrate boundary wrapper ----")
        marker_end = text.find("# ---- End Slice 13A SHE substrate boundary wrapper ----")
        if marker_start != -1:
            if marker_end != -1:
                marker_end = text.find("\n", marker_end)
                if marker_end == -1:
                    marker_end = len(text)
                else:
                    marker_end += 1
            else:
                marker_end = len(text)
            text = text[:marker_start].rstrip() + "\n"
        else:
            fresh_lines = text.splitlines()
            start = second.lineno - 1
            end = second.end_lineno or second.lineno
            del fresh_lines[start:end]
            text = "\n".join(fresh_lines).rstrip() + "\n"
    elif len(run_defs) == 1:
        if "def _run_compile_impl" not in text:
            only = run_defs[0]
            lines = text.splitlines()
            line = lines[only.lineno - 1]
            lines[only.lineno - 1] = line.replace("def run_compile", "def _run_compile_impl", 1)
            text = "\n".join(lines).rstrip() + "\n"
    elif len(run_defs) == 0 and "def _run_compile_impl" in text:
        # Partial prior application: the original public implementation has
        # already been renamed, and strip_existing_slice13c removed the wrapper.
        # Re-append the single public wrapper below.
        text = text.rstrip() + "\n"
    else:
        fail(f"compile.py has unsupported run_compile count during consolidation: {len(run_defs)}")

    if "def _run_compile_impl" not in text:
        fail("compile.py consolidation failed to create _run_compile_impl")

    wrapper = dedent('''

    # ---- Slice 13C compile/substrate contract consolidation ----
    def run_compile(
        *,
        intent_path: str | Path,
        run_dir: str | Path | None = None,
        data_root: str | Path = "data",
        source_manifest: str | Path | None = None,
        require_materialized_external: bool = False,
    ) -> dict[str, Any]:
        # Single public compile boundary. The actual smoke compiler remains in
        # _run_compile_impl(); this wrapper only attaches SHE substrate metadata
        # when source artifact manifests are supplied.
        result = _run_compile_impl(
            intent_path=intent_path,
            run_dir=run_dir,
            data_root=data_root,
            source_manifest=source_manifest,
            require_materialized_external=require_materialized_external,
        )
        if not isinstance(result, dict):
            return result

        if source_manifest is None:
            result.setdefault(
                "substrate_gate",
                {
                    "status": "not_evaluated",
                    "reason": "no_source_manifest_supplied_to_compile",
                    "source_reality_mode": "no_manifest",
                    "registry_backed": None,
                },
            )
            return result

        run_path = result.get("run_dir")
        if run_path is None:
            result["substrate_gate"] = {
                "status": "failed_nonfatal",
                "reason": "compile_result_missing_run_dir",
            }
            return result

        from pegasus.output.validate import validate_output_bundle as _validate_output_bundle
        from pegasus.workflows.construct.build_substrate import run_attach_substrate_to_run as _run_attach_substrate_to_run

        substrate_summary = _run_attach_substrate_to_run(run_dir=run_path, source_manifest=source_manifest)
        result["substrate_gate"] = substrate_summary
        result["validation"] = _validate_output_bundle(run_dir=str(run_path))
        return result
    ''')
    text = text.rstrip() + wrapper
    defs_after = top_level_function_nodes(text, "run_compile")
    if len(defs_after) != 1:
        fail(f"compile.py would have {len(defs_after)} public run_compile definitions after patch")
    if len(top_level_function_nodes(text, "_run_compile_impl")) != 1:
        fail("compile.py would not have exactly one private _run_compile_impl after patch")
    write(path, text)


def patch_acceptance() -> None:
    path = "src/pegasus/acceptance/contracts.py"
    text = read(path)
    if "substrate_present: bool" not in text:
        text = text.replace(
            "    source_reality_production_candidate: bool | None\n    telemetry_stage_status: dict[str, Any]\n",
            "    source_reality_production_candidate: bool | None\n"
            "    substrate_present: bool\n"
            "    substrate_source_reality_mode: str | None\n"
            "    substrate_admissible_candidate_count: int | None\n"
            "    substrate_excluded_field_count: int | None\n"
            "    substrate_registry_backed: bool | None\n"
            "    telemetry_stage_status: dict[str, Any]\n",
        )
    if '"substrate_present": self.substrate_present' not in text:
        text = text.replace(
            '            "source_reality_production_candidate": self.source_reality_production_candidate,\n'
            '            "telemetry_stage_status": self.telemetry_stage_status,\n',
            '            "source_reality_production_candidate": self.source_reality_production_candidate,\n'
            '            "substrate_present": self.substrate_present,\n'
            '            "substrate_source_reality_mode": self.substrate_source_reality_mode,\n'
            '            "substrate_admissible_candidate_count": self.substrate_admissible_candidate_count,\n'
            '            "substrate_excluded_field_count": self.substrate_excluded_field_count,\n'
            '            "substrate_registry_backed": self.substrate_registry_backed,\n'
            '            "telemetry_stage_status": self.telemetry_stage_status,\n',
        )
    substrate_block = dedent('''
        substrate_gate = run_config.get("substrate_gate")
        if not isinstance(substrate_gate, dict):
            substrate_gate = manifest.get("substrate_gate")
        if not isinstance(substrate_gate, dict):
            p_vector = _load_json(root / "P_vector.json")
            substrate_gate = p_vector.get("substrate_gate") if isinstance(p_vector.get("substrate_gate"), dict) else {}
        if not isinstance(substrate_gate, dict):
            substrate_gate = {}
        substrate_present = substrate_gate.get("status") == "evaluated"
        substrate_source_reality_mode = substrate_gate.get("source_reality_mode") if substrate_gate else None
        substrate_admissible_candidate_count = substrate_gate.get("admissible_candidate_count") if substrate_gate else None
        substrate_excluded_field_count = substrate_gate.get("excluded_field_count") if substrate_gate else None
        substrate_registry_backed = substrate_gate.get("registry_backed") if substrate_gate else None
    ''')
    if "substrate_gate = run_config.get(\"substrate_gate\")" not in text:
        anchor = '    stage_status = telemetry.get("stage_status", {}) if isinstance(telemetry.get("stage_status", {}), dict) else {}\n'
        if anchor not in text:
            fail("could not locate acceptance summarize_run telemetry anchor")
        text = text.replace(anchor, anchor + substrate_block + "\n", 1)
    if "substrate_present=substrate_present" not in text:
        text = text.replace(
            "        source_reality_production_candidate=source_reality_production_candidate,\n"
            "        telemetry_stage_status=dict(stage_status),\n",
            "        source_reality_production_candidate=source_reality_production_candidate,\n"
            "        substrate_present=substrate_present,\n"
            "        substrate_source_reality_mode=substrate_source_reality_mode,\n"
            "        substrate_admissible_candidate_count=substrate_admissible_candidate_count,\n"
            "        substrate_excluded_field_count=substrate_excluded_field_count,\n"
            "        substrate_registry_backed=substrate_registry_backed,\n"
            "        telemetry_stage_status=dict(stage_status),\n",
        )
    required = [
        "substrate_present: bool",
        '"substrate_present": self.substrate_present',
        "substrate_present=substrate_present",
    ]
    for token in required:
        if token not in text:
            fail(f"acceptance patch did not install token: {token}")
    write(path, text)


def write_tests() -> None:
    unit_test = r'''
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from pegasus.she.source_registry import resolve_source_field, resolve_source_fields, source_registry_manifest
from pegasus.she.substrate import SourceArtifactRef


def _top_level_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_slice13c_compile_has_single_public_run_compile() -> None:
    path = Path("src/pegasus/workflows/compile.py")
    assert _top_level_defs(path, "run_compile") == 1
    assert _top_level_defs(path, "_run_compile_impl") == 1


def test_slice13c_source_registry_public_api_stable() -> None:
    signature = inspect.signature(resolve_source_fields)
    assert "allow_heuristic" in signature.parameters
    manifest = source_registry_manifest()
    assert manifest["entry_count"] >= 40
    assert manifest["summary"]["entry_count"] == manifest["entry_count"]

    race = resolve_source_field(source_system="SIM-DO", column_name="race_color_admin")
    assert race.registry_backed is True
    assert str(race.spec.carrier) == "Deaths"
    assert race.spec.carrier == "Deaths"
    assert race.spec.carrier == "deaths"

    icd = resolve_source_field(source_system="SIM-DO", column_name="underlying_icd_norm")
    assert icd.spec.unit == "ICD10"
    assert icd.spec.aggregation == "non_aggregable"
    assert "diagnostic_topology" in icd.spec.role


def test_slice13c_source_artifact_ref_default_role_is_stable(tmp_path: Path) -> None:
    artifact = tmp_path / "sim.parquet"
    ref = SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")
    assert ref.artifact_role == "processed_events"
'''
    integration_test = r'''
from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.source_artifacts.contracts import inspect_source_artifact, write_source_artifact_manifest
from pegasus.workflows import compile as compile_workflow


def test_slice13c_run_compile_attaches_substrate_without_duplicate_public_api(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    artifact = tmp_path / "sim.parquet"
    manifest_path = tmp_path / "source_manifest.json"
    create_empty_output_bundle(run_dir)

    pl.DataFrame(
        {
            "year": [2020, 2021, 2021],
            "age_years": [50, 51, 52],
            "race_color_admin": ["1", "4", ""],
            "underlying_icd_norm": ["I10", "J18", "R99"],
            "constant_marker": [1, 1, 1],
            "all_missing_marker": [None, None, None],
        }
    ).write_parquet(artifact)

    source_artifact = inspect_source_artifact(
        path=artifact,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
    )
    write_source_artifact_manifest(artifacts=[source_artifact], output_path=manifest_path)

    def fake_impl(*, intent_path, run_dir=None, data_root="data", source_manifest=None, require_materialized_external=False):
        assert source_manifest == manifest_path
        return {
            "status": "success",
            "run_id": "fake",
            "run_dir": Path(run_dir),
            "validation": validate_output_bundle(run_dir=str(run_dir)),
        }

    monkeypatch.setattr(compile_workflow, "_run_compile_impl", fake_impl)

    result = compile_workflow.run_compile(
        intent_path=tmp_path / "intent.json",
        run_dir=run_dir,
        source_manifest=manifest_path,
    )

    assert result["substrate_gate"]["status"] == "evaluated"
    assert result["substrate_gate"]["admissible_candidate_count"] > 0
    assert result["substrate_gate"]["excluded_field_count"] > 0
    assert result["substrate_gate"]["source_reality_mode"] == "fixture_only"
    assert (run_dir / "Tables" / "substrate_manifest.json").exists()
    assert result["validation"].ok
'''
    write("tests/unit/test_slice13c_public_api_stability.py", unit_test)
    write("tests/integration/test_slice13c_compile_substrate_contract.py", integration_test)


def write_audit() -> None:
    audit = r'''
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    source_registry_path = Path("src/pegasus/she/source_registry.py")

    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must contain exactly one public top-level run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must contain exactly one private _run_compile_impl")
    if _count_defs(source_registry_path, "resolve_source_fields") != 1:
        errors.append("source_registry.py must contain exactly one top-level resolve_source_fields")

    from pegasus.she.source_registry import resolve_source_field, resolve_source_fields, source_registry_manifest
    from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle

    if "allow_heuristic" not in inspect.signature(resolve_source_fields).parameters:
        errors.append("resolve_source_fields missing allow_heuristic parameter")
    manifest = source_registry_manifest()
    if manifest.get("entry_count", 0) < 40:
        errors.append("source_registry_manifest missing top-level entry_count >= 40")
    race = resolve_source_field(source_system="SIM-DO", column_name="race_color_admin")
    if str(race.spec.carrier) != "Deaths":
        errors.append("race_color_admin carrier does not render as canonical Deaths")
    if not (race.spec.carrier == "deaths"):
        errors.append("CarrierId legacy equality with deaths failed")
    icd = resolve_source_field(source_system="SIM-DO", column_name="underlying_icd_norm")
    if icd.spec.unit != "ICD10" or icd.spec.aggregation != "non_aggregable":
        errors.append("underlying_icd_norm did not resolve as ICD10/non_aggregable")
    ref = SourceArtifactRef(path="missing.parquet", source_system="SIM-DO", provenance_mode="fixture")
    if ref.artifact_role != "processed_events":
        errors.append("SourceArtifactRef default artifact_role changed")
    bundle = build_substrate_bundle(artifacts=[ref])
    if bundle.summary().get("excluded_field_count", 0) < 1:
        errors.append("build_substrate_bundle did not quarantine missing artifact")

    payload = {"ok": not errors, "errors": errors, "manifest_entry_count": manifest.get("entry_count")}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 13C compile/substrate contract consolidated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write("scripts/dev/audits/audit_slice13c_compile_substrate_contract.py", audit)


def self_validate() -> None:
    for path in TOUCH_LIST:
        if path.endswith(".py"):
            py_compile.compile(str(ROOT / path), doraise=True)
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    from pegasus.workflows import compile as compile_workflow
    importlib.reload(compile_workflow)
    from pegasus.she.source_registry import resolve_source_fields, source_registry_manifest
    if len(top_level_function_nodes(read("src/pegasus/workflows/compile.py"), "run_compile")) != 1:
        fail("post-validate: compile.py does not have exactly one public run_compile")
    if "allow_heuristic" not in inspect.signature(resolve_source_fields).parameters:
        fail("post-validate: resolve_source_fields missing allow_heuristic")
    if source_registry_manifest().get("entry_count", 0) < 40:
        fail("post-validate: source_registry_manifest entry_count missing")


def main() -> None:
    preflight()
    consolidate_compile()
    patch_acceptance()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 13C updater applied: compile/substrate contract consolidated.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
