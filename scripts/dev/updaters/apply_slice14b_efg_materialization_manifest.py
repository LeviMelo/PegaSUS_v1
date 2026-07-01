from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "14B"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/efg/materialization_manifest.py",
    "src/pegasus/workflows/efg_materialize.py",
    "tests/unit/test_slice14b_efg_materialization_manifest.py",
    "tests/integration/test_slice14b_attach_efg_materialization_to_run.py",
    "scripts/dev/audits/audit_slice14b_efg_materialization_manifest.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/source_registry.py",
    "src/pegasus/she/substrate.py",
    "src/pegasus/output/",
    "src/pegasus/pirs/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)


def fail(message: str) -> None:
    raise SystemExit(f"[slice14b] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_defs(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = (
        "src/pegasus/workflows/compile.py",
        "src/pegasus/efg/materialize.py",
        "src/pegasus/she/substrate.py",
        "src/pegasus/output/validate.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")

    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_defs(compile_text, "run_compile") != 1:
        fail("compile.py must have exactly one public run_compile before Slice 14B")
    if top_level_defs(compile_text, "_run_compile_impl") != 1:
        fail("compile.py must have exactly one private _run_compile_impl before Slice 14B")

    materialize = read("src/pegasus/efg/materialize.py")
    for token in ("materialize_substrate_bundle", "materialized_field_nodes", "SubstrateMaterializationResult"):
        if token not in materialize:
            fail(f"efg/materialize.py missing Slice 14A API: {token}")
    if "BlockedModuleError" in materialize:
        fail("efg/materialize.py still contains blocked scaffold")

    substrate = read("src/pegasus/she/substrate.py")
    for token in ("SubstrateFieldCandidate", "SubstrateFieldExclusion", "SubstrateBundle", "write_substrate_bundle_manifest"):
        if token not in substrate:
            fail(f"substrate.py missing required API: {token}")


def materialization_manifest_module() -> str:
    return dedent(r'''
    """Run-level EFG materialization manifests for SHE substrate bundles.

    Slice 14B makes the Slice 14A substrate-to-FieldNode materializer usable as an
    auditable run artifact.  It reads an existing ``Tables/substrate_manifest.json``
    or standalone substrate manifest, reconstructs a typed SubstrateBundle, emits a
    metadata-only EFG materialization manifest, and optionally attaches a compact
    summary to existing run JSON surfaces.

    It deliberately does not inject these FieldNodes into V_fields, does not build
    E_DAG edges, does not run Q/PIRS/HSIC, and does not add an 18th first-class run
    key.  This is an admission plan and audit artifact, not full EFG integration.
    """

    from __future__ import annotations

    import json
    from dataclasses import fields
    from pathlib import Path
    from typing import Any

    from pegasus.efg.materialize import SubstrateMaterializationResult, materialize_substrate_bundle
    from pegasus.she.substrate import (
        SourceArtifactRef,
        SubstrateBundle,
        SubstrateError,
        SubstrateFieldCandidate,
        SubstrateFieldExclusion,
    )


    def _load_json(path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SubstrateError(f"JSON manifest is not an object: {path}")
        return payload


    def _filter_dataclass_payload(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {field.name for field in fields(cls)}
        return {key: value for key, value in payload.items() if key in allowed}


    def _tuple(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple, set)):
            return tuple(str(item) for item in value)
        return (str(value),)


    def _candidate_from_manifest(payload: dict[str, Any]) -> SubstrateFieldCandidate:
        data = _filter_dataclass_payload(SubstrateFieldCandidate, payload)
        for key in ("role", "provenance", "warnings"):
            if key in data:
                data[key] = _tuple(data[key])
        if "axes" in data and data["axes"] is None:
            data["axes"] = {}
        return SubstrateFieldCandidate(**data)


    def _exclusion_from_manifest(payload: dict[str, Any]) -> SubstrateFieldExclusion:
        data = _filter_dataclass_payload(SubstrateFieldExclusion, payload)
        if "warnings" in data:
            data["warnings"] = _tuple(data["warnings"])
        return SubstrateFieldExclusion(**data)


    def _artifact_from_manifest(payload: dict[str, Any]) -> SourceArtifactRef:
        return SourceArtifactRef(**_filter_dataclass_payload(SourceArtifactRef, payload))


    def substrate_bundle_from_manifest(payload: dict[str, Any]) -> SubstrateBundle:
        """Reconstruct a typed SubstrateBundle from a JSON substrate manifest."""

        if not isinstance(payload, dict):
            raise SubstrateError("Substrate manifest payload is not a dictionary")
        candidates = tuple(_candidate_from_manifest(item) for item in payload.get("candidates", []))
        exclusions = tuple(_exclusion_from_manifest(item) for item in payload.get("exclusions", []))
        artifacts = tuple(_artifact_from_manifest(item) for item in payload.get("source_artifacts", []))
        return SubstrateBundle(
            schema_version=str(payload.get("schema_version", "1.0")),
            substrate_id=str(payload.get("substrate_id", "substrate_bundle_unknown")),
            source_reality_mode=str(payload.get("source_reality_mode", "unknown")),
            source_artifacts=artifacts,
            candidates=candidates,
            exclusions=exclusions,
            table_profiles=(),
            registry_hashes=dict(payload.get("registry_hashes", {})),
            warnings=_tuple(payload.get("warnings", ())),
        )


    def load_substrate_bundle_for_efg(path: str | Path) -> SubstrateBundle:
        """Load a JSON substrate manifest as a typed bundle for EFG planning."""

        return substrate_bundle_from_manifest(_load_json(path))


    def efg_materialization_summary(
        result: SubstrateMaterializationResult,
        *,
        manifest_path: str | None = None,
    ) -> dict[str, Any]:
        """Return the compact run-facing summary for an EFG materialization plan."""

        manifest = result.as_manifest()
        field_columns: list[str] = []
        field_ids: list[str] = []
        for item in manifest.get("fields", []):
            field = item.get("field", {}) if isinstance(item, dict) else {}
            support = field.get("support", {}) if isinstance(field, dict) else {}
            if isinstance(field, dict) and field.get("id"):
                field_ids.append(str(field["id"]))
            if isinstance(support, dict) and support.get("column") is not None:
                field_columns.append(str(support["column"]))
        excluded_columns: list[str] = []
        for item in manifest.get("excluded_source_fields", []):
            if isinstance(item, dict) and item.get("column") is not None:
                excluded_columns.append(str(item["column"]))
        summary = {
            "schema_version": result.schema_version,
            "status": "evaluated",
            "materialization_id": result.materialization_id,
            "substrate_id": result.substrate_id,
            "field_count": result.field_count,
            "excluded_field_count": result.excluded_field_count,
            "metadata_only": True,
            "writes_v_fields": False,
            "writes_e_dag": False,
            "registry_hashes": dict(result.registry_hashes),
            "field_ids": field_ids,
            "field_columns": field_columns,
            "excluded_columns": excluded_columns,
            "warnings": list(result.warnings),
        }
        if manifest_path is not None:
            summary["manifest_path"] = manifest_path
        return summary


    def build_efg_materialization_manifest(*, substrate_manifest: str | Path) -> dict[str, Any]:
        """Build a serializable EFG materialization manifest from a substrate manifest."""

        bundle = load_substrate_bundle_for_efg(substrate_manifest)
        result = materialize_substrate_bundle(bundle)
        payload = result.as_manifest()
        payload["source_substrate_manifest"] = str(substrate_manifest)
        payload["metadata_only"] = True
        payload["writes_v_fields"] = False
        payload["writes_e_dag"] = False
        payload["summary"] = efg_materialization_summary(result)
        return payload


    def write_efg_materialization_manifest(
        *,
        substrate_manifest: str | Path,
        output: str | Path,
    ) -> dict[str, Any]:
        """Write a metadata-only EFG materialization manifest and return it."""

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = build_efg_materialization_manifest(substrate_manifest=substrate_manifest)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return payload


    def attach_efg_materialization_summary_to_run(
        *,
        run_dir: str | Path,
        substrate_manifest: str | Path | None = None,
    ) -> dict[str, Any]:
        """Attach a run-level EFG materialization plan summary.

        The manifest is written under ``Tables/efg_substrate_materialization.json``.
        Existing first-class JSON files receive ``efg_materialization_gate``.  No
        new first-class run key is created.
        """

        root = Path(run_dir)
        tables_dir = root / "Tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        substrate_path = Path(substrate_manifest) if substrate_manifest is not None else tables_dir / "substrate_manifest.json"
        if not substrate_path.exists():
            raise FileNotFoundError(f"Missing substrate manifest for EFG materialization: {substrate_path}")
        output_path = tables_dir / "efg_substrate_materialization.json"
        payload = write_efg_materialization_manifest(substrate_manifest=substrate_path, output=output_path)
        summary = dict(payload["summary"])
        summary["manifest_path"] = str(output_path.relative_to(root)).replace("\\", "/")
        summary["substrate_manifest_path"] = str(substrate_path.relative_to(root)).replace("\\", "/") if substrate_path.is_relative_to(root) else str(substrate_path)
        for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
            path = root / name
            if not path.exists():
                continue
            try:
                target = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(target, dict):
                continue
            target["efg_materialization_gate"] = summary
            if name == "ReproducibilityManifest.json":
                target.setdefault("registry_hashes", {}).update(summary.get("registry_hashes", {}))
            path.write_text(json.dumps(target, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary
    ''')


def workflow_module() -> str:
    return dedent(r'''
    """Workflow wrappers for Slice 14B EFG substrate materialization manifests."""

    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    from pegasus.efg.materialization_manifest import (
        attach_efg_materialization_summary_to_run,
        build_efg_materialization_manifest,
        write_efg_materialization_manifest,
    )


    def run_materialize_substrate_manifest(
        *,
        substrate_manifest: str | Path,
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        if output is None:
            return build_efg_materialization_manifest(substrate_manifest=substrate_manifest)
        return write_efg_materialization_manifest(substrate_manifest=substrate_manifest, output=output)


    def run_attach_efg_materialization_to_run(
        *,
        run_dir: str | Path,
        substrate_manifest: str | Path | None = None,
    ) -> dict[str, Any]:
        return attach_efg_materialization_summary_to_run(run_dir=run_dir, substrate_manifest=substrate_manifest)
    ''')


def unit_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    from pathlib import Path

    from pegasus.efg.materialization_manifest import (
        build_efg_materialization_manifest,
        substrate_bundle_from_manifest,
        write_efg_materialization_manifest,
    )
    from pegasus.she.substrate import (
        SubstrateBundle,
        SubstrateFieldCandidate,
        SubstrateFieldExclusion,
        write_substrate_bundle_manifest,
    )


    def _candidate(column: str = "underlying_icd_norm") -> SubstrateFieldCandidate:
        return SubstrateFieldCandidate(
            candidate_id=f"substrate_candidate_{column}",
            source_system="SIM-DO",
            artifact_path="tests/fixtures/demo.parquet",
            column=column,
            technical_name=f"SIM-DO.{column}",
            carrier="Deaths",
            unit="ICD10",
            aggregation="non_aggregable",
            role=("diagnostic_topology",),
            axes={"diagnosis": "icd10"},
            provenance=("registry", "fixture"),
            substrate_kind="source_measure",
            registry_hash="source_registry_hash:SIM-DO",
            row_count=3,
            non_null_count=3,
            unique_non_null_count=3,
            missing_rate=0.0,
            numeric_min=None,
            numeric_max=None,
            source_manifest_hash="manifest_hash",
            artifact_hash="artifact_hash",
            warnings=(),
        )


    def _bundle() -> SubstrateBundle:
        exclusion = SubstrateFieldExclusion(
            exclusion_id="substrate_exclusion_constant_col",
            source_system="SIM-DO",
            artifact_path="tests/fixtures/demo.parquet",
            column="constant_col",
            reason="zero_variance_constant",
            registry_reason=None,
            row_count=3,
            non_null_count=3,
            unique_non_null_count=1,
            missing_rate=0.0,
            structural_role="constant",
            warnings=("zero_variance_constant",),
        )
        return SubstrateBundle(
            schema_version="1.0",
            substrate_id="substrate_bundle_demo",
            source_reality_mode="fixture_only",
            source_artifacts=(),
            candidates=(_candidate(),),
            exclusions=(exclusion,),
            table_profiles=(),
            registry_hashes={"SIM-DO": "source_registry_hash:SIM-DO"},
            warnings=(),
        )


    def test_slice14b_reconstructs_substrate_bundle_from_manifest_dict() -> None:
        payload = _bundle().as_manifest()
        reconstructed = substrate_bundle_from_manifest(payload)
        assert reconstructed.substrate_id == "substrate_bundle_demo"
        assert len(reconstructed.candidates) == 1
        assert len(reconstructed.exclusions) == 1
        assert reconstructed.candidates[0].column == "underlying_icd_norm"
        assert reconstructed.exclusions[0].column == "constant_col"


    def test_slice14b_writes_metadata_only_efg_manifest(tmp_path: Path) -> None:
        substrate_path = tmp_path / "substrate_manifest.json"
        output = tmp_path / "efg_substrate_materialization.json"
        write_substrate_bundle_manifest(_bundle(), substrate_path)
        payload = write_efg_materialization_manifest(substrate_manifest=substrate_path, output=output)

        assert output.exists()
        assert payload["metadata_only"] is True
        assert payload["writes_v_fields"] is False
        assert payload["writes_e_dag"] is False
        assert payload["field_count"] == 1
        assert payload["excluded_field_count"] == 1
        assert payload["fields"][0]["field"]["unit"] == "ICD10"
        assert payload["fields"][0]["field"]["kind"] == "observer_proxy"
        assert payload["excluded_source_fields"][0]["efg_materialized"] is False
        assert payload["summary"]["field_columns"] == ["underlying_icd_norm"]
        assert payload["summary"]["excluded_columns"] == ["constant_col"]


    def test_slice14b_build_manifest_is_deterministic(tmp_path: Path) -> None:
        substrate_path = tmp_path / "substrate_manifest.json"
        write_substrate_bundle_manifest(_bundle(), substrate_path)
        first = build_efg_materialization_manifest(substrate_manifest=substrate_path)
        second = build_efg_materialization_manifest(substrate_manifest=substrate_path)
        assert first["materialization_id"] == second["materialization_id"]
    ''')


def integration_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import polars as pl

    from pegasus.efg.materialization_manifest import attach_efg_materialization_summary_to_run
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest


    def test_slice14b_attaches_efg_materialization_manifest_without_new_first_class_key(tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        artifact = tmp_path / "sim.parquet"
        create_empty_output_bundle(run_dir)

        pl.DataFrame(
            {
                "year": [2020, 2021, 2021],
                "age_years": [50, 51, 52],
                "race_color_admin": ["1", "4", ""],
                "underlying_icd_norm": ["I10", "J18", "R99"],
                "constant_col": [1, 1, 1],
                "all_missing_col": [None, None, None],
            }
        ).write_parquet(artifact)

        bundle = build_substrate_bundle(
            artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
        )
        substrate_manifest = run_dir / "Tables" / "substrate_manifest.json"
        write_substrate_bundle_manifest(bundle, substrate_manifest)

        before_keys = sorted(path.name for path in run_dir.iterdir())
        summary = attach_efg_materialization_summary_to_run(run_dir=run_dir)
        after_keys = sorted(path.name for path in run_dir.iterdir())

        assert before_keys == after_keys
        assert summary["status"] == "evaluated"
        assert summary["metadata_only"] is True
        assert summary["writes_v_fields"] is False
        assert summary["writes_e_dag"] is False
        assert summary["field_count"] == len(bundle.candidates)
        assert summary["excluded_field_count"] == len(bundle.exclusions)
        assert "constant_col" in summary["excluded_columns"]
        assert "all_missing_col" in summary["excluded_columns"]
        assert (run_dir / "Tables" / "efg_substrate_materialization.json").exists()

        manifest = json.loads((run_dir / "Tables" / "efg_substrate_materialization.json").read_text(encoding="utf-8"))
        materialized_columns = {
            item["field"]["support"]["column"]
            for item in manifest["fields"]
        }
        excluded_columns = {item["column"] for item in manifest["excluded_source_fields"]}
        assert materialized_columns.isdisjoint(excluded_columns)
        assert any(item["field"]["unit"] == "ICD10" and item["field"]["kind"] == "observer_proxy" for item in manifest["fields"])

        for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
            payload = json.loads((run_dir / name).read_text(encoding="utf-8"))
            assert payload["efg_materialization_gate"]["status"] == "evaluated"
            assert payload["efg_materialization_gate"]["manifest_path"] == "Tables/efg_substrate_materialization.json"

        assert validate_output_bundle(run_dir=str(run_dir)).ok
    ''')


def audit_script() -> str:
    return dedent(r'''
    from __future__ import annotations

    import ast
    import json
    import tempfile
    from pathlib import Path

    import polars as pl


    def _count_defs(path: Path, name: str) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


    def main() -> int:
        errors: list[str] = []
        compile_path = Path("src/pegasus/workflows/compile.py")
        if _count_defs(compile_path, "run_compile") != 1:
            errors.append("compile.py must contain exactly one public run_compile")
        if _count_defs(compile_path, "_run_compile_impl") != 1:
            errors.append("compile.py must contain exactly one private _run_compile_impl")

        from pegasus.efg.materialization_manifest import attach_efg_materialization_summary_to_run
        from pegasus.efg.materialize import materialized_field_nodes
        from pegasus.output.bundle import create_empty_output_bundle
        from pegasus.output.validate import validate_output_bundle
        from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            artifact = tmp_path / "sim.parquet"
            create_empty_output_bundle(run_dir)
            pl.DataFrame({
                "year": [2020, 2021, 2021],
                "age_years": [40, 41, 42],
                "underlying_icd_norm": ["I10", "J18", "R99"],
                "constant_col": [1, 1, 1],
                "all_missing_col": [None, None, None],
            }).write_parquet(artifact)
            bundle = build_substrate_bundle(
                artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
            )
            if not any(field.unit == "ICD10" and field.kind == "observer_proxy" for field in materialized_field_nodes(bundle)):
                errors.append("Slice 14A diagnostic observer_proxy materialization is unavailable")
            write_substrate_bundle_manifest(bundle, run_dir / "Tables" / "substrate_manifest.json")
            before_keys = sorted(path.name for path in run_dir.iterdir())
            summary = attach_efg_materialization_summary_to_run(run_dir=run_dir)
            after_keys = sorted(path.name for path in run_dir.iterdir())
            if before_keys != after_keys:
                errors.append("EFG materialization attach created a new first-class run key")
            if summary.get("writes_v_fields") is not False or summary.get("writes_e_dag") is not False:
                errors.append("Slice 14B must not write V_fields or E_DAG")
            if summary.get("field_count") != len(bundle.candidates):
                errors.append("EFG materialization field_count mismatch")
            if summary.get("excluded_field_count") != len(bundle.exclusions):
                errors.append("EFG materialization excluded_field_count mismatch")
            manifest_path = run_dir / "Tables" / "efg_substrate_materialization.json"
            if not manifest_path.exists():
                errors.append("EFG materialization manifest was not written under Tables")
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                materialized_columns = {item["field"]["support"]["column"] for item in manifest.get("fields", [])}
                excluded_columns = {item.get("column") for item in manifest.get("excluded_source_fields", [])}
                if materialized_columns & excluded_columns:
                    errors.append("Excluded substrate fields were promoted in EFG materialization manifest")
            if not validate_output_bundle(run_dir=str(run_dir)).ok:
                errors.append("Output bundle validation failed after EFG materialization attach")

        payload = {"ok": not errors, "errors": errors}
        print(json.dumps(payload, indent=2, sort_keys=True))
        if errors:
            return 1
        print("AUDIT PASSED: Slice 14B EFG materialization manifest attach")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    ''')


def write_files() -> None:
    write("src/pegasus/efg/materialization_manifest.py", materialization_manifest_module())
    write("src/pegasus/workflows/efg_materialize.py", workflow_module())
    write("tests/unit/test_slice14b_efg_materialization_manifest.py", unit_test())
    write("tests/integration/test_slice14b_attach_efg_materialization_to_run.py", integration_test())
    write("scripts/dev/audits/audit_slice14b_efg_materialization_manifest.py", audit_script())


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pegasus.efg.materialization_manifest import build_efg_materialization_manifest  # noqa: F401
    from pegasus.workflows.construct.efg_materialize import run_materialize_substrate_manifest  # noqa: F401


def main() -> None:
    preflight()
    write_files()
    self_validate()
    print("Slice 14B updater applied: EFG materialization manifest attach added.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
