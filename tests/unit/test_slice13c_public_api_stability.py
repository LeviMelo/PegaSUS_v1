
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
