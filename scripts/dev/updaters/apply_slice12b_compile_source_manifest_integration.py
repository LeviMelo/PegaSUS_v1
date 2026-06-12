from __future__ import annotations

from pathlib import Path
from textwrap import dedent
import zipfile

ROOT = Path.cwd()


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dedent(text).lstrip(), encoding="utf-8")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected patch anchor not found in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


COMPILE_POLICY = r'''
"""Compile-time source-reality policy for manifest-backed PegaSUS runs.

Slice 12B does not perform source acquisition and does not change SHE/EFG.
It makes source provenance explicit at compile time so fixture-backed runs cannot
be mistaken for materialized production runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pegasus.core.hashing import sha256_file
from pegasus.source_artifacts.contracts import (
    source_manifest_summary,
    validate_source_artifact_manifest,
)


class CompileSourceRealityError(ValueError):
    """Raised when compile source-artifact policy is violated."""


@dataclass(frozen=True)
class CompileSourceReality:
    schema_version: str
    compile_source_mode: str
    source_artifact_manifest_present: bool
    source_artifact_manifest_path: str | None
    source_artifact_manifest_hash: str | None
    source_artifact_count: int
    source_systems: tuple[str, ...]
    artifact_roles: tuple[str, ...]
    require_materialized_external: bool
    production_candidate: bool
    validation_warnings: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_systems"] = list(self.source_systems)
        payload["artifact_roles"] = list(self.artifact_roles)
        payload["validation_warnings"] = list(self.validation_warnings)
        return payload


def resolve_compile_source_reality(
    *,
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> CompileSourceReality:
    """Resolve compile source mode before running the compiler.

    Missing manifests are explicitly treated as fixture-only. This preserves the
    historical smoke workflow while making it impossible for a strict production
    compile to proceed without materialized external artifacts.
    """

    if source_manifest is None:
        if require_materialized_external:
            raise CompileSourceRealityError(
                "compile requires materialized external source artifacts, but no source manifest was provided"
            )
        return CompileSourceReality(
            schema_version="1.0",
            compile_source_mode="fixture_only",
            source_artifact_manifest_present=False,
            source_artifact_manifest_path=None,
            source_artifact_manifest_hash=None,
            source_artifact_count=0,
            source_systems=(),
            artifact_roles=(),
            require_materialized_external=False,
            production_candidate=False,
            validation_warnings=("compile_source_manifest_missing_assumed_fixture_only",),
        )

    manifest_path = Path(source_manifest)
    result = validate_source_artifact_manifest(
        manifest_path=manifest_path,
        require_materialized_external=require_materialized_external,
    )
    if not result.get("ok", False):
        errors = "; ".join(str(x) for x in result.get("errors", [])) or "unknown source artifact error"
        raise CompileSourceRealityError(errors)

    summary = source_manifest_summary(manifest_path=manifest_path)
    mode = str(summary.get("compile_source_mode") or result.get("compile_source_mode") or "unknown")
    return CompileSourceReality(
        schema_version="1.0",
        compile_source_mode=mode,
        source_artifact_manifest_present=True,
        source_artifact_manifest_path=str(manifest_path),
        source_artifact_manifest_hash=sha256_file(manifest_path),
        source_artifact_count=int(summary.get("artifact_count") or result.get("artifact_count") or 0),
        source_systems=tuple(str(x) for x in result.get("source_systems", [])),
        artifact_roles=tuple(str(x) for x in result.get("artifact_roles", [])),
        require_materialized_external=bool(require_materialized_external),
        production_candidate=mode == "materialized_external",
        validation_warnings=tuple(str(x) for x in result.get("warnings", [])),
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def attach_compile_source_reality(*, run_dir: str | Path, source_reality: CompileSourceReality) -> None:
    """Attach source-reality metadata to existing 17-key run JSONs.

    This function deliberately does not add a new first-class file, because the
    output bundle is exact-key validated. Metadata is embedded only into existing
    JSON keys: RunConfig, P_vector, UserIntent, and ReproducibilityManifest.
    """

    root = Path(run_dir)
    payload = source_reality.as_manifest()

    run_config_path = root / "RunConfig.json"
    run_config = _load_json(run_config_path)
    run_config.update(
        {
            "compile_source_mode": payload["compile_source_mode"],
            "source_artifact_manifest_present": payload["source_artifact_manifest_present"],
            "source_artifact_manifest_path": payload["source_artifact_manifest_path"],
            "source_artifact_manifest_hash": payload["source_artifact_manifest_hash"],
            "source_artifact_count": payload["source_artifact_count"],
            "source_artifact_systems": payload["source_systems"],
            "source_artifact_roles": payload["artifact_roles"],
            "require_materialized_external": payload["require_materialized_external"],
            "source_reality_production_candidate": payload["production_candidate"],
            "source_artifact_reality": payload,
        }
    )
    _write_json(run_config_path, run_config)

    p_vector_path = root / "P_vector.json"
    p_vector = _load_json(p_vector_path)
    p_vector["source_artifact_reality"] = payload
    _write_json(p_vector_path, p_vector)

    user_intent_path = root / "UserIntent.json"
    user_intent = _load_json(user_intent_path)
    user_intent["compile_source_reality"] = {
        "compile_source_mode": payload["compile_source_mode"],
        "source_artifact_manifest_present": payload["source_artifact_manifest_present"],
        "source_artifact_manifest_path": payload["source_artifact_manifest_path"],
        "require_materialized_external": payload["require_materialized_external"],
        "production_candidate": payload["production_candidate"],
    }
    _write_json(user_intent_path, user_intent)

    manifest_path = root / "ReproducibilityManifest.json"
    manifest = _load_json(manifest_path)
    manifest["compile_source_mode"] = payload["compile_source_mode"]
    manifest["source_artifact_manifest_path"] = payload["source_artifact_manifest_path"]
    manifest["source_artifact_manifest_hash"] = payload["source_artifact_manifest_hash"]
    manifest["source_artifact_reality"] = payload
    source_hashes = manifest.get("source_hashes")
    if isinstance(source_hashes, dict) and payload["source_artifact_manifest_hash"]:
        source_hashes["source_artifact_manifest"] = payload["source_artifact_manifest_hash"]
        manifest["source_hashes"] = source_hashes
    _write_json(manifest_path, manifest)
'''


TEST_POLICY = r'''
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pegasus.source_artifacts.compile_policy import (
    CompileSourceRealityError,
    resolve_compile_source_reality,
)
from pegasus.source_artifacts.contracts import inspect_source_artifact, write_source_artifact_manifest


def test_slice12b_missing_source_manifest_is_explicit_fixture_only() -> None:
    reality = resolve_compile_source_reality()
    assert reality.compile_source_mode == "fixture_only"
    assert reality.source_artifact_manifest_present is False
    assert reality.production_candidate is False
    assert "compile_source_manifest_missing_assumed_fixture_only" in reality.validation_warnings


def test_slice12b_strict_compile_requires_manifest() -> None:
    with pytest.raises(CompileSourceRealityError):
        resolve_compile_source_reality(require_materialized_external=True)


def test_slice12b_materialized_external_manifest_is_production_candidate(tmp_path: Path) -> None:
    artifact_path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2020]}).write_parquet(artifact_path)
    artifact = inspect_source_artifact(
        path=artifact_path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="materialized_external",
        source_manifest_hash="datasus_manifest_hash",
    )
    manifest_path = tmp_path / "source_artifacts.json"
    write_source_artifact_manifest(artifacts=[artifact], output_path=manifest_path)

    reality = resolve_compile_source_reality(
        source_manifest=manifest_path,
        require_materialized_external=True,
    )

    assert reality.compile_source_mode == "materialized_external"
    assert reality.source_artifact_manifest_present is True
    assert reality.production_candidate is True
    assert reality.source_artifact_count == 1
    assert reality.source_systems == ("SIM-DO",)
'''


TEST_INTEGRATION = r'''
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pegasus.acceptance.contracts import summarize_run
from pegasus.source_artifacts.compile_policy import CompileSourceRealityError
from pegasus.workflows.compile import run_compile


def test_slice12b_compile_records_missing_manifest_as_fixture_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_fixture_only"
    result = run_compile(
        intent_path=Path("config/intents/alagoas_smoke.json"),
        run_dir=run_dir,
    )
    assert result["validation"].ok is True

    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    user_intent = json.loads((run_dir / "UserIntent.json").read_text(encoding="utf-8"))
    repro = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

    assert run_config["compile_source_mode"] == "fixture_only"
    assert run_config["source_artifact_manifest_present"] is False
    assert run_config["source_reality_production_candidate"] is False
    assert p_vector["source_artifact_reality"]["compile_source_mode"] == "fixture_only"
    assert user_intent["compile_source_reality"]["compile_source_mode"] == "fixture_only"
    assert repro["compile_source_mode"] == "fixture_only"

    summary = summarize_run(run_dir)
    manifest = summary.as_manifest()
    assert manifest["compile_source_mode"] == "fixture_only"
    assert manifest["source_artifact_manifest_present"] is False


def test_slice12b_compile_strict_mode_aborts_without_manifest(tmp_path: Path) -> None:
    with pytest.raises(CompileSourceRealityError):
        run_compile(
            intent_path=Path("config/intents/alagoas_smoke.json"),
            run_dir=tmp_path / "strict_compile",
            require_materialized_external=True,
        )
'''


AUDIT = r'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pegasus.acceptance.contracts import summarize_run
from pegasus.source_artifacts.compile_policy import resolve_compile_source_reality


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Slice 12B compile source-reality integration.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected-mode", default="fixture_only")
    args = parser.parse_args()

    run_dir = Path(args.run)
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    user_intent = json.loads((run_dir / "UserIntent.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    summary = summarize_run(run_dir).as_manifest()

    mode = run_config.get("compile_source_mode")
    assert mode == args.expected_mode, f"RunConfig compile_source_mode mismatch: {mode!r}"
    assert p_vector.get("source_artifact_reality", {}).get("compile_source_mode") == args.expected_mode
    assert user_intent.get("compile_source_reality", {}).get("compile_source_mode") == args.expected_mode
    assert manifest.get("compile_source_mode") == args.expected_mode
    assert summary.get("compile_source_mode") == args.expected_mode

    if args.expected_mode == "fixture_only":
        assert run_config.get("source_reality_production_candidate") is False
        assert run_config.get("source_artifact_manifest_present") in {False, True}

    missing = resolve_compile_source_reality().as_manifest()
    assert missing["compile_source_mode"] == "fixture_only"
    assert missing["source_artifact_manifest_present"] is False

    print(
        "AUDIT PASSED: Slice 12B compile source-reality integration validated "
        f"mode={args.expected_mode} run={run_dir}"
    )


if __name__ == "__main__":
    main()
'''


WORKFLOW_COMPILE_SOURCE = r'''
"""Workflow helpers for Slice 12B compile source-reality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.source_artifacts.compile_policy import resolve_compile_source_reality


def run_compile_source_reality_plan(
    *,
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    return resolve_compile_source_reality(
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    ).as_manifest()
'''


def patch_compile_py() -> None:
    path = "src/pegasus/workflows/compile.py"
    old_sig = '''def run_compile(
    *,
    intent_path: str | Path,
    run_dir: str | Path | None = None,
    data_root: str | Path = "data",
) -> dict[str, Any]:
'''
    new_sig = '''def run_compile(
    *,
    intent_path: str | Path,
    run_dir: str | Path | None = None,
    data_root: str | Path = "data",
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
'''
    replace_once(path, old_sig, new_sig)

    old_anchor = '''    data_root = Path(data_root)
'''
    new_anchor = '''    data_root = Path(data_root)
    from pegasus.source_artifacts.compile_policy import (
        attach_compile_source_reality,
        resolve_compile_source_reality,
    )

    compile_source_reality = resolve_compile_source_reality(
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    )
'''
    replace_once(path, old_anchor, new_anchor)

    old_validation = '''    validation = validate_output_bundle(run_dir=str(run_dir))
'''
    new_validation = '''    attach_compile_source_reality(run_dir=run_dir, source_reality=compile_source_reality)
    validation = validate_output_bundle(run_dir=str(run_dir))
'''
    replace_once(path, old_validation, new_validation)


def patch_cli_py() -> None:
    path = "src/pegasus/cli.py"
    old_sig = '''def compile(
    intent: Path = typer.Option(..., "--intent"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
) -> None:
'''
    new_sig = '''def compile(
    intent: Path = typer.Option(..., "--intent"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    source_manifest: Path | None = typer.Option(None, "--source-manifest"),
    require_materialized_external: bool = typer.Option(False, "--require-materialized-external"),
) -> None:
'''
    replace_once(path, old_sig, new_sig)

    old_call = '''        result = run_compile(intent_path=intent, run_dir=run_dir)
'''
    new_call = '''        result = run_compile(
            intent_path=intent,
            run_dir=run_dir,
            source_manifest=source_manifest,
            require_materialized_external=require_materialized_external,
        )
'''
    replace_once(path, old_call, new_call)

    # Add an optional source-reality planning command under source-artifacts for quick policy inspection.
    text = read(path)
    if 'source_artifacts_compile_reality_plan' not in text:
        anchor = '''@source_artifacts_app.command("summary")
def source_artifacts_summary(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    from pegasus.workflows.source_artifacts import run_source_manifest_summary

    typer.echo(json.dumps(run_source_manifest_summary(manifest=manifest), indent=2, sort_keys=True))
'''
        addition = anchor + '''

@source_artifacts_app.command("compile-reality-plan")
def source_artifacts_compile_reality_plan(
    source_manifest: Path | None = typer.Option(None, "--source-manifest"),
    require_materialized_external: bool = typer.Option(False, "--require-materialized-external"),
) -> None:
    from pegasus.workflows.compile_source import run_compile_source_reality_plan

    typer.echo(json.dumps(
        run_compile_source_reality_plan(
            source_manifest=source_manifest,
            require_materialized_external=require_materialized_external,
        ),
        indent=2,
        sort_keys=True,
    ))
'''
        if anchor not in text:
            raise RuntimeError("Expected source-artifacts summary CLI anchor not found")
        (ROOT / path).write_text(text.replace(anchor, addition, 1), encoding="utf-8")


def patch_acceptance_contracts() -> None:
    path = "src/pegasus/acceptance/contracts.py"
    text = read(path)
    if "compile_source_mode: str | None" not in text:
        replace_once(
            path,
            '''    dashboard_safe_values: tuple[str, ...]
    telemetry_stage_status: dict[str, Any]
''',
            '''    dashboard_safe_values: tuple[str, ...]
    compile_source_mode: str | None
    source_artifact_manifest_present: bool
    source_reality_production_candidate: bool | None
    telemetry_stage_status: dict[str, Any]
''',
        )
    text = read(path)
    if '"compile_source_mode": self.compile_source_mode' not in text:
        replace_once(
            path,
            '''            "dashboard_safe_values": list(self.dashboard_safe_values),
            "telemetry_stage_status": self.telemetry_stage_status,
''',
            '''            "dashboard_safe_values": list(self.dashboard_safe_values),
            "compile_source_mode": self.compile_source_mode,
            "source_artifact_manifest_present": self.source_artifact_manifest_present,
            "source_reality_production_candidate": self.source_reality_production_candidate,
            "telemetry_stage_status": self.telemetry_stage_status,
''',
        )
    text = read(path)
    if "run_config = _load_json(root / \"RunConfig.json\")" not in text:
        replace_once(
            path,
            '''    manifest = _load_json(root / "ReproducibilityManifest.json")
    telemetry = manifest.get("telemetry", {}) if isinstance(manifest.get("telemetry", {}), dict) else {}
''',
            '''    manifest = _load_json(root / "ReproducibilityManifest.json")
    run_config = _load_json(root / "RunConfig.json")
    source_reality = run_config.get("source_artifact_reality")
    if not isinstance(source_reality, dict):
        source_reality = manifest.get("source_artifact_reality")
    if not isinstance(source_reality, dict):
        source_reality = {}
    compile_source_mode = (
        run_config.get("compile_source_mode")
        or manifest.get("compile_source_mode")
        or source_reality.get("compile_source_mode")
    )
    source_artifact_manifest_present = bool(
        run_config.get("source_artifact_manifest_present")
        if "source_artifact_manifest_present" in run_config
        else source_reality.get("source_artifact_manifest_present", False)
    )
    source_reality_production_candidate = run_config.get("source_reality_production_candidate")
    if source_reality_production_candidate is None:
        source_reality_production_candidate = source_reality.get("production_candidate")
    telemetry = manifest.get("telemetry", {}) if isinstance(manifest.get("telemetry", {}), dict) else {}
''',
        )
    text = read(path)
    if "compile_source_mode=compile_source_mode" not in text:
        replace_once(
            path,
            '''        dashboard_safe_values=dashboard_values,
        telemetry_stage_status=dict(stage_status),
''',
            '''        dashboard_safe_values=dashboard_values,
        compile_source_mode=compile_source_mode,
        source_artifact_manifest_present=source_artifact_manifest_present,
        source_reality_production_candidate=(
            bool(source_reality_production_candidate)
            if source_reality_production_candidate is not None
            else None
        ),
        telemetry_stage_status=dict(stage_status),
''',
        )


def patch_source_artifacts_init() -> None:
    path = "src/pegasus/source_artifacts/__init__.py"
    target = ROOT / path
    if not target.exists():
        return
    text = target.read_text(encoding="utf-8")
    if "CompileSourceReality" in text:
        return
    text = text.replace(
        "from pegasus.source_artifacts.contracts import (",
        "from pegasus.source_artifacts.compile_policy import (\n    CompileSourceReality,\n    CompileSourceRealityError,\n    attach_compile_source_reality,\n    resolve_compile_source_reality,\n)\nfrom pegasus.source_artifacts.contracts import (",
    )
    text = text.replace(
        "__all__ = [\n",
        "__all__ = [\n    \"CompileSourceReality\",\n    \"CompileSourceRealityError\",\n    \"attach_compile_source_reality\",\n    \"resolve_compile_source_reality\",\n",
    )
    target.write_text(text, encoding="utf-8")


def main() -> None:
    write("src/pegasus/source_artifacts/compile_policy.py", COMPILE_POLICY)
    write("src/pegasus/workflows/compile_source.py", WORKFLOW_COMPILE_SOURCE)
    write("tests/unit/test_compile_source_reality_policy.py", TEST_POLICY)
    write("tests/integration/test_slice12b_compile_source_reality_integration.py", TEST_INTEGRATION)
    write("scripts/dev/audits/audit_slice12b_compile_source_reality.py", AUDIT)

    patch_compile_py()
    patch_cli_py()
    patch_acceptance_contracts()
    patch_source_artifacts_init()

    print("Slice 12B applied: compile source-manifest reality policy integrated without SHE/EFG changes.")


if __name__ == "__main__":
    main()
