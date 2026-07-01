from __future__ import annotations

from pathlib import Path
import zipfile

ROOT = Path.cwd()

FILES: dict[str, str] = {}

FILES['src/pegasus/source_artifacts/__init__.py'] = '''"""Source artifact reality contracts for PegaSUS data-layer hardening."""

from pegasus.source_artifacts.contracts import (
    SourceArtifact,
    SourceArtifactError,
    inspect_source_artifact,
    load_source_artifact_manifest,
    source_manifest_summary,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)

__all__ = [
    "SourceArtifact",
    "SourceArtifactError",
    "inspect_source_artifact",
    "load_source_artifact_manifest",
    "source_manifest_summary",
    "validate_source_artifact_manifest",
    "write_source_artifact_manifest",
]
'''

FILES['src/pegasus/source_artifacts/contracts.py'] = '''"""Source artifact reality contracts for DATASUS/SIDRA compiler inputs.

Slice 12A intentionally does not perform live acquisition. It defines the hard
boundary that lets the compiler and audits distinguish fixtures from materialized
source artifacts before compile integration is widened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json

import polars as pl

SOURCE_ARTIFACT_SCHEMA_VERSION = "1.0"

ALLOWED_SOURCE_SYSTEMS = {
    "SIM-DO",
    "SINASC",
    "CNES-ST",
    "SIH-RD",
    "SIDRA",
    "IBGE-SIDRA",
}

ALLOWED_ARTIFACT_ROLES = {
    "raw_payload",
    "raw_table",
    "processed_events",
    "normalized_facts",
    "request_manifest",
    "metadata_table",
    "extraction_log",
}

ALLOWED_PROVENANCE_MODES = {
    "fixture",
    "cached_external",
    "materialized_external",
}


class SourceArtifactError(ValueError):
    """Raised when source artifacts cannot satisfy data-layer reality contracts."""


@dataclass(frozen=True)
class SourceArtifact:
    source_system: str
    artifact_role: str
    path: str
    content_hash: str
    byte_size: int
    row_count: int | None
    columns: list[str]
    provenance_mode: str
    source_manifest_hash: str | None
    manifest_path: str | None
    warnings: list[str]
    inspected_at: str

    def as_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_shape(path: Path) -> tuple[int | None, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(path)
        return df.height, list(df.columns)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        df = pl.read_csv(path, separator=separator)
        return df.height, list(df.columns)
    if suffix in {".json", ".jsonl"}:
        return None, []
    return None, []


def _as_required_set(values: Iterable[str] | None) -> set[str]:
    return {str(v) for v in values or [] if str(v)}


def inspect_source_artifact(
    *,
    path: str | Path,
    source_system: str,
    artifact_role: str,
    provenance_mode: str,
    source_manifest_hash: str | None = None,
    manifest_path: str | Path | None = None,
    required_columns: Iterable[str] | None = None,
) -> SourceArtifact:
    artifact_path = Path(path)
    if source_system not in ALLOWED_SOURCE_SYSTEMS:
        raise SourceArtifactError(f"Unsupported source system: {source_system!r}")
    if artifact_role not in ALLOWED_ARTIFACT_ROLES:
        raise SourceArtifactError(f"Unsupported artifact role: {artifact_role!r}")
    if provenance_mode not in ALLOWED_PROVENANCE_MODES:
        raise SourceArtifactError(f"Unsupported provenance mode: {provenance_mode!r}")
    if not artifact_path.exists():
        raise SourceArtifactError(f"Source artifact path does not exist: {artifact_path}")
    if artifact_path.is_dir():
        raise SourceArtifactError(f"Source artifact path is a directory, not a file: {artifact_path}")

    row_count, columns = _table_shape(artifact_path)
    required = _as_required_set(required_columns)
    missing_columns = sorted(required.difference(columns))
    if missing_columns:
        raise SourceArtifactError(
            f"Source artifact {artifact_path} is missing required columns: {missing_columns}"
        )

    warnings: list[str] = []
    if provenance_mode == "fixture":
        warnings.append("fixture_source_artifact_not_production_candidate")
    if provenance_mode != "fixture" and not source_manifest_hash:
        raise SourceArtifactError(
            "Non-fixture source artifacts must carry source_manifest_hash for compile provenance."
        )
    if artifact_role in {"processed_events", "normalized_facts"} and row_count == 0:
        warnings.append("empty_processed_source_artifact")

    return SourceArtifact(
        source_system=source_system,
        artifact_role=artifact_role,
        path=str(artifact_path),
        content_hash=_sha256_file(artifact_path),
        byte_size=artifact_path.stat().st_size,
        row_count=row_count,
        columns=columns,
        provenance_mode=provenance_mode,
        source_manifest_hash=source_manifest_hash,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        warnings=warnings,
        inspected_at=_now(),
    )


def _compile_source_mode(artifacts: list[SourceArtifact]) -> str:
    modes = {artifact.provenance_mode for artifact in artifacts}
    if not modes:
        return "empty"
    if modes == {"fixture"}:
        return "fixture_only"
    if "fixture" in modes:
        return "mixed_fixture_external"
    return "materialized_external"


def write_source_artifact_manifest(
    *,
    artifacts: list[SourceArtifact],
    output_path: str | Path,
    manifest_id: str = "source_artifact_manifest",
    purpose: str = "compile_source_reality_gate",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "purpose": purpose,
        "created_at": _now(),
        "compile_source_mode": _compile_source_mode(artifacts),
        "artifact_count": len(artifacts),
        "artifacts": [artifact.as_manifest() for artifact in artifacts],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return out


def load_source_artifact_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_source_artifact_manifest(
    *,
    manifest_path: str | Path,
    require_materialized_external: bool = False,
    required_roles: Iterable[str] | None = None,
    required_systems: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest = load_source_artifact_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    artifacts = list(manifest.get("artifacts") or [])

    if manifest.get("schema_version") != SOURCE_ARTIFACT_SCHEMA_VERSION:
        errors.append("source artifact manifest schema_version mismatch")

    seen_roles = {str(item.get("artifact_role")) for item in artifacts}
    seen_systems = {str(item.get("source_system")) for item in artifacts}

    for role in _as_required_set(required_roles):
        if role not in seen_roles:
            errors.append(f"missing required artifact_role: {role}")
    for system in _as_required_set(required_systems):
        if system not in seen_systems:
            errors.append(f"missing required source_system: {system}")

    for item in artifacts:
        path = Path(str(item.get("path", "")))
        if not path.exists():
            errors.append(f"artifact path missing: {path}")
            continue
        actual_hash = _sha256_file(path)
        if actual_hash != item.get("content_hash"):
            errors.append(f"artifact hash mismatch: {path}")
        if item.get("provenance_mode") == "fixture":
            warnings.append(f"fixture artifact present: {item.get('source_system')}:{item.get('artifact_role')}")
        if item.get("provenance_mode") != "fixture" and not item.get("source_manifest_hash"):
            errors.append(f"non-fixture artifact missing source_manifest_hash: {path}")

    if require_materialized_external and manifest.get("compile_source_mode") != "materialized_external":
        errors.append(
            "source artifact manifest is not production-candidate materialized_external mode"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "manifest_path": str(manifest_path),
        "compile_source_mode": manifest.get("compile_source_mode"),
        "artifact_count": len(artifacts),
        "source_systems": sorted(seen_systems),
        "artifact_roles": sorted(seen_roles),
    }


def source_manifest_summary(*, manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_source_artifact_manifest(manifest_path)
    artifacts = list(manifest.get("artifacts") or [])
    return {
        "manifest_path": str(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "compile_source_mode": manifest.get("compile_source_mode"),
        "artifact_count": len(artifacts),
        "artifacts": [
            {
                "source_system": item.get("source_system"),
                "artifact_role": item.get("artifact_role"),
                "provenance_mode": item.get("provenance_mode"),
                "row_count": item.get("row_count"),
                "byte_size": item.get("byte_size"),
                "path": item.get("path"),
            }
            for item in artifacts
        ],
    }
'''

FILES['src/pegasus/source_artifacts/resolver.py'] = '''"""Source artifact resolver helpers for manifest-driven compiler inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pegasus.source_artifacts.contracts import (
    SourceArtifact,
    inspect_source_artifact,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)


def build_manifest_from_paths(
    *,
    output_path: str | Path,
    entries: Iterable[dict],
    manifest_id: str = "source_artifact_manifest",
) -> Path:
    artifacts: list[SourceArtifact] = []
    for entry in entries:
        artifacts.append(
            inspect_source_artifact(
                path=entry["path"],
                source_system=entry["source_system"],
                artifact_role=entry["artifact_role"],
                provenance_mode=entry.get("provenance_mode", "fixture"),
                source_manifest_hash=entry.get("source_manifest_hash"),
                manifest_path=entry.get("manifest_path"),
                required_columns=entry.get("required_columns") or [],
            )
        )
    return write_source_artifact_manifest(
        artifacts=artifacts,
        output_path=output_path,
        manifest_id=manifest_id,
    )


def compile_reality_gate(
    *,
    manifest_path: str | Path,
    require_materialized_external: bool = False,
) -> dict:
    return validate_source_artifact_manifest(
        manifest_path=manifest_path,
        require_materialized_external=require_materialized_external,
        required_roles=["processed_events", "normalized_facts"],
    )
'''

FILES['src/pegasus/workflows/source_artifacts.py'] = '''"""Workflow wrappers for Slice 12A source artifact reality gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.source_artifacts.contracts import (
    inspect_source_artifact,
    source_manifest_summary,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)


def run_source_artifact_inspect(
    *,
    path: str | Path,
    source_system: str,
    artifact_role: str,
    provenance_mode: str,
    output: str | Path | None = None,
    source_manifest_hash: str | None = None,
) -> dict[str, Any]:
    artifact = inspect_source_artifact(
        path=path,
        source_system=source_system,
        artifact_role=artifact_role,
        provenance_mode=provenance_mode,
        source_manifest_hash=source_manifest_hash,
    )
    if output is not None:
        write_source_artifact_manifest(artifacts=[artifact], output_path=output)
    return artifact.as_manifest()


def run_source_manifest_validate(
    *,
    manifest: str | Path,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    return validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=require_materialized_external,
    )


def run_source_manifest_summary(*, manifest: str | Path) -> dict[str, Any]:
    return source_manifest_summary(manifest_path=manifest)
'''

FILES['tests/unit/test_source_artifact_contract.py'] = '''from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pegasus.source_artifacts.contracts import (
    SourceArtifactError,
    inspect_source_artifact,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)


def test_source_artifact_inspection_marks_fixture_not_production_candidate(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "municipality_cod6": ["270430"], "year": [2022]}).write_parquet(path)

    artifact = inspect_source_artifact(
        path=path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
        required_columns=["event_id", "year"],
    )

    assert artifact.row_count == 1
    assert "event_id" in artifact.columns
    assert artifact.provenance_mode == "fixture"
    assert "fixture_source_artifact_not_production_candidate" in artifact.warnings


def test_non_fixture_artifacts_require_source_manifest_hash(tmp_path: Path) -> None:
    path = tmp_path / "sidra_facts.parquet"
    pl.DataFrame({"table_id": ["9606"], "value_numeric": [1.0]}).write_parquet(path)

    with pytest.raises(SourceArtifactError, match="source_manifest_hash"):
        inspect_source_artifact(
            path=path,
            source_system="SIDRA",
            artifact_role="normalized_facts",
            provenance_mode="materialized_external",
        )


def test_manifest_validation_blocks_fixture_when_external_required(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2022]}).write_parquet(path)
    artifact = inspect_source_artifact(
        path=path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
    )
    manifest = tmp_path / "source_manifest.json"
    write_source_artifact_manifest(artifacts=[artifact], output_path=manifest)

    relaxed = validate_source_artifact_manifest(manifest_path=manifest)
    strict = validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=True,
    )

    assert relaxed["ok"] is True
    assert strict["ok"] is False
    assert any("materialized_external" in error for error in strict["errors"])


def test_materialized_external_manifest_is_production_candidate(tmp_path: Path) -> None:
    path = tmp_path / "sidra_facts.parquet"
    pl.DataFrame({"table_id": ["9606"], "value_numeric": [1.0]}).write_parquet(path)
    artifact = inspect_source_artifact(
        path=path,
        source_system="SIDRA",
        artifact_role="normalized_facts",
        provenance_mode="materialized_external",
        source_manifest_hash="sidra_request_manifest_hash",
    )
    manifest = tmp_path / "source_manifest.json"
    write_source_artifact_manifest(artifacts=[artifact], output_path=manifest)

    result = validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=True,
        required_roles=["normalized_facts"],
        required_systems=["SIDRA"],
    )

    assert result["ok"] is True
    assert result["compile_source_mode"] == "materialized_external"
'''

FILES['tests/integration/test_slice12a_source_artifact_integration.py'] = '''from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.source_artifacts.contracts import (
    inspect_source_artifact,
    source_manifest_summary,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)
from pegasus.workflows.acquire.source_artifacts import (
    run_source_artifact_inspect,
    run_source_manifest_summary,
    run_source_manifest_validate,
)


def test_slice12a_source_manifest_distinguishes_fixture_from_external(tmp_path: Path) -> None:
    sim_path = tmp_path / "sim_events.parquet"
    sidra_path = tmp_path / "sidra_facts.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2022]}).write_parquet(sim_path)
    pl.DataFrame({"table_id": ["9606"], "value_numeric": [123.0]}).write_parquet(sidra_path)

    artifacts = [
        inspect_source_artifact(
            path=sim_path,
            source_system="SIM-DO",
            artifact_role="processed_events",
            provenance_mode="fixture",
        ),
        inspect_source_artifact(
            path=sidra_path,
            source_system="SIDRA",
            artifact_role="normalized_facts",
            provenance_mode="materialized_external",
            source_manifest_hash="sidra_manifest_hash",
        ),
    ]
    manifest = tmp_path / "source_artifacts.json"
    write_source_artifact_manifest(artifacts=artifacts, output_path=manifest)

    result = validate_source_artifact_manifest(manifest_path=manifest)
    strict = validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=True,
    )
    summary = source_manifest_summary(manifest_path=manifest)

    assert result["ok"] is True
    assert result["compile_source_mode"] == "mixed_fixture_external"
    assert strict["ok"] is False
    assert summary["artifact_count"] == 2


def test_slice12a_workflow_wrappers_are_json_ready(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2022]}).write_parquet(path)
    manifest = tmp_path / "manifest.json"

    inspected = run_source_artifact_inspect(
        path=path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
        output=manifest,
    )
    validation = run_source_manifest_validate(manifest=manifest)
    summary = run_source_manifest_summary(manifest=manifest)

    assert inspected["source_system"] == "SIM-DO"
    assert validation["ok"] is True
    assert summary["compile_source_mode"] == "fixture_only"
'''

FILES['scripts/dev/audits/audit_slice12a_source_artifacts.py'] = '''from __future__ import annotations

import argparse
import sys

from pegasus.source_artifacts.contracts import (
    source_manifest_summary,
    validate_source_artifact_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Slice 12A source artifact manifest reality gates.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-materialized-external", action="store_true")
    args = parser.parse_args()

    validation = validate_source_artifact_manifest(
        manifest_path=args.manifest,
        require_materialized_external=args.require_materialized_external,
    )
    if not validation["ok"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}")
        return 1

    summary = source_manifest_summary(manifest_path=args.manifest)
    if summary["artifact_count"] < 1:
        print("ERROR: source artifact manifest is empty")
        return 1

    print(
        "AUDIT PASSED: Slice 12A source artifact manifest reality gate validated "
        f"mode={summary['compile_source_mode']} artifacts={summary['artifact_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_file(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _patch_cli() -> None:
    path = ROOT / 'src/pegasus/cli.py'
    text = path.read_text(encoding='utf-8')
    if 'source_artifacts_app' not in text:
        if 'acceptance_app = typer.Typer(no_args_is_help=True)' in text:
            text = text.replace(
                'acceptance_app = typer.Typer(no_args_is_help=True)\n',
                'acceptance_app = typer.Typer(no_args_is_help=True)\nsource_artifacts_app = typer.Typer(no_args_is_help=True)\n',
            )
            text = text.replace(
                'app.add_typer(acceptance_app, name="acceptance")\n',
                'app.add_typer(acceptance_app, name="acceptance")\napp.add_typer(source_artifacts_app, name="source-artifacts")\n',
            )
        else:
            text = text.replace(
                'pirs_app = typer.Typer(no_args_is_help=True)\n',
                'pirs_app = typer.Typer(no_args_is_help=True)\nsource_artifacts_app = typer.Typer(no_args_is_help=True)\n',
            )
            text = text.replace(
                'app.add_typer(pirs_app, name="pirs")\n',
                'app.add_typer(pirs_app, name="pirs")\napp.add_typer(source_artifacts_app, name="source-artifacts")\n',
            )
    marker = '# Slice 12A source artifact reality gate commands'
    if marker not in text:
        text = text.rstrip() + '''


# Slice 12A source artifact reality gate commands
@source_artifacts_app.command("inspect")
def source_artifacts_inspect(
    path: Path = typer.Option(..., "--path"),
    source_system: str = typer.Option(..., "--source-system"),
    artifact_role: str = typer.Option(..., "--role"),
    provenance_mode: str = typer.Option("fixture", "--provenance-mode"),
    output: Path | None = typer.Option(None, "--output"),
    source_manifest_hash: str | None = typer.Option(None, "--source-manifest-hash"),
) -> None:
    from pegasus.workflows.acquire.source_artifacts import run_source_artifact_inspect

    result = run_source_artifact_inspect(
        path=path,
        source_system=source_system,
        artifact_role=artifact_role,
        provenance_mode=provenance_mode,
        output=output,
        source_manifest_hash=source_manifest_hash,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@source_artifacts_app.command("validate-manifest")
def source_artifacts_validate_manifest(
    manifest: Path = typer.Option(..., "--manifest"),
    require_materialized_external: bool = typer.Option(False, "--require-materialized-external"),
) -> None:
    from pegasus.workflows.acquire.source_artifacts import run_source_manifest_validate

    result = run_source_manifest_validate(
        manifest=manifest,
        require_materialized_external=require_materialized_external,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise typer.Exit(1)


@source_artifacts_app.command("summary")
def source_artifacts_summary(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    from pegasus.workflows.acquire.source_artifacts import run_source_manifest_summary

    typer.echo(json.dumps(run_source_manifest_summary(manifest=manifest), indent=2, sort_keys=True))
'''
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def main() -> None:
    for rel, content in FILES.items():
        _write_file(rel, content)
    _patch_cli()
    print('Slice 12A updater applied: source artifact reality gate, workflows, CLI, tests, and audit added.')


if __name__ == '__main__':
    main()
