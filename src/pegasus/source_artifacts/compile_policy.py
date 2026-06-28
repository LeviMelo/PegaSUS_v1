"""Compile-time source-reality policy for manifest-backed PegaSUS runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pegasus.core.hashing import sha256_file
from pegasus.source_artifacts.contracts import source_manifest_summary, validate_source_artifact_manifest


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
    if source_manifest is None:
        raise CompileSourceRealityError(
            "production compile requires an explicit materialized_external source artifact manifest"
        )

    manifest_path = Path(source_manifest)
    validation = validate_source_artifact_manifest(
        manifest_path,
        require_materialized_external=require_materialized_external,
    )
    if not validation["ok"]:
        raise CompileSourceRealityError("; ".join(validation["errors"]))

    summary = source_manifest_summary(manifest_path)
    mode = str(summary["compile_source_mode"])
    if mode != "materialized_external":
        raise CompileSourceRealityError(
            f"compile requires materialized_external source artifacts; got {mode}"
        )

    return CompileSourceReality(
        schema_version="2.0",
        compile_source_mode=mode,
        source_artifact_manifest_present=True,
        source_artifact_manifest_path=str(manifest_path),
        source_artifact_manifest_hash=sha256_file(manifest_path),
        source_artifact_count=int(summary["source_artifact_count"]),
        source_systems=tuple(summary["source_systems"]),
        artifact_roles=tuple(summary["artifact_roles"]),
        require_materialized_external=True,
        production_candidate=True,
        validation_warnings=tuple(summary.get("warnings", []) or []),
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def attach_compile_source_reality(*, run_dir: str | Path, source_reality: CompileSourceReality) -> None:
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


__all__ = [
    "CompileSourceReality",
    "CompileSourceRealityError",
    "resolve_compile_source_reality",
    "attach_compile_source_reality",
]
