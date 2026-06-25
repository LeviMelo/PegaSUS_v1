"""Substrate Harmonization Engine boundary for source artifacts.

Slice 13A introduces the first real SHE boundary. It consumes local source
artifacts, applies source-field registry semantics and zero-variance/all-missing
exclusion, and emits a typed SubstrateBundle. It does not build the EFG, compute
rates, fetch data, run PIRS/HSIC, or materialize first-class output keys.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.she.source_registry import SourceFieldSpec, resolve_source_field, resolve_source_fields, source_system_from_path
from pegasus.she.zero_variance import ColumnVarianceProfile, TableVarianceProfile, profile_table_variance


class SubstrateError(ValueError):
    """Raised when the SHE substrate boundary cannot evaluate an artifact."""


@dataclass(frozen=True)
class SourceArtifactRef:
    path: str
    source_system: str
    artifact_role: str = "processed_events"
    provenance_mode: str = "development"
    source_manifest_hash: str | None = None
    artifact_hash: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubstrateFieldCandidate:
    candidate_id: str
    source_system: str
    artifact_path: str
    column: str
    technical_name: str
    carrier: str
    unit: str
    aggregation: str
    role: tuple[str, ...]
    axes: dict[str, Any]
    provenance: tuple[str, ...]
    substrate_kind: str
    registry_hash: str
    row_count: int
    non_null_count: int
    unique_non_null_count: int
    missing_rate: float | None
    numeric_min: float | None
    numeric_max: float | None
    source_manifest_hash: str | None
    artifact_hash: str | None
    warnings: tuple[str, ...]

    @property
    def column_name(self) -> str:
        return self.column

    @property
    def quality_role(self) -> str | None:
        try:
            return resolve_source_field(source_system=self.source_system, column_name=self.column).spec.quality_role
        except Exception:
            return None

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role"] = list(self.role)
        payload["provenance"] = list(self.provenance)
        payload["warnings"] = list(self.warnings)
        payload["column_name"] = self.column_name
        payload["quality_role"] = self.quality_role
        return payload


@dataclass(frozen=True)
class SubstrateFieldExclusion:
    exclusion_id: str
    source_system: str
    artifact_path: str
    column: str
    reason: str
    registry_reason: str | None
    row_count: int
    non_null_count: int
    unique_non_null_count: int
    missing_rate: float | None
    structural_role: str
    warnings: tuple[str, ...]

    @property
    def column_name(self) -> str:
        return self.column

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        payload["column_name"] = self.column_name
        return payload


@dataclass(frozen=True)
class SubstrateBundle:
    schema_version: str
    substrate_id: str
    source_reality_mode: str
    source_artifacts: tuple[SourceArtifactRef, ...]
    candidates: tuple[SubstrateFieldCandidate, ...]
    exclusions: tuple[SubstrateFieldExclusion, ...]
    table_profiles: tuple[TableVarianceProfile, ...]
    registry_hashes: dict[str, str]
    warnings: tuple[str, ...]

    @property
    def admissible_candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def excluded_field_count(self) -> int:
        return len(self.exclusions)

    @property
    def zero_variance_exclusion_count(self) -> int:
        return sum(1 for e in self.exclusions if e.reason == "zero_variance_constant")

    @property
    def all_missing_exclusion_count(self) -> int:
        return sum(1 for e in self.exclusions if e.reason == "all_missing")

    @property
    def structural_exclusion_count(self) -> int:
        return sum(1 for e in self.exclusions if e.reason == "structural_or_audit_only")

    @property
    def candidate_fields(self) -> tuple[SubstrateFieldCandidate, ...]:
        return self.candidates

    @property
    def excluded_fields(self) -> tuple[SubstrateFieldExclusion, ...]:
        return self.exclusions

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "substrate_id": self.substrate_id,
            "source_reality_mode": self.source_reality_mode,
            "source_artifact_count": len(self.source_artifacts),
            "candidate_count": len(self.candidates),
            "excluded_count": len(self.exclusions),
            "admissible_candidate_count": self.admissible_candidate_count,
            "excluded_field_count": self.excluded_field_count,
            "zero_variance_exclusion_count": self.zero_variance_exclusion_count,
            "all_missing_exclusion_count": self.all_missing_exclusion_count,
            "structural_exclusion_count": self.structural_exclusion_count,
            "registry_backed": bool(self.registry_hashes) and all(bool(v) for v in self.registry_hashes.values()),
            "registry_hashes": dict(self.registry_hashes),
            "warnings": list(self.warnings),
        }

    def as_manifest(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "source_artifacts": [a.as_manifest() for a in self.source_artifacts],
            "candidates": [c.as_manifest() for c in self.candidates],
            "exclusions": [e.as_manifest() for e in self.exclusions],
            "table_profiles": [p.as_manifest() for p in self.table_profiles],
        }


def _artifact_hash(path: str | Path) -> str | None:
    p = Path(path)
    return sha256_file(p) if p.exists() and p.is_file() else None


def _stable_id(payload: dict[str, Any], prefix: str) -> str:
    return f"{prefix}_{content_hash(payload)[:20]}"


def normalize_source_artifact_ref(payload: SourceArtifactRef | dict[str, Any] | str | Path) -> SourceArtifactRef:
    if isinstance(payload, SourceArtifactRef):
        return payload
    if isinstance(payload, (str, Path)):
        path = str(payload)
        source_system = source_system_from_path(path) or "UNKNOWN"
        return SourceArtifactRef(
            path=path,
            source_system=source_system,
            artifact_role="processed_events",
            provenance_mode="development" if "development" in path.lower() else "cached_external",
            artifact_hash=_artifact_hash(path),
        )
    if not isinstance(payload, dict):
        raise SubstrateError(f"Invalid source artifact reference: {payload!r}")
    path = payload.get("path") or payload.get("artifact_path") or payload.get("processed_path")
    if not path:
        raise SubstrateError(f"Source artifact reference has no path: {payload}")
    source_system = payload.get("source_system") or payload.get("system") or source_system_from_path(path) or "UNKNOWN"
    role = payload.get("artifact_role") or payload.get("role") or payload.get("source_role") or "processed_events"
    mode = payload.get("provenance_mode") or payload.get("mode") or payload.get("source_mode") or ("development" if "development" in str(path).lower() else "cached_external")
    source_manifest_hash = payload.get("source_manifest_hash") or payload.get("manifest_hash")
    artifact_hash = payload.get("artifact_hash") or payload.get("content_hash") or payload.get("sha256") or _artifact_hash(path)
    return SourceArtifactRef(
        path=str(path),
        source_system=str(source_system),
        artifact_role=str(role),
        provenance_mode=str(mode),
        source_manifest_hash=None if source_manifest_hash is None else str(source_manifest_hash),
        artifact_hash=None if artifact_hash is None else str(artifact_hash),
    )


def load_source_artifacts_from_manifest(path: str | Path) -> tuple[SourceArtifactRef, ...]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw_artifacts = payload.get("artifacts") or payload.get("source_artifacts") or []
    if isinstance(raw_artifacts, dict):
        raw_artifacts = list(raw_artifacts.values())
    if not isinstance(raw_artifacts, list):
        raise SubstrateError(f"Source artifact manifest does not contain a list of artifacts: {p}")
    manifest_hash = sha256_file(p)
    refs: list[SourceArtifactRef] = []
    for item in raw_artifacts:
        if isinstance(item, dict) and not item.get("source_manifest_hash"):
            item = {**item, "source_manifest_hash": manifest_hash}
        refs.append(normalize_source_artifact_ref(item))
    return tuple(refs)


def source_reality_mode(artifacts: Iterable[SourceArtifactRef]) -> str:
    modes = {a.provenance_mode for a in artifacts}
    if not modes:
        return "unresolved_external"
    if modes == {"materialized_external"}:
        return "materialized_external"
    if modes <= {"cached_external", "materialized_external"}:
        return "cached_external"
    if modes == {"development"}:
        return "unresolved_external"
    return "mixed_development_external"


def _candidate_from_profile(
    *,
    artifact: SourceArtifactRef,
    spec: SourceFieldSpec,
    registry_hash: str,
    profile: ColumnVarianceProfile,
) -> SubstrateFieldCandidate:
    payload = {
        "source_system": artifact.source_system,
        "path": artifact.path,
        "column": profile.column,
        "registry_hash": registry_hash,
        "artifact_hash": artifact.artifact_hash,
    }
    warnings = tuple(sorted(set(profile.warnings)))
    return SubstrateFieldCandidate(
        candidate_id=_stable_id(payload, "substrate_candidate"),
        source_system=artifact.source_system,
        artifact_path=artifact.path,
        column=profile.column,
        technical_name=spec.technical_name,
        carrier=spec.carrier,
        unit=spec.unit,
        aggregation=spec.aggregation,
        role=spec.role,
        axes=spec.axes,
        provenance=tuple(dict.fromkeys((*spec.provenance, artifact.provenance_mode))),
        substrate_kind=spec.substrate_kind,
        registry_hash=registry_hash,
        row_count=profile.row_count,
        non_null_count=profile.non_null_count,
        unique_non_null_count=profile.unique_non_null_count,
        missing_rate=profile.missing_rate,
        numeric_min=profile.numeric_min,
        numeric_max=profile.numeric_max,
        source_manifest_hash=artifact.source_manifest_hash,
        artifact_hash=artifact.artifact_hash,
        warnings=warnings,
    )


def _exclusion_from_profile(
    *,
    artifact: SourceArtifactRef,
    spec: SourceFieldSpec,
    profile: ColumnVarianceProfile,
    reason: str,
) -> SubstrateFieldExclusion:
    payload = {
        "source_system": artifact.source_system,
        "path": artifact.path,
        "column": profile.column,
        "reason": reason,
        "artifact_hash": artifact.artifact_hash,
    }
    warnings = list(profile.warnings)
    if spec.registry_reason:
        warnings.append(f"registry_{spec.registry_reason}")
    return SubstrateFieldExclusion(
        exclusion_id=_stable_id(payload, "substrate_exclusion"),
        source_system=artifact.source_system,
        artifact_path=artifact.path,
        column=profile.column,
        reason=reason,
        registry_reason=spec.registry_reason,
        row_count=profile.row_count,
        non_null_count=profile.non_null_count,
        unique_non_null_count=profile.unique_non_null_count,
        missing_rate=profile.missing_rate,
        structural_role=profile.structural_role,
        warnings=tuple(sorted(set(warnings))),
    )


def build_substrate_bundle(
    *,
    artifacts: Iterable[SourceArtifactRef | dict[str, Any] | str | Path],
    allow_heuristic_registry: bool = True,
) -> SubstrateBundle:
    refs = tuple(normalize_source_artifact_ref(a) for a in artifacts)
    candidates: list[SubstrateFieldCandidate] = []
    exclusions: list[SubstrateFieldExclusion] = []
    profiles: list[TableVarianceProfile] = []
    registry_hashes: dict[str, str] = {}
    warnings: list[str] = []

    if not refs:
        warnings.append("substrate_no_source_artifacts")

    for artifact in refs:
        path = Path(artifact.path)
        if not path.exists():
            exclusions.append(SubstrateFieldExclusion(
                exclusion_id=_stable_id({"path": artifact.path, "reason": "missing_artifact"}, "substrate_exclusion"),
                source_system=artifact.source_system,
                artifact_path=artifact.path,
                column="*",
                reason="missing_artifact",
                registry_reason=None,
                row_count=0,
                non_null_count=0,
                unique_non_null_count=0,
                missing_rate=None,
                structural_role="missing_artifact",
                warnings=("substrate_artifact_missing",),
            ))
            warnings.append(f"missing_artifact:{artifact.path}")
            continue
        table_profile = profile_table_variance(path)
        profiles.append(table_profile)
        resolution = resolve_source_fields(
            source_system=artifact.source_system,
            columns=[p.column for p in table_profile.profiles],
            allow_heuristic=allow_heuristic_registry,
        )
        registry_hashes[artifact.source_system] = resolution.registry_hash
        spec_by_column = {s.column: s for s in resolution.specs}
        for profile in table_profile.profiles:
            spec = spec_by_column.get(profile.column)
            if spec is None:
                exclusions.append(_exclusion_from_profile(
                    artifact=artifact,
                    spec=SourceFieldSpec(
                        source_system=artifact.source_system,
                        column=profile.column,
                        technical_name=f"{artifact.source_system}.{profile.column}",
                        carrier="unknown",
                        unit="unknown",
                        aggregation="non_aggregable",
                        role=("unresolved",),
                        axes={},
                        provenance=("unresolved_registry",),
                        substrate_kind="unresolved",
                        admissible_by_registry=False,
                        registry_reason="unresolved_source_field",
                    ),
                    profile=profile,
                    reason="unresolved_source_field",
                ))
                continue
            reason = profile.exclusion_reason or spec.registry_reason
            if profile.admissible and spec.admissible_by_registry:
                candidates.append(_candidate_from_profile(
                    artifact=artifact,
                    spec=spec,
                    registry_hash=resolution.registry_hash,
                    profile=profile,
                ))
            else:
                exclusions.append(_exclusion_from_profile(
                    artifact=artifact,
                    spec=spec,
                    profile=profile,
                    reason=reason or "registry_excluded",
                ))

    substrate_payload = {
        "source_reality_mode": source_reality_mode(refs),
        "artifacts": [a.as_manifest() for a in refs],
        "candidate_ids": [c.candidate_id for c in candidates],
        "exclusion_ids": [e.exclusion_id for e in exclusions],
        "registry_hashes": registry_hashes,
    }
    return SubstrateBundle(
        schema_version="1.0",
        substrate_id=_stable_id(substrate_payload, "substrate_bundle"),
        source_reality_mode=source_reality_mode(refs),
        source_artifacts=refs,
        candidates=tuple(candidates),
        exclusions=tuple(exclusions),
        table_profiles=tuple(profiles),
        registry_hashes=registry_hashes,
        warnings=tuple(sorted(set(warnings))),
    )


def write_substrate_bundle_manifest(bundle: SubstrateBundle, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle.as_manifest(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_substrate_bundle_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SubstrateError(f"Substrate manifest is not a JSON object: {path}")
    return payload


def attach_substrate_summary_to_run(*, run_dir: str | Path, bundle: SubstrateBundle) -> dict[str, Any]:
    root = Path(run_dir)
    tables_dir = root / "Tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_substrate_bundle_manifest(bundle, tables_dir / "substrate_manifest.json")
    summary = {
        **bundle.summary(),
        "manifest_path": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "status": "evaluated",
    }
    for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
        except Exception:
            continue
        payload["substrate_gate"] = summary
        if name == "ReproducibilityManifest.json":
            payload.setdefault("registry_hashes", {}).update(bundle.registry_hashes)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
