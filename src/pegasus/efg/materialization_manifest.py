
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
