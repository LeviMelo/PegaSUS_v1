"""Hard materialized-external source-reality guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq


FORBIDDEN_MATERIALIZED_EXTERNAL_TERMS: tuple[str, ...] = (
    "fixture",
    "synthetic",
    "placeholder",
    "explicit_smoke",
    "smoke_municipality",
    "legacy_computed_fields_preserved\": true",
    "legacy_graph_authority\": true",
)

PARQUET_SURFACES: tuple[str, ...] = (
    "V_fields.parquet",
    "E_DAG.parquet",
    "Q_tensor.parquet",
    "Warnings.parquet",
    "ModelAssociations.parquet",
    "ResidualAssociations.parquet",
    "Hypotheses.parquet",
    "VariableDictionary.parquet",
    "FailedBranches.parquet",
    "QuarantinedFields.parquet",
    "ForcedFields.parquet",
)

JSON_SURFACES: tuple[str, ...] = (
    "UserIntent.json",
    "RunConfig.json",
    "ReproducibilityManifest.json",
    "P_vector.json",
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).lower()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_mode(root: Path) -> str | None:
    for rel in ("RunConfig.json", "ReproducibilityManifest.json"):
        payload = _load_json(root / rel)
        source_reality = payload.get("source_artifact_reality")
        if not isinstance(source_reality, dict):
            source_reality = {}
        mode = payload.get("compile_source_mode") or source_reality.get("compile_source_mode")
        if mode:
            return str(mode)
    return None


def _contains_forbidden(text: str) -> str | None:
    lowered = text.lower()
    for term in FORBIDDEN_MATERIALIZED_EXTERNAL_TERMS:
        if term in lowered:
            return term
    return None


def _rows(table: Any) -> list[dict[str, Any]]:
    try:
        return table.to_pylist()
    except Exception:
        return []


def materialized_external_semantic_errors(
    *,
    root: str | Path,
    tables: Mapping[str, Any] | None = None,
    include_json: bool = True,
) -> list[str]:
    root = Path(root)
    if _source_mode(root) != "materialized_external":
        return []

    errors: list[str] = []

    if tables is not None:
        for table_name, table in tables.items():
            for idx, row in enumerate(_rows(table)):
                term = _contains_forbidden(_json_text(row))
                if term is not None:
                    errors.append(f"materialized_external forbidden semantic term {term!r} in {table_name} row {idx}")
    else:
        for rel in PARQUET_SURFACES:
            path = root / rel
            if not path.exists():
                continue
            try:
                for idx, row in enumerate(pq.read_table(path).to_pylist()):
                    term = _contains_forbidden(_json_text(row))
                    if term is not None:
                        errors.append(f"materialized_external forbidden semantic term {term!r} in {rel} row {idx}")
            except Exception as exc:
                errors.append(f"could not inspect materialized_external table {rel}: {type(exc).__name__}: {exc}")

    if include_json:
        for rel in JSON_SURFACES:
            path = root / rel
            if not path.exists():
                continue
            term = _contains_forbidden(_json_text(_load_json(path)))
            if term is not None:
                errors.append(f"materialized_external forbidden semantic term {term!r} in {rel}")

        for dirname in ("Tables", "Maps"):
            folder = root / dirname
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.json")):
                term = _contains_forbidden(_json_text(_load_json(path)))
                if term is not None:
                    errors.append(f"materialized_external forbidden semantic term {term!r} in {path.relative_to(root).as_posix()}")

    return errors


def assert_materialized_external_source_purity(*, root: str | Path) -> None:
    errors = materialized_external_semantic_errors(root=root, include_json=True)
    if errors:
        raise RuntimeError("materialized_external source-reality contamination: " + "; ".join(errors))
