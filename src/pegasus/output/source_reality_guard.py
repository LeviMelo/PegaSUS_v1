"""Materialized-external source-reality semantic guard.

This module is intentionally production-facing. It is not a fixture helper
and it must not repair output rows by string substitution. Its only job is
to detect semantic contamination in completed run bundles.
"""

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
)

# Phrases that are allowed only as architectural proof that legacy/demo
# paths are quarantined, not as field, warning, denominator, or source
# semantics. Keep this list narrow.
ARCHITECTURE_PROOF_ALLOWLIST: tuple[str, ...] = (
    "quarantined_fixture_only",
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
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).lower()
    for allowed in ARCHITECTURE_PROOF_ALLOWLIST:
        text = text.replace(allowed.lower(), "")
    return text


def _source_mode_from_json(root: Path) -> str | None:
    for name in ("RunConfig.json", "ReproducibilityManifest.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        source_reality = payload.get("source_artifact_reality")
        if not isinstance(source_reality, dict):
            source_reality = {}
        mode = (
            payload.get("compile_source_mode")
            or source_reality.get("compile_source_mode")
        )
        if mode:
            return str(mode)
    return None


def _contains_forbidden(text: str) -> str | None:
    lowered = text.lower()
    for term in FORBIDDEN_MATERIALIZED_EXTERNAL_TERMS:
        if term in lowered:
            return term
    return None


def _table_rows_from_pyarrow_table(table: Any) -> list[dict[str, Any]]:
    try:
        return table.to_pylist()
    except Exception:
        return []


def _scan_table_rows(*, table_name: str, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows):
        text = _json_text(row)
        term = _contains_forbidden(text)
        if term is not None:
            errors.append(
                f"materialized_external forbidden semantic term {term!r} "
                f"in {table_name} row {idx}"
            )
    return errors


def _scan_json_payload(*, surface: str, payload: Any) -> list[str]:
    text = _json_text(payload)
    term = _contains_forbidden(text)
    if term is None:
        return []
    return [f"materialized_external forbidden semantic term {term!r} in {surface}"]


def materialized_external_semantic_errors(
    *,
    root: str | Path,
    tables: Mapping[str, Any] | None = None,
    include_json: bool = True,
) -> list[str]:
    root = Path(root)
    source_mode = _source_mode_from_json(root)
    if source_mode != "materialized_external":
        return []

    errors: list[str] = []

    if tables is not None:
        for table_name, table in tables.items():
            errors.extend(_scan_table_rows(table_name=table_name, rows=_table_rows_from_pyarrow_table(table)))
    else:
        for rel in PARQUET_SURFACES:
            path = root / rel
            if not path.exists():
                continue
            try:
                rows = pq.read_table(path).to_pylist()
            except Exception as exc:
                errors.append(f"could not inspect materialized_external table {rel}: {type(exc).__name__}: {exc}")
                continue
            errors.extend(_scan_table_rows(table_name=rel, rows=rows))

    if include_json:
        for rel in JSON_SURFACES:
            path = root / rel
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"could not inspect materialized_external JSON {rel}: {type(exc).__name__}: {exc}")
                continue
            errors.extend(_scan_json_payload(surface=rel, payload=payload))

        tables_dir = root / "Tables"
        if tables_dir.exists():
            for path in sorted(tables_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"could not inspect materialized_external JSON {path.relative_to(root)}: {type(exc).__name__}: {exc}")
                    continue
                errors.extend(_scan_json_payload(surface=str(path.relative_to(root)).replace("\\", "/"), payload=payload))

        maps_dir = root / "Maps"
        if maps_dir.exists():
            for path in sorted(maps_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"could not inspect materialized_external JSON {path.relative_to(root)}: {type(exc).__name__}: {exc}")
                    continue
                errors.extend(_scan_json_payload(surface=str(path.relative_to(root)).replace("\\", "/"), payload=payload))

    return errors


def assert_no_materialized_external_fixture_semantics(*, root: str | Path) -> None:
    errors = materialized_external_semantic_errors(root=root, include_json=True)
    if errors:
        raise RuntimeError("materialized_external source-reality contamination: " + "; ".join(errors))


__all__ = [
    "FORBIDDEN_MATERIALIZED_EXTERNAL_TERMS",
    "materialized_external_semantic_errors",
    "assert_no_materialized_external_fixture_semantics",
]
