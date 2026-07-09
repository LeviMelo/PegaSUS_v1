"""Query engine — materialize a derived dataset from an emitted run bundle (FEAT-P3).

Additive reader over the bundle; never mutates first-class tables. P3b implements the ``edge``
query (the LDO's certified hypotheses, whose typed LinkRecord fields already carry their own
provenance). ``rate``/``standardized_rate``/``count``/``raw_field`` share a field-tensor reader
built in P3c/P3d. See PEGASUS_OUTPUT_QUERY_LAYER.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.output.query.denominators import resolve_denominator
from pegasus.output.query.spec import QuerySpec
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES


class QueryError(ValueError):
    """Raised when a query cannot be posed or served against a bundle."""


@dataclass(frozen=True)
class MaterializedDataset:
    frame: pl.DataFrame
    provenance: dict[str, Any]
    kind: str
    warnings: tuple[str, ...] = ()


def _bundle_file(bundle_dir: str | Path, key: str) -> Path:
    return Path(bundle_dir) / OUTPUT_BUNDLE_FILES[key]


def _run_identity(bundle_dir: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {"bundle_dir": str(bundle_dir)}
    repro = _bundle_file(bundle_dir, "ReproducibilityManifest")
    if repro.exists():
        try:
            data = json.loads(repro.read_text(encoding="utf-8"))
            for k in ("run_id", "manifest_hash", "intent_hash", "source_manifest_hash"):
                if isinstance(data, dict) and k in data:
                    out[k] = data[k]
        except Exception:
            pass
    return out


def _apply_filters(frame: pl.DataFrame, filters: dict[str, Any]) -> pl.DataFrame:
    for col, val in (filters or {}).items():
        if col not in frame.columns:
            raise QueryError(f"filter column {col!r} not in dataset columns {frame.columns}")
        values = list(val) if isinstance(val, (list, tuple, set)) else [val]
        frame = frame.filter(pl.col(col).is_in(values))
    return frame


# Per-variable metadata joined onto each edge endpoint (the self-describing-hypotheses bridge).
# Epidemiologically-relevant, user-facing columns only — never compute-lifecycle internals.
_ENRICH_COLS = ("name", "carrier", "unit", "diagnostic_role", "icd_group_id", "icd_group_kind")


def _variable_metadata(bundle_dir: str | Path) -> pl.DataFrame | None:
    """The EFG per-variable dictionary keyed by ``field_id`` (== the LDO variable id). Prefer
    ``VariableDictionary`` (richest), fall back to ``V_fields``. None if neither is present."""
    for key in ("VariableDictionary", "V_fields"):
        path = _bundle_file(bundle_dir, key)
        if path.exists():
            frame = pl.read_parquet(path)
            if "field_id" in frame.columns:
                keep = ["field_id"] + [c for c in _ENRICH_COLS if c in frame.columns]
                return frame.select(keep).unique(subset=["field_id"])
    return None


def _enrich_edges(frame: pl.DataFrame, bundle_dir: str | Path) -> tuple[pl.DataFrame, list[str]]:
    """Join variable metadata onto ``source_var`` and ``target_var`` so each edge is self-describing.
    Pure export-layer join over the EFG dictionary — the LDO estimator stays decoupled from labels."""
    meta = _variable_metadata(bundle_dir)
    if meta is None:
        return frame, ["variable_metadata_unavailable_edges_unenriched"]
    for endpoint in ("source", "target"):
        var_col = f"{endpoint}_var"
        if var_col not in frame.columns:
            continue
        rename = {"field_id": var_col}
        rename.update({c: f"{endpoint}_{c}" for c in meta.columns if c != "field_id"})
        frame = frame.join(meta.rename(rename), on=var_col, how="left")
    return frame, []


def _edge_query(bundle_dir: str | Path, spec: QuerySpec) -> MaterializedDataset:
    path = _bundle_file(bundle_dir, "Hypotheses")
    if not path.exists():
        raise QueryError(f"bundle has no Hypotheses.parquet at {path}")
    frame = _apply_filters(pl.read_parquet(path), spec.filters)
    warnings: list[str] = []
    if spec.enrich:
        frame, warnings = _enrich_edges(frame, bundle_dir)
    provenance = {
        "kind": "edge",
        "quantity": spec.quantity,
        "query": spec.model_dump(mode="json"),
        "run": _run_identity(bundle_dir),
        "n_rows": frame.height,
        "enriched": bool(spec.enrich and not warnings),
        "note": (
            "Edge provenance is carried per row: uncertainty, stability, fdr_qvalue, code_system, "
            "projection_status, overlap_jaccard, certification_status, causal_rung/assumptions, warnings. "
            "With enrich=True, source_/target_ variable metadata (name/carrier/unit/ICD-group/topology) "
            "is joined from the EFG VariableDictionary so the table is self-describing."
        ),
    }
    return MaterializedDataset(frame=frame, provenance=provenance, kind="edge", warnings=tuple(warnings))


def materialize_query(bundle_dir: str | Path, spec: QuerySpec) -> MaterializedDataset:
    """Serve ``spec`` from the bundle at ``bundle_dir``. P3b: the ``edge`` kind."""
    if spec.kind == "edge":
        return _edge_query(bundle_dir, spec)
    if spec.requires_denominator():
        # Fail fast + informatively: the denominator DOES resolve (FEAT-P4 substrate is live),
        # the rate/standardized-rate math + field-tensor reader land in P3c/P3d.
        opt = resolve_denominator(spec.quantity, spec)
        raise QueryError(
            f"kind={spec.kind!r}: denominator resolves to {opt.id!r} (FEAT-P4 live), but the "
            f"rate/standardized-rate materialization is P3c/P3d — not yet wired."
        )
    raise QueryError(
        f"kind={spec.kind!r} (count/raw_field) needs the shared field-tensor reader (P3c) — not yet wired."
    )


__all__ = ["QueryError", "MaterializedDataset", "materialize_query"]
