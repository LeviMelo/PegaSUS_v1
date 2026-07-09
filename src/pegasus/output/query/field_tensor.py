"""Shared field-tensor reader (P3c) — read a materialized EFG field's values from a run bundle.

Each EFG field materialized during compile is written to ``Tables/efg_tensors/{field_id}.parquet`` in
long form: the cell dimensions (e.g. ``year``, ``municipality_cod6``, and any strata) plus a ``value``
column and self-describing ``field_id``/``field_name``/``operator``. This module resolves a query's
``quantity`` to a ``field_id`` via the VariableDictionary and reads that tensor **lazily**
(``scan_parquet`` + predicate pushdown), so a national field tensor never has to fit in RAM.

Gotcha handled: the ``efg_execution_manifest.json`` records tensor paths inside the per-run
``__efg_stage_workspace`` — which the compile deletes after flush (T1.4). Those paths are stale; the
canonical copy lives under the bundle's own ``Tables/efg_tensors/``, keyed by the same ``field_id``.
This reader always goes through the bundle, never the manifest path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

# Units (VariableDictionary.unit) that denote an extensive, summable count — the only fields a
# ``count`` query may legitimately target (the denominator principle: you count counts, not rates).
COUNT_UNITS = frozenset({"counts", "count", "person_years", "person-years", "live_births", "livebirths"})

# Non-dimension columns present in every field tensor; everything else is a cell dimension.
_NON_DIM_COLS = frozenset({"value", "field_id", "field_name", "operator"})


class FieldResolutionError(ValueError):
    """Raised when a quantity cannot be resolved to exactly one materialized field."""


def efg_tensor_path(bundle_dir: str | Path, field_id: str) -> Path:
    return Path(bundle_dir) / "Tables" / "efg_tensors" / f"{field_id}.parquet"


def resolve_field(bundle_dir: str | Path, quantity: str) -> tuple[str, dict[str, Any]]:
    """Resolve ``quantity`` (a field_id, technical_name, or display_name) to (field_id, metadata row).

    Exact match, tried in that order of specificity. Raises on no match or an ambiguous one rather
    than silently picking a field — a query must be unambiguous about which quantity it means.
    """
    vd_path = Path(bundle_dir) / "VariableDictionary.parquet"
    if not vd_path.exists():
        raise FieldResolutionError(f"bundle has no VariableDictionary.parquet at {vd_path}")
    vd = pl.read_parquet(vd_path)
    for col in ("field_id", "technical_name", "display_name"):
        if col not in vd.columns:
            continue
        hit = vd.filter(pl.col(col) == quantity)
        if hit.height == 0:
            continue
        # Ambiguity is about DISTINCT field_ids: a real VariableDictionary can carry duplicate rows
        # per field_id, so match on the number of distinct fields, not the row count.
        distinct = hit["field_id"].unique().to_list()
        if len(distinct) == 1:
            return str(distinct[0]), hit.row(0, named=True)
        raise FieldResolutionError(
            f"quantity {quantity!r} is ambiguous: matches {len(distinct)} distinct fields on {col!r}. "
            "Query by field_id to disambiguate."
        )
    raise FieldResolutionError(
        f"quantity {quantity!r} not found in the bundle VariableDictionary (by field_id, "
        "technical_name, or display_name)."
    )


def resolve_field_by_carrier(bundle_dir: str | Path, carrier: str) -> tuple[str, dict[str, Any]]:
    """Resolve a field by its exact carrier (e.g. ``"HospitalAdmissions/Population"`` for an RN rate).

    Raises on no match or an ambiguous one (more than one distinct field_id carries it), so a rate
    selection is never silently the wrong field.
    """
    vd_path = Path(bundle_dir) / "VariableDictionary.parquet"
    if not vd_path.exists():
        raise FieldResolutionError(f"bundle has no VariableDictionary.parquet at {vd_path}")
    vd = pl.read_parquet(vd_path)
    if "carrier" not in vd.columns:
        raise FieldResolutionError("VariableDictionary has no 'carrier' column.")
    hit = vd.filter(pl.col("carrier") == carrier)
    if hit.height == 0:
        raise FieldResolutionError(f"no materialized field with carrier {carrier!r} in the bundle.")
    distinct = hit["field_id"].unique().to_list()
    if len(distinct) != 1:
        raise FieldResolutionError(
            f"carrier {carrier!r} is ambiguous: {len(distinct)} distinct fields. Query by field_id."
        )
    return str(distinct[0]), hit.row(0, named=True)


def cell_dimensions(bundle_dir: str | Path, field_id: str) -> list[str]:
    """The cell-dimension columns of a field tensor (everything but value/id/name/operator)."""
    path = efg_tensor_path(bundle_dir, field_id)
    if not path.exists():
        raise FieldResolutionError(f"field {field_id} has no materialized tensor at {path}")
    schema = pl.read_parquet_schema(str(path))
    return [c for c in schema if c not in _NON_DIM_COLS]


def read_field_tensor(
    bundle_dir: str | Path,
    field_id: str,
    *,
    filters: dict[str, Any] | None = None,
    value_alias: str | None = None,
) -> pl.DataFrame:
    """Read a materialized field's long-form tensor, lazily with predicate pushdown.

    ``filters`` maps a cell-dimension column to a value (or list). A filter on a column the tensor
    lacks is an error (never a silent no-op). ``value_alias`` renames the ``value`` column (e.g. to
    the numerator/denominator name for a rate join).
    """
    path = efg_tensor_path(bundle_dir, field_id)
    if not path.exists():
        raise FieldResolutionError(f"field {field_id} has no materialized tensor at {path}")
    lf = pl.scan_parquet(str(path))
    available = set(lf.collect_schema().names())
    for col, val in (filters or {}).items():
        if col not in available:
            raise FieldResolutionError(
                f"filter column {col!r} not in field tensor columns {sorted(available)}"
            )
        vals = list(val) if isinstance(val, (list, tuple, set)) else [val]
        lf = lf.filter(pl.col(col).is_in(vals))
    if value_alias:
        lf = lf.rename({"value": value_alias})
    return lf.collect()


def is_count_unit(unit: Any) -> bool:
    return str(unit or "").strip().lower() in COUNT_UNITS


__all__ = [
    "COUNT_UNITS",
    "FieldResolutionError",
    "efg_tensor_path",
    "resolve_field",
    "cell_dimensions",
    "read_field_tensor",
    "is_count_unit",
]
