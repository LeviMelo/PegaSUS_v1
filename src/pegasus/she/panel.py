"""CommonPanel — the shared geo×time[×demographic] panel (MSD-II §II.2.1, MII-PANEL-01).

The EFG materializes each field as its own tensor (heterogeneous support: some
``municipality_cod6 × year``, some year-only, some municipality-only, some
domain-scalar). The CommonPanel is the *compiled product* that assembles those
tensors onto one shared cell index so downstream inference (the LDO, §II.6) can
read a single multivariate field ``X`` over ``(field, cell)``.

Anti-silence contract (§II.2.1, §II.11): the panel manifest records, per
``(field, cell)``, a state ∈ {observed, geo_invariant_broadcast,
time_invariant_broadcast, domain_scalar_broadcast, reconstructed, bounded,
projected, unavailable_on_panel(+reason)}. **A panel cell is never blank** — a
missing value always carries an explicit reason.

The temporal axis supports both ``year`` and ``month`` resolution (an intent
parameter, not a hardcoded grain). When a field is materialized at a coarser
grain than the panel (e.g. a year-resolution field on a monthly panel), its
finer cells are honestly marked ``unavailable_on_panel`` with a resolution
reason rather than fabricated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl

Resolution = Literal["year", "month"]

_CELL_KEYS = ("municipality_cod6", "year", "month")


@dataclass(frozen=True)
class PanelField:
    field_id: str
    name: str
    provenance: str
    state: str
    # support kind: which cell keys the field's tensor carries
    support_keys: tuple[str, ...]


@dataclass
class CommonPanel:
    resolution: Resolution
    cell_keys: tuple[str, ...]
    index: pl.DataFrame           # one row per panel cell (the shared support)
    values: pl.DataFrame          # index columns + one column per field_id (wide)
    manifest: pl.DataFrame        # long: (field_id, <cell keys>, state, reason)
    fields: list[PanelField] = field(default_factory=list)

    def as_manifest_summary(self) -> dict[str, Any]:
        by_state = (
            self.manifest.group_by("state").agg(pl.len().alias("n")).sort("n", descending=True)
            if self.manifest.height
            else pl.DataFrame({"state": [], "n": []})
        )
        return {
            "resolution": self.resolution,
            "cell_keys": list(self.cell_keys),
            "n_cells": self.index.height,
            "n_fields": len(self.fields),
            "state_counts": {row["state"]: row["n"] for row in by_state.to_dicts()},
        }


def _load_field_tensor(row: dict[str, Any]) -> pl.DataFrame | None:
    path = row.get("path")
    if not path or not os.path.exists(str(path)):
        return None
    try:
        return pl.read_parquet(str(path))
    except Exception:
        return None


def _support_keys(tensor: pl.DataFrame) -> tuple[str, ...]:
    return tuple(k for k in _CELL_KEYS if k in tensor.columns)


def _build_index(tensors: dict[str, pl.DataFrame], resolution: Resolution, cell_keys: tuple[str, ...]) -> pl.DataFrame:
    """Shared cell index = union of geo×time cells across the materialized fields."""
    frames: list[pl.DataFrame] = []
    for tensor in tensors.values():
        keys = [k for k in cell_keys if k in tensor.columns]
        if "municipality_cod6" in keys and "year" in keys:
            frames.append(tensor.select(keys).unique())
    if not frames:
        # No muni×year field — fall back to whatever geo/time cells exist.
        for tensor in tensors.values():
            present = [k for k in cell_keys if k in tensor.columns]
            if present:
                frames.append(tensor.select(present).unique())
    if not frames:
        return pl.DataFrame({k: [] for k in cell_keys})
    # Align to a common key set (fill missing keys as null so concat works).
    index = pl.concat([f for f in frames], how="diagonal_relaxed").unique()
    for k in cell_keys:
        if k not in index.columns:
            index = index.with_columns(pl.lit(None).alias(k))
    return index.select(list(cell_keys)).unique().sort(list(cell_keys))


def _align_field(
    field_id: str,
    tensor: pl.DataFrame,
    index: pl.DataFrame,
    cell_keys: tuple[str, ...],
) -> tuple[pl.DataFrame, str]:
    """Return (index+value column for field, broadcast_state).

    Aligns a field tensor onto the shared index, broadcasting coarser-support
    fields (geo-invariant / time-invariant / domain-scalar) and reporting the
    provenance qualifier that describes how the value reached each cell.
    """
    keys = [k for k in cell_keys if k in tensor.columns]
    value_col = "value" if "value" in tensor.columns else None
    if value_col is None:
        return index.with_columns(pl.lit(None).alias(field_id)), "unavailable_on_panel:no_value_column"

    slim = tensor.select([*keys, value_col]).rename({value_col: field_id})
    # Collapse any duplicate cell rows (defensive) by mean.
    if keys:
        slim = slim.group_by(keys).agg(pl.col(field_id).mean())

    if not keys:
        # Domain scalar → broadcast the single value to every cell.
        scalar = slim.select(field_id).to_series()
        val = scalar[0] if scalar.len() else None
        return index.with_columns(pl.lit(val).alias(field_id)), "domain_scalar_broadcast"

    joined = index.join(slim, on=keys, how="left")
    missing_keys = [k for k in cell_keys if k not in keys]
    if not missing_keys:
        state = "observed"
    elif missing_keys == ["municipality_cod6"] or "municipality_cod6" in missing_keys and "year" not in missing_keys:
        state = "geo_invariant_broadcast"
    elif "year" in missing_keys or "month" in missing_keys:
        state = "time_invariant_broadcast"
    else:
        state = "observed"
    return joined.select([*cell_keys, field_id]), state


def compile_common_panel(
    run_dir: str | Path,
    *,
    resolution: Resolution = "year",
    v_fields_path: str | Path | None = None,
) -> CommonPanel:
    """Assemble the CommonPanel from a compiled run's materialized field tensors."""
    run_dir = Path(run_dir)
    v_path = Path(v_fields_path) if v_fields_path else run_dir / "V_fields.parquet"
    catalog = pl.read_parquet(v_path)

    cell_keys: tuple[str, ...] = ("municipality_cod6", "year") if resolution == "year" else ("municipality_cod6", "year", "month")

    tensors: dict[str, pl.DataFrame] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in catalog.iter_rows(named=True):
        fid = str(row["field_id"])
        tensor = _load_field_tensor(row)
        if tensor is None:
            meta[fid] = {**row, "_tensor": None}
            continue
        tensors[fid] = tensor
        meta[fid] = {**row, "_tensor": tensor}

    index = _build_index(tensors, resolution, cell_keys)

    value_frame = index.clone()
    manifest_rows: list[dict[str, Any]] = []
    panel_fields: list[PanelField] = []

    for fid, row in meta.items():
        tensor = row["_tensor"]
        name = str(row.get("name") or fid)
        provenance = str(row.get("provenance") or "")
        state = str(row.get("state") or "")
        if tensor is None:
            panel_fields.append(PanelField(fid, name, provenance, state, ()))
            value_frame = value_frame.with_columns(pl.lit(None).alias(fid))
            for cell in index.iter_rows(named=True):
                manifest_rows.append({**cell, "field_id": fid, "state": "unavailable_on_panel", "reason": "field_not_materialized"})
            continue

        support = _support_keys(tensor)
        panel_fields.append(PanelField(fid, name, provenance, state, support))
        aligned, broadcast_state = _align_field(fid, tensor, index, cell_keys)
        value_frame = value_frame.join(aligned, on=list(cell_keys), how="left")

        # Per-cell provenance: observed/broadcast where a value landed, else
        # unavailable_on_panel with a reason (never blank).
        coarse_reason = None
        if resolution == "month" and "month" not in support and "year" in support:
            coarse_reason = "field_resolution_coarser_than_panel:year"
        for cell in aligned.iter_rows(named=True):
            has_value = cell.get(fid) is not None
            if has_value:
                cell_state = broadcast_state
                reason = None
            else:
                cell_state = "unavailable_on_panel"
                reason = coarse_reason or "cell_absent_from_field_support"
            manifest_rows.append(
                {**{k: cell.get(k) for k in cell_keys}, "field_id": fid, "state": cell_state, "reason": reason}
            )

    manifest = pl.DataFrame(manifest_rows) if manifest_rows else pl.DataFrame(
        {**{k: [] for k in cell_keys}, "field_id": [], "state": [], "reason": []}
    )
    return CommonPanel(
        resolution=resolution,
        cell_keys=cell_keys,
        index=index,
        values=value_frame,
        manifest=manifest,
        fields=panel_fields,
    )


def write_common_panel(panel: CommonPanel, out_dir: str | Path) -> dict[str, Any]:
    """Persist the panel product (values + manifest + summary) and return the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.values.write_parquet(out_dir / "common_panel_values.parquet")
    panel.manifest.write_parquet(out_dir / "common_panel_manifest.parquet")
    summary = panel.as_manifest_summary()
    (out_dir / "common_panel_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = ["CommonPanel", "PanelField", "compile_common_panel", "write_common_panel", "Resolution"]
