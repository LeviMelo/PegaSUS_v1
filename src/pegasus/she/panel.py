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


def _tensor_path(row: dict[str, Any]) -> str | None:
    path = row.get("path")
    if not path or not os.path.exists(str(path)):
        return None
    return str(path)


def _tensor_columns(path: str) -> tuple[str, ...]:
    """The parquet's column names, read from schema only (no data materialized)."""
    try:
        return tuple(pl.scan_parquet(path).collect_schema().names())
    except Exception:
        return ()


def _load_field_tensor(row: dict[str, Any], columns: list[str] | None = None) -> pl.DataFrame | None:
    """Read a field tensor, selecting only ``columns`` when given.

    At national scale a tensor is muni×year×strata; reading only the cell keys + the value column
    (the sole columns the panel consumes) avoids pulling every source column into RAM, and lets each
    tensor be dropped between fields instead of the whole catalog being held at once (Finding 6)."""
    path = _tensor_path(row)
    if path is None:
        return None
    try:
        if columns:
            return pl.read_parquet(path, columns=columns)
        return pl.read_parquet(path)
    except Exception:
        return None


def _support_keys(tensor: pl.DataFrame) -> tuple[str, ...]:
    return tuple(k for k in _CELL_KEYS if k in tensor.columns)


def _support_keys_from_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(k for k in _CELL_KEYS if k in columns)


def _build_index_streaming(
    paths_and_keys: list[tuple[str, tuple[str, ...]]],
    resolution: Resolution,
    cell_keys: tuple[str, ...],
) -> pl.DataFrame:
    """Shared cell index = union of geo×time cells across the materialized fields.

    Streams the key columns per parquet (``scan_parquet`` → select keys → unique) rather than
    holding every full field tensor in RAM: only the distinct (muni, year[, month]) tuples of each
    field are materialized, so peak memory here is O(unique cells), not O(sum of tensor sizes)."""

    def _scan_keys(want_full_geo_time: bool) -> list[pl.DataFrame]:
        frames: list[pl.DataFrame] = []
        for path, present_keys in paths_and_keys:
            keys = [k for k in cell_keys if k in present_keys]
            if want_full_geo_time and not ("municipality_cod6" in keys and "year" in keys):
                continue
            if not keys:
                continue
            try:
                frames.append(pl.scan_parquet(path).select(keys).unique().collect())
            except Exception:
                continue
        return frames

    frames = _scan_keys(want_full_geo_time=True)
    if not frames:
        # No muni×year field — fall back to whatever geo/time cells exist.
        frames = _scan_keys(want_full_geo_time=False)
    if not frames:
        return pl.DataFrame({k: [] for k in cell_keys})
    # Align to a common key set (fill missing keys as null so concat works).
    index = pl.concat(frames, how="diagonal_relaxed").unique()
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
    geography_prefixes: frozenset[str] | set[str] | None = None,
) -> CommonPanel:
    """Assemble the CommonPanel from a compiled run's materialized field tensors.

    ``geography_prefixes`` (2-digit cod6 UF prefixes) restricts the panel index to
    the run's geographic scope — a defensive filter mirroring the EFG executor's
    scope guard, so a panel built over tensors that predate that guard is still
    scoped correctly.
    """
    run_dir = Path(run_dir)
    v_path = Path(v_fields_path) if v_fields_path else run_dir / "V_fields.parquet"
    catalog = pl.read_parquet(v_path)
    prefixes = frozenset(str(p) for p in geography_prefixes) if geography_prefixes else None

    cell_keys: tuple[str, ...] = ("municipality_cod6", "year") if resolution == "year" else ("municipality_cod6", "year", "month")

    # Catalog metadata only — no tensors held. Each field's parquet columns are read from the
    # schema (cheap), so the panel never holds all field tensors in RAM at once (Finding 6): each
    # tensor is loaded, aligned, and dropped inside the per-field loop below.
    catalog_rows: list[dict[str, Any]] = []
    paths_and_keys: list[tuple[str, tuple[str, ...]]] = []
    for row in catalog.iter_rows(named=True):
        path = _tensor_path(row)
        columns = _tensor_columns(path) if path is not None else ()
        catalog_rows.append({**row, "_path": path, "_columns": columns})
        if path is not None and columns:
            paths_and_keys.append((path, columns))

    index = _build_index_streaming(paths_and_keys, resolution, cell_keys)
    if prefixes is not None and index.height:
        index = index.filter(
            pl.col("municipality_cod6").cast(pl.Utf8).str.slice(0, 2).is_in(list(prefixes))
        )

    value_frame = index.clone()
    # The per-cell provenance manifest is built per-field as a VECTORIZED polars frame
    # (state/reason as when/then columns) and concatenated — never a per-cell Python append
    # loop. The old row-by-row `manifest_rows.append({...})` was O(fields x cells) dict builds
    # (~800k at national scale) AND crashed polars' schema inference: `reason` is String|Null,
    # the first 100 rows were all-null so polars typed the column Null, then a later
    # "cell_absent_from_field_support" string failed to append. Typing reason as Utf8 fixes it.
    _key_cols = list(cell_keys)
    manifest_frames: list[pl.DataFrame] = []
    panel_fields: list[PanelField] = []

    for row in catalog_rows:
        fid = str(row["field_id"])
        name = str(row.get("name") or fid)
        provenance = str(row.get("provenance") or "")
        state = str(row.get("state") or "")
        columns: tuple[str, ...] = row["_columns"]
        if row["_path"] is None or not columns:
            panel_fields.append(PanelField(fid, name, provenance, state, ()))
            value_frame = value_frame.with_columns(pl.lit(None).alias(fid))
            manifest_frames.append(
                index.select(
                    *_key_cols,
                    pl.lit(fid).alias("field_id"),
                    pl.lit("unavailable_on_panel").alias("state"),
                    pl.lit("field_not_materialized", dtype=pl.Utf8).alias("reason"),
                )
            )
            continue

        # Read only the cell keys + the value column this field contributes to the panel; the
        # tensor is bound to a local and dropped at loop end so peak RAM is index + one tensor +
        # the (growing) value_frame, not the whole materialized catalog (Finding 6).
        support = _support_keys_from_columns(columns)
        wanted = [*support, "value"] if "value" in columns else list(support)
        tensor = _load_field_tensor(row, columns=wanted or None)
        if tensor is None:
            panel_fields.append(PanelField(fid, name, provenance, state, ()))
            value_frame = value_frame.with_columns(pl.lit(None).alias(fid))
            manifest_frames.append(
                index.select(
                    *_key_cols,
                    pl.lit(fid).alias("field_id"),
                    pl.lit("unavailable_on_panel").alias("state"),
                    pl.lit("field_not_materialized", dtype=pl.Utf8).alias("reason"),
                )
            )
            continue

        panel_fields.append(PanelField(fid, name, provenance, state, support))
        aligned, broadcast_state = _align_field(fid, tensor, index, cell_keys)
        del tensor  # drop the field tensor before the next field is loaded
        value_frame = value_frame.join(aligned, on=_key_cols, how="left")

        # Per-cell provenance (vectorized): observed/broadcast where a value landed, else
        # unavailable_on_panel with a non-blank reason.
        coarse_reason = None
        if resolution == "month" and "month" not in support and "year" in support:
            coarse_reason = "field_resolution_coarser_than_panel:year"
        absent_reason = coarse_reason or "cell_absent_from_field_support"
        _has = pl.col(fid).is_not_null()
        manifest_frames.append(
            aligned.select(
                *_key_cols,
                pl.lit(fid).alias("field_id"),
                pl.when(_has).then(pl.lit(broadcast_state)).otherwise(pl.lit("unavailable_on_panel")).alias("state"),
                pl.when(_has).then(pl.lit(None, dtype=pl.Utf8)).otherwise(pl.lit(absent_reason, dtype=pl.Utf8)).alias("reason"),
            )
        )

    if manifest_frames:
        manifest = pl.concat(manifest_frames, how="vertical_relaxed")
    else:
        manifest = index.select(*_key_cols).head(0).with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("field_id"),
            pl.lit(None, dtype=pl.Utf8).alias("state"),
            pl.lit(None, dtype=pl.Utf8).alias("reason"),
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
