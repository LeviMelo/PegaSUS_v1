from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, axis_for_classification, map_category


class SIDRAProjectionBoundaryError(ValueError):
    """Raised when raw SIDRA facts cannot legally cross the SHE boundary."""


@dataclass(frozen=True)
class SIDRAProjectionBoundaryResult:
    path: str
    rows: int
    metadata: dict[str, Any]


def _pairs(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for item in parsed or []:
        if item is None or len(item) < 2:
            continue
        out.append((str(item[0]), str(item[1])))
    return out


def _category_map(raw: Any) -> dict[str, str]:
    return {classification: category for classification, category in _pairs(raw)}


def _measure_kind(unit: str | None) -> str:
    text = (unit or "").strip().casefold()
    if text in {"%", "percent", "percentual"}:
        return "percentage"
    if text in {"r$", "milr$", "sm"}:
        return "monetary"
    return "additive"


def _project_row(row: dict[str, Any]) -> dict[str, Any]:
    categories = _category_map(row.get("category_tuple"))
    projected: dict[str, str] = {}
    warnings: list[str] = []
    for classification_id, category_id in categories.items():
        axis = axis_for_classification(classification_id)
        if axis is None:
            continue
        canonical = map_category(axis, "SIDRA", category_id)
        if canonical == UNKNOWN:
            raise SIDRAProjectionBoundaryError(
                f"unmapped SIDRA category {category_id} for classification {classification_id} axis {axis}"
            )
        projected[axis] = canonical
        if canonical == TOTAL:
            warnings.append(f"sidra_{axis}_total_only_view")
    return {
        **row,
        "projection_axes_json": json.dumps(projected, sort_keys=True),
        "projection_warnings_json": json.dumps(sorted(set(warnings))),
    }


def project_and_bound_context_facts(
    facts_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> SIDRAProjectionBoundaryResult:
    """Project SIDRA classification tuples and enforce bounded EFG exposure.

    Context facts may cross into EFG only when every demographic classification category
    maps to a canonical axis, total categories remain total-only views, and percentage
    variables are not physically projected as ratios. Acquisition already requests total
    categories for non-demographic classifications; this boundary re-checks the actual
    facts so a malformed artifact cannot be admitted by labels.
    """
    facts_path = Path(facts_path)
    frame = pl.read_parquet(facts_path)
    if frame.height == 0:
        out_path = Path(output_path) if output_path is not None else facts_path.with_name(f"{facts_path.stem}.projected.parquet")
        frame.write_parquet(out_path)
        return SIDRAProjectionBoundaryResult(
            path=str(out_path),
            rows=0,
            metadata={
                "status": "projected_empty",
                "pushforward_status": "not_required",
                "projection_matrix_id": "sidra_context_total_only_v1",
                "warnings": ["sidra_context_empty"],
            },
        )

    rows: list[dict[str, Any]] = []
    blocked_ratio_projection: list[str] = []
    unmapped: list[str] = []
    mixed_total_non_total: list[str] = []
    category_values: dict[tuple[str, str, str], set[str]] = {}

    for row in frame.iter_rows(named=True):
        unit = None if row.get("unit") is None else str(row.get("unit"))
        measure_kind = _measure_kind(unit)
        categories = _category_map(row.get("category_tuple"))
        for classification_id, category_id in categories.items():
            axis = axis_for_classification(classification_id)
            key = (str(row.get("table_id")), str(row.get("variable_id")), classification_id)
            category_values.setdefault(key, set()).add(category_id)
            if axis is None:
                continue
            canonical = map_category(axis, "SIDRA", category_id)
            if canonical == UNKNOWN:
                unmapped.append(f"{classification_id}:{category_id}")
            if canonical != TOTAL and measure_kind == "percentage":
                blocked_ratio_projection.append(f"{row.get('table_id')}:{row.get('variable_id')}:{classification_id}:{category_id}")
        if unmapped or blocked_ratio_projection:
            continue
        rows.append(_project_row(row))

    for key, values in category_values.items():
        classification_id = key[2]
        axis = axis_for_classification(classification_id)
        if axis is None:
            continue
        canonical_values = {map_category(axis, "SIDRA", value) for value in values}
        if TOTAL in canonical_values and len(canonical_values) > 1:
            mixed_total_non_total.append(":".join(key))

    if unmapped:
        raise SIDRAProjectionBoundaryError(f"SIDRA context has unmapped demographic categories: {sorted(set(unmapped))[:10]}")
    if blocked_ratio_projection:
        raise SIDRAProjectionBoundaryError(
            "SIDRA percentage/rate context cannot be directly category-projected without "
            f"numerator/denominator recovery: {sorted(set(blocked_ratio_projection))[:10]}"
        )
    if mixed_total_non_total:
        raise SIDRAProjectionBoundaryError(
            f"SIDRA context mixes total and non-total demographic categories: {sorted(set(mixed_total_non_total))[:10]}"
        )

    out_path = Path(output_path) if output_path is not None else facts_path.with_name(f"{facts_path.stem}.projected.parquet")
    projected = pl.DataFrame(rows) if rows else frame.head(0)
    projected.write_parquet(out_path)
    metadata = {
        "status": "projected",
        "projection_matrix_id": "sidra_context_total_only_v1",
        "total_category_policy": "total_only_view",
        "pushforward_status": "not_required_total_only",
        "source_path": str(facts_path),
        "projected_path": str(out_path),
        "rows_in": int(frame.height),
        "rows_out": int(projected.height),
        "warnings": ["sidra_context_classification_projection_executed"],
    }
    return SIDRAProjectionBoundaryResult(path=str(out_path), rows=int(projected.height), metadata=metadata)


__all__ = [
    "SIDRAProjectionBoundaryError",
    "SIDRAProjectionBoundaryResult",
    "project_and_bound_context_facts",
]
