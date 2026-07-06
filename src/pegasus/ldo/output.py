"""Serialize LDO LinkRecords to the Hypotheses bundle key (MSD-II §II.8, MII-OUT-01).

The Hypotheses key carries typed link records (the enriched replacement for the
pairwise hypothesis row, MSD §8.3). List-valued fields are JSON-encoded so the
row is flat-parquet writable while remaining round-trippable.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.ldo.records import LINK_RECORD_COLUMNS, LinkRecord

_LIST_FIELDS = {"confounding_factor_refs", "warnings"}


def link_records_to_table(records: list[LinkRecord]) -> pl.DataFrame:
    """Flat table of link records with JSON-encoded list columns."""
    rows: list[dict] = []
    for record in records:
        row = record.as_row()
        for key in _LIST_FIELDS:
            row[key] = json.dumps(row.get(key) or [], ensure_ascii=False)
        rows.append(row)
    if not rows:
        return pl.DataFrame({col: [] for col in LINK_RECORD_COLUMNS})
    return pl.DataFrame(rows).select([c for c in LINK_RECORD_COLUMNS])


def write_hypotheses(records: list[LinkRecord], path: str | Path) -> dict[str, int]:
    """Write link records to a Hypotheses parquet; return a small summary."""
    table = link_records_to_table(records)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return {
        "n_records": table.height,
        "n_selected": sum(1 for r in records if r.certification_status == "selected"),
        "n_lagged": sum(1 for r in records if r.edge_type == "lagged_directed"),
        "n_latent_shared": sum(1 for r in records if r.edge_type == "latent_shared"),
        "n_nonlinear": sum(1 for r in records if r.edge_type == "nonlinear_residual"),
    }


__all__ = ["link_records_to_table", "write_hypotheses"]
