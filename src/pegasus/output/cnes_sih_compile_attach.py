from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.core.hashing import sha256_file
from pegasus.output.cnes_sih_efg_bundle import (
    CNES_SIH_FIELD_PREFIXES,
    append_rows_like,
    build_cnes_sih_rows,
    write_cnes_sih_summary_tables,
)


def _field_prefix_filter(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        value = str(row.get(column) or "")
        if not any(value.startswith(prefix) for prefix in CNES_SIH_FIELD_PREFIXES):
            filtered.append(row)
    return filtered


def attach_cnes_sih_compile_fields(
    *,
    run_dir: str | Path,
    cnes_events_path: str | Path,
    sih_events_path: str | Path,
    municipality_cod6: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cnes_events_path = Path(cnes_events_path)
    sih_events_path = Path(sih_events_path)
    built = build_cnes_sih_rows(cnes_events_path=cnes_events_path, sih_events_path=sih_events_path, municipality_cod6=municipality_cod6)

    append_rows_like(run_dir / "V_fields.parquet", built.fields, remove_field_prefixes=CNES_SIH_FIELD_PREFIXES, id_column="field_id")
    append_rows_like(run_dir / "Q_tensor.parquet", built.q_rows, remove_field_prefixes=CNES_SIH_FIELD_PREFIXES, id_column="field_id")
    append_rows_like(run_dir / "VariableDictionary.parquet", built.vd_rows, remove_field_prefixes=CNES_SIH_FIELD_PREFIXES, id_column="field_id")
    append_rows_like(run_dir / "E_DAG.parquet", built.edges, remove_field_prefixes=("edge_cnes_", "edge_sih_"), id_column="edge_id")
    append_rows_like(run_dir / "Warnings.parquet", built.warnings, remove_field_prefixes=("slice5a_",), id_column="warning_id")
    append_rows_like(run_dir / "FailedBranches.parquet", built.failed_branches, remove_field_prefixes=("failed_cnes_", "failed_facility_", "failed_sih_"), id_column="failed_branch_id")

    summary_paths = write_cnes_sih_summary_tables(run_dir, built)
    metadata = {
        **built.metadata,
        "source_hashes": {
            "cnes_events": sha256_file(cnes_events_path),
            "sih_events": sha256_file(sih_events_path),
            "cnes_capacity_summary": sha256_file(summary_paths["cnes_capacity_summary"]),
            "sih_cost_summary": sha256_file(summary_paths["sih_cost_summary"]),
        },
        "rows_written": {
            "V_fields": len(built.fields),
            "Q_tensor": len(built.q_rows),
            "VariableDictionary": len(built.vd_rows),
            "FailedBranches": len(built.failed_branches),
        },
    }
    return {"run_dir": run_dir, "cnes_sih": metadata}
