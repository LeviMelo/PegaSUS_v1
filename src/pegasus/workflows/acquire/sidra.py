from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.sidra.api import SidraClient
from pegasus.sidra.extract import extract_chunk_plan, read_chunk_plan, write_chunk_plan, write_extraction_log
from pegasus.sidra.metadata import (
    fetch_official_metadata,
    metadata_dir_hash,
    read_normalized_metadata_tables,
    table_ids_from_seed,
    write_normalized_metadata_tables,
)
from pegasus.sidra.plan import plan_sidra_chunks
from pegasus.sidra.registry import request_from_view


def sidra_runtime_config() -> dict[str, Any]:
    data = load_yaml("config/sidra.yaml")
    return data.get("sidra", data)


def run_sidra_metadata(
    *,
    tables: str | Path,
    level: str,
    output_dir: str | Path,
    raw_dir: str | Path,
) -> dict[str, Any]:
    table_ids = table_ids_from_seed(tables)
    if not table_ids:
        raise ValueError("No SIDRA table IDs found in seed.")

    client = SidraClient()
    metadata = fetch_official_metadata(
        table_ids=table_ids,
        client=client,
        locality_level=level,
        raw_dir=raw_dir,
    )
    outputs = write_normalized_metadata_tables(metadata, output_dir=output_dir)
    return {"table_ids": table_ids, "outputs": outputs}


def run_sidra_plan(
    *,
    view: str,
    metadata_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    sidra_cfg = sidra_runtime_config()
    metadata = read_normalized_metadata_tables(metadata_dir)
    request = request_from_view(view)
    chunks = plan_sidra_chunks(
        request,
        metadata,
        max_cells_per_request=int(sidra_cfg.get("max_cells_per_request", 49900)),
    )
    output_path = Path(output) if output is not None else Path("data/manifests/sidra") / f"{view}.json"
    write_chunk_plan(chunks, output_path=output_path)
    return {"chunks": chunks, "output": output_path}


def run_sidra_extract(
    *,
    plan: str | Path,
    metadata_dir: str | Path,
    dry_run: bool,
    concurrency: int | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    sidra_cfg = sidra_runtime_config()
    chunks = read_chunk_plan(plan)
    meta_hash = metadata_dir_hash(metadata_dir)

    if dry_run:
        return {
            "status": "dry_run",
            "chunks": chunks,
            "estimated_cells": sum(c.estimated_cells for c in chunks),
            "metadata_hash": meta_hash,
            "log_path": None,
        }

    client = SidraClient()
    results = extract_chunk_plan(
        chunks,
        client=client,
        concurrency=concurrency or int(sidra_cfg.get("concurrency", 4)),
        metadata_hash=meta_hash,
    )
    resolved_log_path = Path(log_path) if log_path is not None else Path("data/diagnostics/sidra") / f"{Path(plan).stem}.extraction_log.json"
    write_extraction_log(results, output_path=resolved_log_path)
    failures = [r for r in results if r.status != "success"]
    return {
        "status": "success" if not failures else "failed",
        "chunks": chunks,
        "results": results,
        "failures": failures,
        "metadata_hash": meta_hash,
        "log_path": resolved_log_path,
    }
