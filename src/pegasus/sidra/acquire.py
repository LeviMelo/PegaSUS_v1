"""Ruthless, cell-budget-aware SIDRA acquisition orchestrator (MSD national-scale).

End-to-end: a registry-trusted ``SIDRARequest`` is split into cell-budgeted chunks
(``plan_sidra_chunks_unchecked``), fetched with wide concurrency
(``SidraClient.fetch_chunks_parallel``), normalized to canonical fact rows, and written
to a single facts parquet. SIDRA throttles by cells-per-request, not request rate, so the
strategy is: fat chunks just under the cell cap + many concurrent requests + no
inter-request sleep (only per-request backoff on transient errors).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash
from pegasus.sidra.api import SidraClient
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.sidra.plan import plan_sidra_chunks_unchecked
from pegasus.sidra.schemas import SIDRARequest


@dataclass(frozen=True)
class AcquisitionReport:
    table_id: str
    chunk_count: int
    estimated_cells: int
    fact_rows: int
    seconds: float
    output_path: str | None
    failed_chunks: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.failed_chunks

    @property
    def cells_per_second(self) -> float:
        return round(self.estimated_cells / self.seconds, 1) if self.seconds > 0 else 0.0

    def as_manifest(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "chunk_count": self.chunk_count,
            "estimated_cells": self.estimated_cells,
            "fact_rows": self.fact_rows,
            "seconds": round(self.seconds, 3),
            "cells_per_second": self.cells_per_second,
            "output_path": self.output_path,
            "failed_chunk_count": len(self.failed_chunks),
            "failed_chunks": list(self.failed_chunks),
            "ok": self.ok,
        }


def acquire_table_facts(
    request: SIDRARequest,
    *,
    output_path: str | Path,
    client: SidraClient | None = None,
    max_cells_per_request: int = 95000,
    max_workers: int = 12,
    unit_by_variable: dict[str, str | None] | None = None,
    metadata_hash: str = "sidra_compendium",
    view: str | None = None,
) -> AcquisitionReport:
    client = client or SidraClient()
    chunks = plan_sidra_chunks_unchecked(request, max_cells_per_request=max_cells_per_request)
    estimated_cells = sum(chunk.estimated_cells for chunk in chunks)

    started = time.time()
    responses = client.fetch_chunks_parallel(chunks, max_workers=max_workers, view=view)
    elapsed = time.time() - started

    facts = []
    failed: list[dict[str, Any]] = []
    for chunk in chunks:
        response = responses.get(chunk.chunk_id)
        if response is None or response.status_code >= 400:
            failed.append({
                "chunk_id": chunk.chunk_id,
                "status_code": None if response is None else response.status_code,
                "estimated_cells": chunk.estimated_cells,
            })
            continue
        facts.extend(normalize_sidra_payload_to_facts(
            response.payload,
            table_id=chunk.table_id,
            request_hash=content_hash(chunk.request_params),
            metadata_hash=metadata_hash,
            chunk_request=dict(chunk.request_params),
            unit_by_variable=unit_by_variable,
        ))

    out_path: str | None = None
    if facts:
        out_path = str(write_facts_parquet(facts, output_path=output_path))

    return AcquisitionReport(
        table_id=request.table_id,
        chunk_count=len(chunks),
        estimated_cells=estimated_cells,
        fact_rows=len(facts),
        seconds=elapsed,
        output_path=out_path,
        failed_chunks=tuple(failed),
    )


__all__ = ["AcquisitionReport", "acquire_table_facts"]
