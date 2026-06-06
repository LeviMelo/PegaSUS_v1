from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from pegasus.core.hashing import content_hash
from pegasus.sidra.api import SidraClient
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.sidra.schemas import SIDRAChunk


class SIDRAChunkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    table_id: str
    status: str
    status_code: int
    raw_path: str | None
    facts_path: str | None
    row_count: int
    request_hash: str
    error_message: str | None = None


def write_chunk_plan(chunks: list[SIDRAChunk], *, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def read_chunk_plan(path: str | Path) -> list[SIDRAChunk]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("SIDRA chunk plan must be a JSON list.")
    return [SIDRAChunk.model_validate(x) for x in payload]


def _raw_chunk_path(chunk: SIDRAChunk, *, raw_dir: str | Path) -> Path:
    return Path(raw_dir) / f"{chunk.chunk_id}.json"


def _facts_chunk_path(chunk: SIDRAChunk, *, facts_root: str | Path) -> Path:
    return Path(facts_root) / chunk.table_id / f"{chunk.chunk_id}.parquet"


def extract_one_chunk(
    chunk: SIDRAChunk,
    *,
    client: SidraClient,
    raw_dir: str | Path = "data/raw/sidra/chunks",
    facts_root: str | Path = "data/processed/sidra/facts",
    metadata_hash: str = "metadata_unset",
    unit_by_variable: dict[str, str | None] | None = None,
) -> SIDRAChunkResult:
    raw_path = _raw_chunk_path(chunk, raw_dir=raw_dir)
    facts_path = _facts_chunk_path(chunk, facts_root=facts_root)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    facts_path.parent.mkdir(parents=True, exist_ok=True)

    response = client.values_from_chunk(chunk)
    request_hash = content_hash(chunk.model_dump(mode="json"))

    raw_path.write_text(
        json.dumps(
            {
                "chunk": chunk.model_dump(mode="json"),
                "status_code": response.status_code,
                "from_cache": response.from_cache,
                "sidecar": response.sidecar,
                "payload": response.payload,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if response.status_code >= 400:
        return SIDRAChunkResult(
            chunk_id=chunk.chunk_id,
            table_id=chunk.table_id,
            status="failed",
            status_code=response.status_code,
            raw_path=str(raw_path),
            facts_path=None,
            row_count=0,
            request_hash=request_hash,
            error_message=f"SIDRA HTTP status {response.status_code}",
        )

    facts = normalize_sidra_payload_to_facts(
        response.payload,
        table_id=chunk.table_id,
        request_hash=request_hash,
        metadata_hash=metadata_hash,
        chunk_request=chunk.request_params,
        unit_by_variable=unit_by_variable,
        fetched_at=(response.sidecar or {}).get("fetched_at"),
    )
    write_facts_parquet(facts, output_path=facts_path)

    return SIDRAChunkResult(
        chunk_id=chunk.chunk_id,
        table_id=chunk.table_id,
        status="success",
        status_code=response.status_code,
        raw_path=str(raw_path),
        facts_path=str(facts_path),
        row_count=len(facts),
        request_hash=request_hash,
    )


def extract_chunk_plan(
    chunks: list[SIDRAChunk],
    *,
    client: SidraClient,
    concurrency: int = 4,
    raw_dir: str | Path = "data/raw/sidra/chunks",
    facts_root: str | Path = "data/processed/sidra/facts",
    metadata_hash: str = "metadata_unset",
    unit_by_variable: dict[str, str | None] | None = None,
) -> list[SIDRAChunkResult]:
    if concurrency <= 1:
        return [
            extract_one_chunk(
                chunk,
                client=client,
                raw_dir=raw_dir,
                facts_root=facts_root,
                metadata_hash=metadata_hash,
                unit_by_variable=unit_by_variable,
            )
            for chunk in chunks
        ]

    results: list[SIDRAChunkResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                extract_one_chunk,
                chunk,
                client=client,
                raw_dir=raw_dir,
                facts_root=facts_root,
                metadata_hash=metadata_hash,
                unit_by_variable=unit_by_variable,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            results.append(future.result())

    return sorted(results, key=lambda r: r.chunk_id)


def write_extraction_log(results: list[SIDRAChunkResult], *, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
