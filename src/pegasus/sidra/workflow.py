from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from pegasus.core.config import load_config
from pegasus.datasus.io import write_json, write_parquet
from pegasus.sidra.client import SIDRAClient, read_raw_sidra_json, write_raw_sidra_json
from pegasus.sidra.models import SIDRAFetchManifest, SIDRAQuerySpec
from pegasus.sidra.normalize import normalize_sidra_response


class SIDRAEmptyResultError(RuntimeError):
    pass


def load_sidra_query_spec(path: str | Path) -> SIDRAQuerySpec:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data: dict[str, Any] = yaml.safe_load(f)
        else:
            data = json.load(f)

    return SIDRAQuerySpec.model_validate(data)


def fetch_sidra_query_to_disk(
    *,
    root: str | Path,
    spec_path: str | Path,
    out_root: str | Path,
    allow_empty: bool = False,
    use_cache: bool = True,
) -> SIDRAFetchManifest:
    total_start = time.perf_counter()

    root = Path(root).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = load_config(root)
    spec = load_sidra_query_spec(spec_path)

    client = SIDRAClient(
        base_url=config.sidra.base_url,
        timeout_seconds=config.sidra.timeout_seconds,
        max_retries=config.sidra.max_retries,
        backoff_initial_seconds=config.sidra.backoff_initial_seconds,
        backoff_max_seconds=config.sidra.backoff_max_seconds,
    )

    cache_key = client.cache_key(spec)
    raw_json_path = out_root / f"{cache_key}.raw.json"
    facts_path = out_root / f"{cache_key}.facts.parquet"
    manifest_path = out_root / f"{cache_key}.manifest.json"

    url = client.build_url(spec)
    cache_hit = False
    download_bytes: int | None = None
    timings: dict[str, float] = {}

    if use_cache and raw_json_path.exists():
        cache_start = time.perf_counter()
        payload = read_raw_sidra_json(raw_json_path)
        timings["cache_read"] = time.perf_counter() - cache_start
        cache_hit = True
    else:
        fetch_start = time.perf_counter()
        result = client.fetch_json(spec)
        timings["http_fetch"] = result.elapsed_seconds
        timings["http_fetch_wall"] = time.perf_counter() - fetch_start
        timings["http_attempts"] = float(result.attempts)
        download_bytes = result.download_bytes
        url = result.url
        payload = result.payload

        write_start = time.perf_counter()
        write_raw_sidra_json(raw_json_path, payload)
        timings["raw_json_write"] = time.perf_counter() - write_start

    normalize_start = time.perf_counter()
    facts, normalization_warnings = normalize_sidra_response(
        payload,
        table_id=spec.table_id,
        expected_variables=set(spec.variables),
        expected_periods=set(spec.periods),
    )
    timings["normalize"] = time.perf_counter() - normalize_start

    write_facts_start = time.perf_counter()
    write_parquet(facts, facts_path)
    timings["facts_write"] = time.perf_counter() - write_facts_start

    n_raw_top_level_items = len(payload) if isinstance(payload, list) else None
    empty_result = facts.height == 0

    timings["total"] = time.perf_counter() - total_start

    manifest = SIDRAFetchManifest(
        table_id=spec.table_id,
        periods=spec.periods,
        variables=spec.variables,
        localities=spec.localities,
        classifications=spec.classifications,
        view=spec.view,
        url=url,
        cache_key=cache_key,
        cache_hit=cache_hit,
        download_bytes=download_bytes,
        timings_seconds={k: round(v, 6) for k, v in timings.items()},
        raw_json_path=str(raw_json_path),
        facts_path=str(facts_path),
        manifest_path=str(manifest_path),
        n_raw_top_level_items=n_raw_top_level_items,
        n_facts=facts.height,
        empty_result=empty_result,
        warnings=normalization_warnings,
    )

    write_json(manifest_path, manifest)

    if empty_result and not allow_empty:
        raise SIDRAEmptyResultError(
            "SIDRA query returned zero normalized facts. "
            f"Raw response was saved at {raw_json_path}. "
            f"Manifest was saved at {manifest_path}. "
            f"URL: {url}"
        )

    return manifest