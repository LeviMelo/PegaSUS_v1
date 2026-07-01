"""Typed Python client for the isolated microdatasus R subprocess."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.schemas import DATASUSRequestManifest
from pegasus.datasus.cache import DatasusCache
from pegasus.datasus.manifests import build_datasus_manifests, write_request_manifest
from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk


@dataclass(frozen=True)
class MicrodatasusBatchResult:
    requests: tuple[DATASUSRequestManifest, ...]
    manifest_paths: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.requests) and all(item.status in {"success", "cached"} for item in self.requests)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_count": len(self.requests),
            "success_count": sum(item.status in {"success", "cached"} for item in self.requests),
            "requests": [item.model_dump(mode="json") for item in self.requests],
            "manifest_paths": list(self.manifest_paths),
        }


class MicrodatasusClient:
    def __init__(
        self,
        *,
        config: DatasusConfig | None = None,
        cache: DatasusCache | None = None,
        data_root: str | Path = "data",
        manifest_root: str | Path = "data/manifests/datasus",
    ) -> None:
        self.config = config or DatasusConfig.from_file()
        self.cache = cache or DatasusCache()
        self.data_root = Path(data_root)
        self.manifest_root = Path(manifest_root)

    def _run_requests(self, requests: list[DATASUSRequestManifest]) -> MicrodatasusBatchResult:
        if not requests:
            return MicrodatasusBatchResult((), ())

        def run_one(request: DATASUSRequestManifest) -> tuple[DATASUSRequestManifest, str]:
            result = fetch_datasus_chunk(
                request, config=self.config, cache=self.cache,
                timeout_seconds=self.config.r_timeout_seconds,
                heartbeat_timeout_seconds=self.config.heartbeat_timeout_seconds,
            )
            return result, str(write_request_manifest(result, root=self.manifest_root))

        completed: list[DATASUSRequestManifest | None] = [None] * len(requests)
        paths: list[str | None] = [None] * len(requests)
        worker_count = min(max(1, self.config.max_parallel_requests), len(requests))
        if worker_count == 1:
            for idx, request in enumerate(requests):
                completed[idx], paths[idx] = run_one(request)
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="datasus-fetch") as pool:
                futures = {pool.submit(run_one, request): idx for idx, request in enumerate(requests)}
                for future in as_completed(futures):
                    idx = futures[future]
                    completed[idx], paths[idx] = future.result()
        if any(item is None for item in completed) or any(item is None for item in paths):
            raise RuntimeError("DATASUS parallel fetch did not complete all request slots")
        return MicrodatasusBatchResult(tuple(completed), tuple(paths))

    def fetch(self, *, system: str, uf: str, years: str) -> MicrodatasusBatchResult:
        requests = build_datasus_manifests(
            system=system, uf=uf, years=years,
            config={"rscript_path": self.config.rscript_path}, data_root=self.data_root,
        )
        return self._run_requests(list(requests))

    def fetch_systems(self, *, systems: list[str], uf: str, years: str) -> dict[str, MicrodatasusBatchResult]:
        """Fetch multiple systems in ONE global worker pool.

        The per-system ``fetch`` parallelizes only that system's year requests, so
        the harness/pipeline (which loop systems serially) leave systems waiting on
        each other. Flattening every system's requests into a single pool bounded
        by ``max_parallel_requests`` parallelizes across systems AND years, so a
        slow SIH month no longer blocks SIM/SINASC/CNES — the dominant wall-clock
        win for an all-source multi-year run.
        """
        all_requests: list[DATASUSRequestManifest] = []
        spans: dict[str, tuple[int, int]] = {}
        for system in systems:
            reqs = list(build_datasus_manifests(
                system=system, uf=uf, years=years,
                config={"rscript_path": self.config.rscript_path}, data_root=self.data_root,
            ))
            spans[system] = (len(all_requests), len(all_requests) + len(reqs))
            all_requests.extend(reqs)
        batch = self._run_requests(all_requests)
        out: dict[str, MicrodatasusBatchResult] = {}
        for system, (lo, hi) in spans.items():
            out[system] = MicrodatasusBatchResult(batch.requests[lo:hi], batch.manifest_paths[lo:hi])
        return out
