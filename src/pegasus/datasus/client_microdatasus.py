"""Typed Python client for the isolated microdatasus R subprocess."""

from __future__ import annotations

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

    def fetch(self, *, system: str, uf: str, years: str) -> MicrodatasusBatchResult:
        requests = build_datasus_manifests(
            system=system, uf=uf, years=years,
            config={"rscript_path": self.config.rscript_path}, data_root=self.data_root,
        )
        completed: list[DATASUSRequestManifest] = []
        paths: list[str] = []
        for request in requests:
            result = fetch_datasus_chunk(
                request, config=self.config, cache=self.cache,
                timeout_seconds=self.config.r_timeout_seconds,
                heartbeat_timeout_seconds=self.config.heartbeat_timeout_seconds,
            )
            completed.append(result)
            paths.append(str(write_request_manifest(result, root=self.manifest_root)))
        return MicrodatasusBatchResult(tuple(completed), tuple(paths))
