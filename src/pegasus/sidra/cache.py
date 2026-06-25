from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash
from pegasus.core.io_utils import _write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_request_hash(*, url: str, params: dict[str, Any] | None = None) -> str:
    return content_hash(
        {
            "url": url,
            "params": params or {},
        }
    )


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheRead:
    hit: bool
    payload: Any | None
    sidecar: dict[str, Any] | None
    payload_path: Path
    sidecar_path: Path


class SidraJsonCache:
    def __init__(self, root: str | Path = "data/cache/sidra") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def paths(self, *, namespace: str, request_hash: str) -> tuple[Path, Path]:
        directory = self.root / namespace
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{request_hash}.json", directory / f"{request_hash}.sidecar.json"

    def read(
        self,
        *,
        namespace: str,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> CacheRead:
        request_hash = stable_request_hash(url=url, params=params)
        payload_path, sidecar_path = self.paths(namespace=namespace, request_hash=request_hash)

        if not payload_path.exists() or not sidecar_path.exists():
            return CacheRead(False, None, None, payload_path, sidecar_path)

        try:
            return CacheRead(
                True,
                json.loads(payload_path.read_text(encoding="utf-8")),
                json.loads(sidecar_path.read_text(encoding="utf-8")),
                payload_path,
                sidecar_path,
            )
        except json.JSONDecodeError:
            return CacheRead(False, None, None, payload_path, sidecar_path)

    def write(
        self,
        *,
        namespace: str,
        url: str,
        params: dict[str, Any] | None,
        payload: Any,
        status_code: int,
        attempt: int,
        seconds: float,
    ) -> tuple[Path, Path, dict[str, Any]]:
        request_hash = stable_request_hash(url=url, params=params)
        payload_path, sidecar_path = self.paths(namespace=namespace, request_hash=request_hash)

        raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        _write_json(payload_path, payload)

        sidecar = {
            "request_hash": request_hash,
            "url": url,
            "params": params or {},
            "status_code": status_code,
            "fetched_at": utc_now(),
            "attempt": attempt,
            "seconds": round(seconds, 6),
            "bytes": len(raw_text.encode("utf-8")),
            "sha256": sha256_text(raw_text),
            "payload_path": str(payload_path),
        }
        _write_json(sidecar_path, sidecar)
        return payload_path, sidecar_path, sidecar
