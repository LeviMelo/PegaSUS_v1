from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash


class DatasusCache:
    def __init__(self, root: str | Path = "data/cache/datasus/microdatasus") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def request_hash(self, request: dict[str, Any]) -> str:
        return content_hash(request)

    def request_dir(self, request_hash: str) -> Path:
        path = self.root / request_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_request(self, request_hash: str, request: dict[str, Any]) -> Path:
        path = self.request_dir(request_hash) / "request.json"
        path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
