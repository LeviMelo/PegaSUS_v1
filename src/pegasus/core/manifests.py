from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pegasus import __version__
from pegasus.core.hashing import sha256_file


class FileManifest(BaseModel):
    path: str
    sha256: str
    bytes: int


class EnvironmentManifest(BaseModel):
    python_version: str
    platform: str
    pegasus_version: str
    created_at: str


class ReproducibilityManifest(BaseModel):
    run_id: str
    created_at: str
    environment: EnvironmentManifest
    files: list[FileManifest] = Field(default_factory=list)
    registry_hashes: dict[str, str] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_environment_manifest() -> EnvironmentManifest:
    return EnvironmentManifest(
        python_version=sys.version,
        platform=platform.platform(),
        pegasus_version=__version__,
        created_at=utc_now_iso(),
    )


def manifest_file(path: str | Path, root: str | Path | None = None) -> FileManifest:
    path = Path(path)
    display_path = str(path if root is None else path.relative_to(root))
    return FileManifest(
        path=display_path,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
    )


def write_json(path: str | Path, payload: BaseModel | dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    else:
        data = payload

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)