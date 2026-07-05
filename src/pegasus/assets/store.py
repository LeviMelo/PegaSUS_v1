"""Artifact store + query version-pinning (MSD-III §VI.2-3)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from pegasus.assets.version import AssetVersion


class AssetStore:
    """An append-only store of immutable asset versions.

    New data arrival appends a new version (never mutates an old one), so any prior
    result stays reproducible. ``latest`` serves the current version; ``get`` retrieves
    a pinned one; ``pin`` records the versions a query consumed (§VI.2 reproducibility).
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[AssetVersion]] = defaultdict(list)

    def put(self, version: AssetVersion) -> None:
        history = self._versions[version.name]
        if any(v.version == version.version for v in history):
            raise ValueError(f"asset {version.name}@{version.version} already exists (immutable)")
        history.append(version)

    def latest(self, name: str) -> AssetVersion:
        history = self._versions.get(name)
        if not history:
            raise KeyError(f"no asset named {name!r} in the store")
        return history[-1]

    def get(self, name: str, version: str) -> AssetVersion:
        for v in self._versions.get(name, ()):
            if v.version == version:
                return v
        raise KeyError(f"asset {name}@{version} not found")

    def has(self, name: str) -> bool:
        return bool(self._versions.get(name))

    def history(self, name: str) -> tuple[AssetVersion, ...]:
        return tuple(self._versions.get(name, ()))

    def find_by_identity(self, name: str, input_identity: str) -> AssetVersion | None:
        """The existing version whose inputs match ``input_identity`` (the build-once key), or None.

        This is what makes the build idempotent: a foundation build is triggered by *data arrival*
        (§VI.2), so if the inputs are unchanged the existing version is reused rather than rebuilt.
        """
        for v in self._versions.get(name, ()):
            if v.input_manifest.get("input_identity") == input_identity:
                return v
        return None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._versions))

    def pin(self, names) -> dict[str, str]:
        """The ``{name: version}`` a query consumed — its exact-reproducibility manifest."""
        return {name: self.latest(name).version for name in names}


class PersistentAssetStore(AssetStore):
    """An :class:`AssetStore` backed by an on-disk JSON index, so build-once survives across runs.

    The index at ``<root>/index.json`` records every version's metadata and its ``payload`` path
    (the versioned tensor parquet under ``<root>/<name>/<version>/``). Foundation builds are triggered
    by data arrival and persist here; every later query loads this store and *slices* the latest
    version instead of rebuilding (§VI.1-3). Payloads are large arrays referenced by path, never
    inlined (§V.7(4)).
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self._index_path = self.root / "index.json"
        self._load()

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        for rec in data.get("versions", []):
            super().put(AssetVersion(
                name=rec["name"],
                version=rec["version"],
                build_scope=rec["build_scope"],
                input_manifest=rec.get("input_manifest", {}),
                certification=rec.get("certification", "unverified"),
                payload=rec.get("payload"),
            ))

    def put(self, version: AssetVersion) -> None:
        super().put(version)
        self._save()

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        records = [asdict(v) for name in self._versions for v in self._versions[name]]
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"versions": records}, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._index_path)

    def version_dir(self, name: str, version: str) -> Path:
        return self.root / name / version


__all__ = ["AssetStore", "PersistentAssetStore"]
