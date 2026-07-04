"""Artifact store + query version-pinning (MSD-III §VI.2-3)."""

from __future__ import annotations

from collections import defaultdict

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

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._versions))

    def pin(self, names) -> dict[str, str]:
        """The ``{name: version}`` a query consumed — its exact-reproducibility manifest."""
        return {name: self.latest(name).version for name in names}


__all__ = ["AssetStore"]
