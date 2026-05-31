from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    raw: Path
    processed: Path
    metadata: Path
    cache: Path
    manifests: Path
    intermediate: Path
    runs: Path
    diagnostics: Path
    config: Path
    registries: Path
    intents: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root = Path(root).resolve()

        return cls(
            root=root,
            data_root=root / "data",
            raw=root / "data" / "raw",
            processed=root / "data" / "processed",
            metadata=root / "data" / "metadata",
            cache=root / "data" / "cache",
            manifests=root / "data" / "manifests",
            intermediate=root / "data" / "intermediate",
            runs=root / "data" / "runs",
            diagnostics=root / "data" / "diagnostics",
            config=root / "config",
            registries=root / "config" / "registries",
            intents=root / "config" / "intents",
        )

    def ensure_all(self) -> None:
        for path in (
            self.data_root,
            self.raw,
            self.processed,
            self.metadata,
            self.cache,
            self.manifests,
            self.intermediate,
            self.runs,
            self.diagnostics,
            self.config,
            self.registries,
            self.intents,
        ):
            path.mkdir(parents=True, exist_ok=True)