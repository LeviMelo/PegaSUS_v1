"""Typed run context threaded through the compiler (MSD §1.2 execution chain).

This replaces the previous practice of passing ``run_dir`` strings plus a relay
of JSON manifests between every stage, and of dispatching stages by string name
through ``importlib``. The context carries the live, typed state of a single
compile so downstream phases consume objects instead of re-reading disk.

It is intentionally additive: stages still write their first-class artifacts for
the immutable output bundle, but they no longer have to round-trip intermediate
state through JSON to communicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Live state of one PegaSUS compile run."""

    run_dir: Path
    run_id: str | None = None
    data_root: Path | None = None
    budget: str = "fast"
    intent: Any | None = None
    geo_scope: Any | None = None
    bundle: Any | None = None  # OutputBundleManager
    telemetry: Any | None = None
    compiler_stage_plan: Any | None = None

    # Typed in-memory carriers populated as phases run. Stages read these
    # instead of re-loading the manifest a previous stage just wrote.
    substrate: Any | None = None
    efg_result: Any | None = None
    pirs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.run_id is None:
            self.run_id = self.run_dir.name
        self.data_root = Path(self.data_root) if self.data_root is not None else self.run_dir.parent

    @property
    def pirs_root(self) -> Path:
        """Workspace the PIRS stages operate in.

        When a bundle manager is staging tables, PIRS runs in its staging
        workspace so partial artifacts never leak into the run dir before the
        atomic flush; otherwise it operates directly on the run dir.
        """
        staged = self.pirs.get("workspace")
        return Path(staged) if staged is not None else self.run_dir

    def stage_pirs_workspace(self) -> Path:
        if self.bundle is None:
            self.pirs["workspace"] = self.run_dir
            return self.run_dir
        workspace = self.run_dir.parent / f"{self.run_dir.name}__pirs_stage_workspace"
        resolved = self.bundle.write_stage_workspace(workspace)
        self.pirs["workspace"] = resolved
        return Path(resolved)
