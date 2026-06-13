
"""Workflow wrappers for Slice 14B EFG substrate materialization manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.efg.materialization_manifest import (
    attach_efg_materialization_summary_to_run,
    build_efg_materialization_manifest,
    write_efg_materialization_manifest,
)


def run_materialize_substrate_manifest(
    *,
    substrate_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    if output is None:
        return build_efg_materialization_manifest(substrate_manifest=substrate_manifest)
    return write_efg_materialization_manifest(substrate_manifest=substrate_manifest, output=output)


def run_attach_efg_materialization_to_run(
    *,
    run_dir: str | Path,
    substrate_manifest: str | Path | None = None,
) -> dict[str, Any]:
    return attach_efg_materialization_summary_to_run(run_dir=run_dir, substrate_manifest=substrate_manifest)
