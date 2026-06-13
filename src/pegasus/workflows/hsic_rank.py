from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.hsic_ranking import build_hsic_ranking_artifacts, inspect_hsic_ranking_manifest


def run_rank_hsic_residual_scan(
    *,
    run_dir: str | Path,
    scan_manifest: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return build_hsic_ranking_artifacts(run_dir=run_dir, scan_manifest=scan_manifest, output=output)


def run_attach_hsic_ranking_to_run(
    *,
    run_dir: str | Path,
    scan_manifest: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return build_hsic_ranking_artifacts(run_dir=run_dir, scan_manifest=scan_manifest, output=output)


__all__ = [
    "run_rank_hsic_residual_scan",
    "run_attach_hsic_ranking_to_run",
    "inspect_hsic_ranking_manifest",
]
