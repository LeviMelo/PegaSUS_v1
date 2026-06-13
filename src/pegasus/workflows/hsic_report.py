from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.hsic_report import build_hsic_report_artifacts, inspect_hsic_report_manifest


def run_export_hsic_report(
    *,
    run_dir: str | Path,
    ranking_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    return build_hsic_report_artifacts(run_dir=run_dir, ranking_manifest=ranking_manifest, output_dir=output_dir)


def run_attach_hsic_report_to_run(
    *,
    run_dir: str | Path,
    ranking_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    return build_hsic_report_artifacts(run_dir=run_dir, ranking_manifest=ranking_manifest, output_dir=output_dir)


__all__ = [
    "run_export_hsic_report",
    "run_attach_hsic_report_to_run",
    "inspect_hsic_report_manifest",
]
