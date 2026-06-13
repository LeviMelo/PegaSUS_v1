
"""Workflow wrappers for Slice 17A PIRS design-matrix artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.design_matrix import (
    attach_pirs_design_matrix_to_run,
    inspect_pirs_design_matrix_manifest,
    write_pirs_design_matrix_artifact,
)


def run_write_pirs_design_matrix(
    *,
    run_dir: str | Path,
    design_plan: str | Path | dict[str, Any] | None = None,
    readiness_manifest: str | Path | dict[str, Any] | None = None,
    output_manifest: str | Path | None = None,
    output_matrix: str | Path | None = None,
    write_matrix: bool = True,
) -> dict[str, Any]:
    return write_pirs_design_matrix_artifact(
        run_dir=run_dir,
        design_plan=design_plan,
        readiness_manifest=readiness_manifest,
        output_manifest=output_manifest,
        output_matrix=output_matrix,
        write_matrix=write_matrix,
    )


def run_attach_pirs_design_matrix_to_run(
    *,
    run_dir: str | Path,
    design_plan: str | Path | dict[str, Any] | None = None,
    readiness_manifest: str | Path | dict[str, Any] | None = None,
    output_manifest: str | Path | None = None,
    output_matrix: str | Path | None = None,
    write_matrix: bool = True,
) -> dict[str, Any]:
    return attach_pirs_design_matrix_to_run(
        run_dir=run_dir,
        design_plan=design_plan,
        readiness_manifest=readiness_manifest,
        output_manifest=output_manifest,
        output_matrix=output_matrix,
        write_matrix=write_matrix,
    )


__all__ = [
    "inspect_pirs_design_matrix_manifest",
    "run_attach_pirs_design_matrix_to_run",
    "run_write_pirs_design_matrix",
]
