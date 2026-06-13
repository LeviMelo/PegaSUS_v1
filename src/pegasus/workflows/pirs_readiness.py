
"""Workflow wrappers for Slice 16D PIRS design-readiness gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.design_readiness import (
    attach_pirs_design_readiness_to_run,
    build_pirs_design_readiness_gate,
    write_pirs_design_readiness_manifest,
)


def run_pirs_design_readiness_gate(
    *,
    run_dir: str | Path,
    design_plan: str | Path | None = None,
) -> dict[str, Any]:
    return build_pirs_design_readiness_gate(run_dir=run_dir, design_plan=design_plan).as_manifest()


def run_write_pirs_design_readiness_manifest(
    *,
    run_dir: str | Path,
    design_plan: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return write_pirs_design_readiness_manifest(run_dir=run_dir, design_plan=design_plan, output=output)


def run_attach_pirs_design_readiness_to_run(
    *,
    run_dir: str | Path,
    design_plan: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return attach_pirs_design_readiness_to_run(run_dir=run_dir, design_plan=design_plan, output=output)
