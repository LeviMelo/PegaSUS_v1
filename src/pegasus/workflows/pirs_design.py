
from __future__ import annotations

"""Workflow wrappers for Slice 16C PIRS design-plan artifacts."""

from pathlib import Path
from typing import Any

from pegasus.pirs.design_plan import attach_pirs_design_plan_to_run, write_pirs_design_plan


def run_plan_pirs_design(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | dict[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    return write_pirs_design_plan(
        run_dir=run_dir,
        selection_plan=selection_plan,
        output=output,
        budget=budget,
    )


def run_attach_pirs_design_plan_to_run(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | dict[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    return attach_pirs_design_plan_to_run(
        run_dir=run_dir,
        selection_plan=selection_plan,
        output=output,
        budget=budget,
    )
