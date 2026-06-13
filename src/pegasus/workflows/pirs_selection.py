
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pegasus.pirs.selection_plan import attach_pirs_selection_plan_to_run, write_pirs_selection_plan

Budget = Literal["fast", "standard", "deep"]


def run_write_pirs_selection_plan(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    return write_pirs_selection_plan(run_dir=run_dir, candidate_manifest=candidate_manifest, output=output, budget=budget)


def run_attach_pirs_selection_plan(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    return attach_pirs_selection_plan_to_run(run_dir=run_dir, candidate_manifest=candidate_manifest, output=output, budget=budget)
