from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.acceptance.contracts import acceptance_plan, evaluate_level3_acceptance, summarize_run


def run_acceptance_plan() -> dict[str, Any]:
    return acceptance_plan()


def run_acceptance_check_run(*, run_dir: str | Path, require_non_scaffold: bool = False) -> dict[str, Any]:
    return summarize_run(run_dir, require_non_scaffold=require_non_scaffold).as_manifest()


def run_acceptance_level3(*, run_dir: str | Path) -> dict[str, Any]:
    return evaluate_level3_acceptance(run_dir).as_manifest()
