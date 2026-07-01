
"""Workflow wrappers for Slice 15A EFG promotion planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.efg.promotion_plan import attach_efg_promotion_plan_to_run, write_efg_promotion_plan


def run_plan_efg_promotion(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return write_efg_promotion_plan(run_dir=run_dir, materialization_manifest=materialization_manifest, output=output)


def run_attach_efg_promotion_plan_to_run(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path | None = None,
) -> dict[str, Any]:
    return attach_efg_promotion_plan_to_run(run_dir=run_dir, materialization_manifest=materialization_manifest)
