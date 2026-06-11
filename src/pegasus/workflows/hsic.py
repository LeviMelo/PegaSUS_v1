"""Workflow wrappers for Slice 9A HSIC residual scanner fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.output.hsic_bundle import plan_hsic_fixture, write_hsic_fixture_bundle


def run_hsic_plan_fixture(*, input_path: str | Path, budget: str = "standard", cuda_required: bool = False) -> dict[str, Any]:
    return plan_hsic_fixture(input_path=input_path, budget=budget, cuda_required=cuda_required)


def run_hsic_build_fixture(*, input_path: str | Path, run_dir: str | Path, budget: str = "standard", cuda_required: bool = False) -> dict[str, Any]:
    write_hsic_fixture_bundle(input_path=input_path, run_dir=run_dir, budget=budget, cuda_required=cuda_required)
    return {"run_dir": str(run_dir)}
