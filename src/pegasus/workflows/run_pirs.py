from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.output.pirs_bundle import pirs_plan_from_fixture, write_pirs_fixture_bundle
from pegasus.output.validate import validate_output_bundle


def run_pirs_plan_fixture(*, input_path: str | Path, budget: str = "standard") -> dict[str, Any]:
    return pirs_plan_from_fixture(input_path=input_path, budget=budget)


def run_pirs_build_fixture(*, input_path: str | Path, run_dir: str | Path, budget: str = "standard") -> dict[str, Any]:
    out = write_pirs_fixture_bundle(input_path=input_path, run_dir=run_dir, budget=budget)
    validation = validate_output_bundle(run_dir=str(out))
    return {"run_dir": out, "validation": validation, "status": "success" if validation.ok else "failed"}
