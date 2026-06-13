
"""Workflow wrappers for Slice 17B PIRS model execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.model_execution import execute_pirs_model_from_design_matrix, inspect_pirs_model_execution_manifest


def run_execute_pirs_model(*, run_dir: str | Path, design_matrix_manifest: str | Path | dict[str, Any] | None = None, output_manifest: str | Path | None = None, mutate_output_bundle: bool = True, validate: bool = True, attach: bool = True) -> dict[str, Any]:
    return execute_pirs_model_from_design_matrix(run_dir=run_dir, design_matrix_manifest=design_matrix_manifest, output_manifest=output_manifest, mutate_output_bundle=mutate_output_bundle, validate=validate, attach=attach)


__all__ = ["inspect_pirs_model_execution_manifest", "run_execute_pirs_model"]
