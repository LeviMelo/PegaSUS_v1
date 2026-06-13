
"""Workflow wrappers for run-bundle HSIC residual scanning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pegasus.pirs.hsic_run import execute_hsic_residual_scan, inspect_hsic_residual_scan_manifest


def run_execute_hsic_residual_scan(*, run_dir: str | Path, model_execution_manifest: str | Path | Mapping[str, Any] | None = None, design_matrix_manifest: str | Path | Mapping[str, Any] | None = None, output_manifest: str | Path | None = None, budget: str = "fast", permutations: int = 199, min_support: int = 3, seed: int = 20260613, mutate_output_bundle: bool = True, validate: bool = True, attach: bool = True) -> dict[str, Any]:
    return execute_hsic_residual_scan(run_dir=run_dir, model_execution_manifest=model_execution_manifest, design_matrix_manifest=design_matrix_manifest, output_manifest=output_manifest, budget=budget, permutations=permutations, min_support=min_support, seed=seed, mutate_output_bundle=mutate_output_bundle, validate=validate, attach=attach)


def run_inspect_hsic_residual_scan(*, manifest: str | Path) -> dict[str, Any]:
    return inspect_hsic_residual_scan_manifest(manifest)
