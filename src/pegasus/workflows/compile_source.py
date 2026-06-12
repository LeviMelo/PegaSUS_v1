"""Workflow helpers for Slice 12B compile source-reality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.source_artifacts.compile_policy import resolve_compile_source_reality


def run_compile_source_reality_plan(
    *,
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    return resolve_compile_source_reality(
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    ).as_manifest()
