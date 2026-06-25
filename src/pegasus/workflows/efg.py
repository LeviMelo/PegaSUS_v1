from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.workflows.build_efg import build_autonomous_efg_from_manifest


def run_build_autonomous_efg(
    *,
    substrate_manifest: str | Path,
    output_path: str | Path,
    registry_root: str | Path = "config/registries",
    operator_budget: int = 256,
    operator_mode: str = "standard",
) -> dict[str, Any]:
    result = build_autonomous_efg_from_manifest(
        substrate_manifest=substrate_manifest,
        output_path=output_path,
        registry_root=registry_root,
        operator_budget=operator_budget,
        operator_mode=operator_mode,
    )
    return {
        "output_path": Path(output_path),
        "efg_id": result.efg_id,
        "field_count": result.field_count,
        "edge_count": result.edge_count,
        "failed_branch_count": len(result.failed_branches),
        "legality_summary": result.legality_summary,
    }


__all__ = ["run_build_autonomous_efg"]
