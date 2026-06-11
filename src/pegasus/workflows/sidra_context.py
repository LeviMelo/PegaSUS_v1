from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.output.sidra_stdfm_bundle import build_sidra_stdfm_fixture_bundle
from pegasus.sidra.projection import load_projection_matrix, projection_metadata
from pegasus.sidra.stitching import SIDRASegment, stitch_sidra_longitudinal_segments
from pegasus.she.high_dimensional import bound_high_dimensional_sidra_exposure
from pegasus.she.stdfm.blocked import blocked_solver_pending


def run_sidra_context_plan_fixture(*, input_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    stitch = stitch_sidra_longitudinal_segments([SIDRASegment.from_mapping(x) for x in payload["stitching"]["segments"]])
    matrix = load_projection_matrix(payload["projection"])
    projection = projection_metadata(measure_kind="additive", has_denominator=False, matrix=matrix)
    bound = bound_high_dimensional_sidra_exposure(
        raw_axes=payload["high_dimensional"]["raw_axes"],
        demanded_axes=payload["high_dimensional"]["demanded_axes"],
        axis_cardinalities={k: int(v) for k, v in payload["high_dimensional"]["axis_cardinalities"].items()},
        aggregation=payload["high_dimensional"].get("aggregation", "additive"),
        high_dimensional=True,
    )
    stdfm = blocked_solver_pending(field_id="sidra_stdfm_blocked_candidate")
    return {
        "segments": len(stitch.segment_provenance),
        "stitch_status": stitch.status,
        "projection_status": projection["status"],
        "projection_warnings": projection["warnings"],
        "high_dimensional_status": bound.status,
        "stdfm_status": stdfm.status,
    }


def run_sidra_context_build_fixture(*, input_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    path = build_sidra_stdfm_fixture_bundle(input_path=input_path, run_dir=run_dir)
    return {"run_dir": str(path)}
