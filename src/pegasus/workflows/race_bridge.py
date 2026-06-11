from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.output.race_bridge_attach import attach_race_bridge_to_run
from pegasus.output.validate import validate_output_bundle


def run_attach_race_bridge(
    *,
    run_dir: str | Path,
    sim_events_path: str | Path,
    bridge_prior_path: str | Path,
    municipality_cod6: str | None = None,
) -> dict[str, Any]:
    output = attach_race_bridge_to_run(
        run_dir=run_dir,
        sim_events_path=sim_events_path,
        bridge_prior_path=bridge_prior_path,
        municipality_cod6=municipality_cod6,
    )
    validation = validate_output_bundle(run_dir=str(output))
    return {"run_dir": output, "validation": validation}
