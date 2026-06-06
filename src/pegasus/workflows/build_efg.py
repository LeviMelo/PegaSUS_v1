from __future__ import annotations

from pathlib import Path

from pegasus.output.sim_efg_bundle import write_sim_fixture_efg_bundle


def build_sim_fixture_efg_run(
    *,
    sim_events_path: str | Path,
    run_dir: str | Path,
) -> Path:
    return write_sim_fixture_efg_bundle(
        sim_events_path=sim_events_path,
        run_dir=run_dir,
    )
