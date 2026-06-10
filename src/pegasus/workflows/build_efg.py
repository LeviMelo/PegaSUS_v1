from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.output.sim_efg_bundle import write_sim_fixture_efg_bundle


def build_sim_fixture_efg_run(
    *,
    sim_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    source_events_path = Path(sim_events_path)

    if municipality_cod6 is not None:
        filtered_path = run_dir / "Tables" / f"sim_events_mun_{municipality_cod6}.parquet"
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.read_parquet(source_events_path)
        filtered = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
        if filtered.height == 0:
            raise ValueError(f"No SIM fixture events remain after municipality filter {municipality_cod6!r}.")
        filtered.write_parquet(filtered_path)
        source_events_path = filtered_path

    return write_sim_fixture_efg_bundle(
        sim_events_path=source_events_path,
        run_dir=run_dir,
    )
