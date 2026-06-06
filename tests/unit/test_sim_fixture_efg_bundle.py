from pathlib import Path

import polars as pl

from pegasus.datasus.normalize import normalize_sim_do_events
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.build_efg import build_sim_fixture_efg_run


def test_sim_fixture_efg_bundle_validates(tmp_path: Path):
    sim_events = tmp_path / "sim_events.parquet"
    run_dir = tmp_path / "run"

    normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=sim_events,
        source_manifest_hash="fixture_manifest_hash",
    )

    build_sim_fixture_efg_run(
        sim_events_path=sim_events,
        run_dir=run_dir,
    )

    result = validate_output_bundle(run_dir=str(run_dir))
    assert result.ok, result.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    assert {"SIMDeathsAll", "SIMAdministrativeRaceDeaths", "IBGESelfDeclaredPopulationPlaceholder"} <= set(v["name"].to_list())

    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    assert q.height == v.height

    failed = pl.read_parquet(run_dir / "FailedBranches.parquet")
    assert failed.height == 1
    assert failed.row(0, named=True)["failure_stage"] == "declaration"
    assert "declaration" in failed.row(0, named=True)["failed_terms"]
