import polars as pl

from pegasus.workflows.compile import run_compile
from pegasus.workflows.race_bridge import run_attach_race_bridge


def test_race_bridge_attach_is_idempotent_on_compile_run(tmp_path):
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=tmp_path / "run",
        data_root=tmp_path / "data",
    )
    assert result["validation"].ok, result["validation"].errors
    sim_events = tmp_path / "data" / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
    for _ in range(2):
        attach = run_attach_race_bridge(
            run_dir=tmp_path / "run",
            sim_events_path=sim_events,
            bridge_prior_path="config/priors/race_bridge/fixedC_sim_admin_to_ibge_selfdeclared_v1.json",
            municipality_cod6="270430",
        )
        assert attach["validation"].ok, attach["validation"].errors
    v = pl.read_parquet(tmp_path / "run" / "V_fields.parquet")
    assert v.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").height == 1
    assert v.filter(pl.col("field_id") == "SIMRaceBridgeMissingRaceObserver").height == 1
