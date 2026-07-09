"""RunCatalog — discover run bundles by scope/time/seed/stage instead of path-guessing."""
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.run_catalog import (
    build_catalog,
    extract_record,
    find_runs,
    latest_run,
    load_catalog,
    scan_runs,
)


def _make_bundle(runs_root: Path, run_id: str, *, uf, y0, y1, stage, scale, seeds, created):
    d = runs_root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "UserIntent.json").write_text(json.dumps({
        "geography": {"level": "state", "uf": [uf], "codes": []},
        "time": {"start_year": y0, "end_year": y1},
        "health_seeds": seeds,
        "execution_stage": stage,
        "execution_scale": scale,
        "run_profile": "core_vital",
    }), encoding="utf-8")
    (d / "ReproducibilityManifest.json").write_text(json.dumps({
        "run_id": run_id,
        "generated_at": created,
        "code_version": "0.1.0",
        "intent_hash": "abc123",
        "intent_path": f"config/intents/{run_id}.json",
    }), encoding="utf-8")
    return d


def test_scan_discovers_and_sorts_newest_first(tmp_path):
    root = tmp_path / "data/runs"
    _make_bundle(root, "run_old", uf="AL", y0=2015, y1=2020, stage="investigate", scale="state",
                 seeds=["mortality"], created="2026-07-01T00:00:00+00:00")
    _make_bundle(root, "run_new", uf="SP", y0=2000, y1=2024, stage="full", scale="national",
                 seeds=["mortality", "hospitalization"], created="2026-07-09T00:00:00+00:00")
    recs = scan_runs(root)
    assert [r.run_id for r in recs] == ["run_new", "run_old"]  # newest first
    assert recs[0].execution_scale == "national"
    assert recs[1].ufs == ("AL",)


def test_find_and_latest_filter_correctly(tmp_path):
    root = tmp_path / "data/runs"
    _make_bundle(root, "al_2018", uf="AL", y0=2015, y1=2020, stage="investigate", scale="state",
                 seeds=["mortality"], created="2026-07-01T00:00:00+00:00")
    _make_bundle(root, "nat_c25", uf="SP", y0=2000, y1=2024, stage="investigate", scale="national",
                 seeds=["mortality", "cancer"], created="2026-07-08T00:00:00+00:00")
    recs = scan_runs(root)

    assert {r.run_id for r in find_runs(recs, uf="AL")} == {"al_2018"}
    assert {r.run_id for r in find_runs(recs, execution_scale="national")} == {"nat_c25"}
    assert {r.run_id for r in find_runs(recs, seed="cancer")} == {"nat_c25"}
    assert {r.run_id for r in find_runs(recs, year=2019)} == {"al_2018", "nat_c25"}  # both cover 2019
    assert {r.run_id for r in find_runs(recs, year=2023)} == {"nat_c25"}             # only national spans to 2024
    assert latest_run(recs, execution_stage="investigate").run_id == "nat_c25"       # newest of the two


def test_skips_workspaces_and_non_bundles(tmp_path):
    root = tmp_path / "data/runs"
    _make_bundle(root, "real_run", uf="AL", y0=2015, y1=2020, stage="investigate", scale="state",
                 seeds=["mortality"], created="2026-07-01T00:00:00+00:00")
    (root / "real_run__efg_stage_workspace").mkdir(parents=True)  # workspace sibling
    (root / "not_a_bundle").mkdir(parents=True)                   # dir with no identity JSONs
    recs = scan_runs(root)
    assert [r.run_id for r in recs] == ["real_run"]
    assert extract_record(root / "not_a_bundle") is None


def test_catalog_persist_roundtrip(tmp_path):
    root = tmp_path / "data/runs"
    _make_bundle(root, "r1", uf="AL", y0=2015, y1=2020, stage="investigate", scale="state",
                 seeds=["mortality"], created="2026-07-01T00:00:00+00:00")
    out = tmp_path / "data/manifests/runs/catalog.json"
    built = build_catalog(root, out)
    loaded = load_catalog(out)
    assert [r.run_id for r in loaded] == [r.run_id for r in built]
    assert loaded[0].ufs == ("AL",)
    assert loaded[0].health_seeds == ("mortality",)
