from __future__ import annotations

from pathlib import Path

from pegasus.core.hashing import sha256_file
from pegasus.dashboard.read_only import bundle_overview
from pegasus.workflows.compile import run_compile


def test_release_dashboard_overview_is_bundle_bound_and_read_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run_dir,
        data_root=tmp_path / "data",
    )
    assert result["validation"].ok
    before = {path.relative_to(run_dir): sha256_file(path) for path in run_dir.rglob("*") if path.is_file()}
    overview = bundle_overview(run_dir=run_dir, limit=3)
    after = {path.relative_to(run_dir): sha256_file(path) for path in run_dir.rglob("*") if path.is_file()}
    assert overview["read_only"] is True
    assert overview["validation_ok"] is True
    assert overview["source_reality"]["compile_source_mode"] == "fixture_only"
    assert overview["registry_hashes"]
    assert overview["efg"]["fields"]["row_count"] > 0
    assert overview["efg"]["edges"]["row_count"] > 0
    assert before == after
