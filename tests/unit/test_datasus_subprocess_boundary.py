from pathlib import Path

from pegasus.datasus.cache import DatasusCache
from pegasus.datasus.manifests import build_datasus_request_manifest
from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk


def test_fetch_datasus_chunk_blocks_when_rscript_missing(tmp_path: Path):
    manifest = build_datasus_request_manifest(
        system="SIM-DO",
        uf="AL",
        year_start=2022,
        year_end=2022,
        config={"rscript_path": "__definitely_missing_Rscript__"},
        data_root=tmp_path / "data",
    )

    result = fetch_datasus_chunk(
        manifest,
        config=DatasusConfig(
            rscript_path="__definitely_missing_Rscript__",
            r_timeout_seconds=1,
            heartbeat_timeout_seconds=1,
        ),
        cache=DatasusCache(tmp_path / "cache"),
        timeout_seconds=1,
        heartbeat_timeout_seconds=1,
    )

    assert result.status == "blocked"
    assert result.exit_code == 41
    assert "Rscript not found" in (result.error_message or "")
    assert result.duration_seconds >= 0
