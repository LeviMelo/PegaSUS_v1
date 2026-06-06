from pathlib import Path

from pegasus.datasus.manifests import (
    build_datasus_manifests,
    build_datasus_request_manifest,
    parse_years,
    read_request_manifest,
    write_request_manifest,
)


def test_parse_years_single_range_and_list():
    assert parse_years("2022") == [2022]
    assert parse_years("2020-2022") == [2020, 2021, 2022]
    assert parse_years("2020,2022") == [2020, 2022]


def test_build_sim_manifest_paths_are_content_addressed(tmp_path: Path):
    manifest = build_datasus_request_manifest(
        system="SIM-DO",
        uf="AL",
        year_start=2022,
        year_end=2022,
        config={"rscript_path": "Rscript"},
        data_root=tmp_path / "data",
    )

    assert manifest.system == "SIM-DO"
    assert manifest.uf == "AL"
    assert manifest.year_start == 2022
    assert manifest.raw_path.endswith(f"{manifest.request_hash}\\raw.rds") or manifest.raw_path.endswith(f"{manifest.request_hash}/raw.rds")
    assert "data" in manifest.raw_path
    assert "raw" in manifest.raw_path
    assert "processed" in manifest.processed_path


def test_build_yearly_manifests_for_range(tmp_path: Path):
    manifests = build_datasus_manifests(
        system="SIM-DO",
        uf="AL",
        years="2021-2022",
        config={"rscript_path": "Rscript"},
        data_root=tmp_path / "data",
    )
    assert [m.year_start for m in manifests] == [2021, 2022]
    assert len({m.request_hash for m in manifests}) == 2


def test_manifest_write_read_roundtrip(tmp_path: Path):
    manifest = build_datasus_request_manifest(
        system="SIM-DO",
        uf="AL",
        year_start=2022,
        year_end=2022,
        config={"rscript_path": "Rscript"},
        data_root=tmp_path / "data",
    )

    path = write_request_manifest(manifest, root=tmp_path / "manifests")
    loaded = read_request_manifest(path)

    assert loaded.request_hash == manifest.request_hash
    assert loaded.system == "SIM-DO"
