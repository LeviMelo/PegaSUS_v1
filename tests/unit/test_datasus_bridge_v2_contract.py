from __future__ import annotations

from pathlib import Path

from pegasus.datasus.manifests import _request_identity
from pegasus.datasus.subprocess import DatasusConfig

MANIFEST_CONTRACT = "datasus_r_bridge_v2_raw_canonical_plus_microdatasus_sidecar"
R_BRIDGE_CONTRACT = "datasus_r_bridge_v4_zstd_canonical_only"


def test_datasus_config_from_mapping_preserves_local_r_library() -> None:
    config = DatasusConfig.from_mapping({"r_library_path": ".r-library"})
    assert config.r_library_path == ".r-library"


def test_request_identity_uses_pegasus_dispatch_and_bridge_contract() -> None:
    identity = _request_identity(
        system="SIM-DO",
        uf="AL",
        year_start=2022,
        year_end=2022,
        month_start=None,
        month_end=None,
        config={"backend": "microdatasus"},
    )

    assert identity["fetch_function"] == "fetch_datasus"
    assert identity["process_function"] == "pegasus_process_datasus_dispatch"
    assert identity["processing_contract_version"] == MANIFEST_CONTRACT


def test_r_bridge_restores_microdatasus_dispatch_without_removed_api() -> None:
    script = Path("src/pegasus/datasus/r_scripts/fetch_process_microdatasus.R").read_text(encoding="utf-8")

    assert "library(microdatasus)" in script
    assert "process_datasus_dispatch" in script
    assert "process_sim" in script
    assert "process_sinasc" in script
    assert "microdatasus::process_datasus" not in script
    assert "read.dbc::read.dbc(dbc_path, as.is = TRUE)" in script or "read.dbc(dbc_path, as.is = TRUE)" in script
    assert R_BRIDGE_CONTRACT in script


def test_python_subprocess_passes_timeout_and_rejects_obsolete_cached_manifests() -> None:
    source = Path("src/pegasus/datasus/subprocess.py").read_text(encoding="utf-8")

    assert "\"--timeout-seconds\"" in source
    assert "str(timeout_seconds)" in source
    assert "cached R manifest uses an obsolete or non-UTF8-sanitized DATASUS bridge contract" in source
    assert R_BRIDGE_CONTRACT in source
