from __future__ import annotations

from pathlib import Path


def test_r_bridge_declares_utf8_sanitized_contract() -> None:
    script = Path("src/pegasus/datasus/r_scripts/fetch_process_microdatasus.R").read_text(encoding="utf-8")
    assert "sanitize_utf8_dataframe <- function" in script
    assert "write_utf8_parquet" in script
    assert "datasus_r_bridge_v3_utf8_sanitized_raw_canonical_plus_microdatasus_sidecar" in script
    assert "utf8_sanitization" in script


def test_subprocess_cache_rejects_pre_utf8_bridge_contracts() -> None:
    source = Path("src/pegasus/datasus/subprocess.py").read_text(encoding="utf-8")
    assert "DATASUS_BRIDGE_CONTRACT_VERSION" in source
    assert "datasus_r_bridge_v3_utf8_sanitized_raw_canonical_plus_microdatasus_sidecar" in source
    assert "obsolete or non-UTF8-sanitized" in source
