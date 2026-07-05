from __future__ import annotations

from pathlib import Path


def test_r_bridge_declares_utf8_sanitized_contract() -> None:
    script = Path("src/pegasus/datasus/r_scripts/fetch_process_microdatasus.R").read_text(encoding="utf-8")
    assert "sanitize_utf8_dataframe <- function" in script
    assert "write_utf8_parquet" in script
    # v4 bridge: canonical raw-coded parquet only (zstd), no raw.rds, microdatasus sidecar gated off.
    assert "datasus_r_bridge_v4_zstd_canonical_only" in script
    assert "utf8_sanitization" in script
    assert 'compression = "zstd"' in script
    # raw.rds is no longer serialized (it was never read back by PegaSUS).
    assert "saveRDS(raw, raw_path)" not in script
    # the microdatasus semantic sidecar is emitted only behind an explicit flag.
    assert "emit_sidecar" in script


def test_subprocess_cache_accepts_v3_legacy_and_v4() -> None:
    source = Path("src/pegasus/datasus/subprocess.py").read_text(encoding="utf-8")
    assert "DATASUS_BRIDGE_CONTRACT_VERSION" in source
    assert "ACCEPTED_DATASUS_CONTRACT_VERSIONS" in source
    assert "datasus_r_bridge_v4_zstd_canonical_only" in source
    # legacy v3 chunks stay cache-valid (their processed.parquet is schema-identical) after GC.
    assert "datasus_r_bridge_v3_utf8_sanitized_raw_canonical_plus_microdatasus_sidecar" in source
    assert "obsolete or non-UTF8-sanitized" in source
