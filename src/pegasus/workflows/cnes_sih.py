from __future__ import annotations

from pathlib import Path

from pegasus.datasus.normalize import normalize_cnes_st_events
from pegasus.datasus.normalize import normalize_sih_rd_events


def run_datasus_normalize_cnes(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, object]:
    return normalize_cnes_st_events(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )


def run_datasus_normalize_sih(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, object]:
    return normalize_sih_rd_events(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )


__all__ = ["run_datasus_normalize_cnes", "run_datasus_normalize_sih"]
