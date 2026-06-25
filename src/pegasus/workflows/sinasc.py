from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.datasus.sinasc_normalize import normalize_sinasc_events


def run_datasus_normalize_sinasc(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    return normalize_sinasc_events(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )


__all__ = ["run_datasus_normalize_sinasc"]
