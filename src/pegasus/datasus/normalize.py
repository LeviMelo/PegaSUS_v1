from __future__ import annotations

from pathlib import Path

from pegasus.datasus.adapters.registry import get_datasus_adapter
from pegasus.datasus.io import read_table, write_parquet


def normalize_datasus_table(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_system: str,
    source_manifest_hash: str = "",
) -> Path:
    df = read_table(input_path)
    adapter = get_datasus_adapter(source_system)

    result = adapter.normalize(
        df,
        source_manifest_hash=source_manifest_hash,
    )

    write_parquet(result.normalized, output_path)
    return Path(output_path)