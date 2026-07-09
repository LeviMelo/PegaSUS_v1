"""Export a materialized dataset — the frame (parquet/csv) + a provenance sidecar (FEAT-P3).

Nothing is written without its provenance manifest (§VIII coverage contract): the query, the run
identity, and — for rates — the resolved denominator + reference population land in the sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.query.engine import MaterializedDataset, QueryError
from pegasus.output.query.spec import OUTPUT_FORMATS


def write_dataset(dataset: MaterializedDataset, out_dir: str | Path, *, name: str) -> dict[str, Path]:
    """Write ``dataset.frame`` as ``<name>.{parquet|csv}`` and ``<name>.provenance.json`` under
    ``out_dir``. Format is read from the recorded QuerySpec (``fmt``), default parquet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = str(dataset.provenance.get("query", {}).get("fmt", "parquet"))
    if fmt not in OUTPUT_FORMATS:
        raise QueryError(f"unsupported export format {fmt!r}; allowed: {OUTPUT_FORMATS}")
    if fmt == "csv":
        data_path = out / f"{name}.csv"
        dataset.frame.write_csv(data_path)
    else:
        data_path = out / f"{name}.parquet"
        dataset.frame.write_parquet(data_path, compression="zstd")
    provenance_path = out / f"{name}.provenance.json"
    provenance_path.write_text(
        json.dumps(dataset.provenance, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"data": data_path, "provenance": provenance_path}


__all__ = ["write_dataset"]
