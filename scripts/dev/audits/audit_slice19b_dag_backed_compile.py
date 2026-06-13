from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.workflows.compile import run_compile


def main() -> int:
    Path(".tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="pegasus_slice19b_", dir=".tmp") as temp:
        root = Path(temp)
        run = root / "run"
        result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=run,
            data_root=root / "data",
        )
        metadata = result.get("autonomous_efg") or {}
        fields = pq.read_table(run / "V_fields.parquet").to_pylist()
        edges = pq.read_table(run / "E_DAG.parquet").to_pylist()
        checks = {
            "compile_valid": bool(result["validation"].ok),
            "autonomous_authority": metadata.get("graph_authority") == "autonomous_efg_core",
            "legacy_fields_preserved": any(row.get("name") == "SIMDeathsAll" for row in fields),
            "autonomous_edges_present": bool(edges),
            "manifest_present": (run / str(metadata.get("manifest_path", "missing"))).exists(),
            "exact_17_keys": set(path.name for path in run.iterdir()) == set(OUTPUT_BUNDLE_FILES.values()),
        }
        print(json.dumps({
            "slice": "19B",
            "checks": checks,
            "efg_id": metadata.get("efg_id"),
            "field_count": metadata.get("field_count"),
            "edge_count": metadata.get("edge_count"),
        }, indent=2, sort_keys=True))
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
