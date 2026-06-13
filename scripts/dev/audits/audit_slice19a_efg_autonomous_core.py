from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from pegasus.efg.dag import build_efg
from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle


def main() -> int:
    with TemporaryDirectory(prefix="pegasus_slice19a_") as temp:
        artifact = Path(temp) / "sim_processed.parquet"
        pl.DataFrame({
            "year": [2021, 2022, 2022],
            "mun_residence_cod6": ["270430", "270430", "270430"],
            "underlying_icd_norm": ["A00", "B20", "A00"],
            "event_id": ["a", "b", "c"],
            "constant_unknown": [1, 1, 1],
        }).write_parquet(artifact)
        substrate = build_substrate_bundle(artifacts=[SourceArtifactRef(
            path=str(artifact),
            source_system="SIM-DO",
            provenance_mode="fixture",
            source_manifest_hash="audit-manifest",
        )])
        result = build_efg(substrate=substrate)
        diagnostic = [field for field in result.fields if field.unit == "ICD10"]
        excluded_columns = {
            branch.support_axis_mismatch.get("column") for branch in result.failed_branches
        }
        checks = {
            "count_measure_emitted": any(field.name.endswith(".count") for field in result.fields),
            "dag_edges_emitted": bool(result.edges),
            "diagnostic_observer_preserved": bool(diagnostic) and all(
                field.kind == "observer_proxy" for field in diagnostic
            ),
            "constant_excluded": "constant_unknown" in excluded_columns,
            "promotion_handoff_present": bool(result.as_manifest()["fields"]),
        }
        print(json.dumps({
            "slice": "19A",
            "checks": checks,
            "efg_id": result.efg_id,
            "field_count": result.field_count,
            "edge_count": result.edge_count,
            "failed_branch_count": len(result.failed_branches),
        }, indent=2, sort_keys=True))
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
