from __future__ import annotations

import json

from pegasus.efg.equivalence import precompress_fields
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node


def _projection(name: str, source_hash: str):
    lineage = make_lineage(
        parent_ids=["parent"],
        operator_type="pi_*",
        operator_params={"drop_axes": ["sex"]},
        registry_versions={"registry": "v1"},
        source_manifest_hashes=[source_hash],
        code_version="slice20a",
    )
    return make_field_node(
        name=name,
        kind="extensive_measure",
        carrier="Deaths",
        unit="counts",
        support={"artifact_path": "same.parquet", "column": name},
        axes={"time": "year"},
        aggregation="additive",
        role=["projected"],
        source=["SIM-DO", "same.parquet", name],
        operator="pi_*",
        provenance=["source_normalized"],
        state="verified",
        warnings=[],
        lineage=lineage,
        materialization_state="planned",
    )


def main() -> int:
    compressed, report = precompress_fields([
        _projection("projection-a", "hash-a"),
        _projection("projection-b", "hash-b"),
    ])
    payload = report.as_manifest()
    checks = {
        "projection_duplicate_suppressed": len(compressed) == 1 and report.suppressed_count == 1,
        "proof_recorded": bool(payload["suppressed"][0]["proof"]),
        "metadata_only": payload["uses_numerical_arrays"] is False,
        "class_count_recorded": payload["equivalence_class_counts"].get("exact_projection_redundancy") == 1,
    }
    print(json.dumps({"slice": "20A", "checks": checks, "report": payload}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
