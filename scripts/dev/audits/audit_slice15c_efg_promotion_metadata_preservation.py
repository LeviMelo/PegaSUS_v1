
from __future__ import annotations

import json
from pathlib import Path

from pegasus.efg.promotion_apply import planned_promotion_fields
from pegasus.efg.promotion_plan import materialized_fields_from_manifest, write_efg_promotion_plan
from pegasus.output.bundle import create_empty_output_bundle


def main() -> int:
    errors: list[str] = []
    field = {
        "field_id": "efg_substrate__SIM_DO__underlying_icd_norm",
        "name": "SIM-DO.underlying_icd_norm",
        "kind": "observer_proxy",
        "carrier": "Deaths",
        "unit": "ICD10",
        "aggregation": "non_aggregable",
        "role": ["diagnostic_topology", "source_field", "substrate_materialized"],
        "source": ["SIM-DO"],
        "support": {"column": "underlying_icd_norm", "row_count": 3, "source_system": "SIM-DO"},
        "axes": {"diagnosis": "icd10"},
        "operator": "metadata_only_substrate_materialization",
        "provenance": ["registry", "fixture"],
        "warnings": [],
        "lineage_hash": "lineage-demo",
        "registry_hash": "registry-demo",
        "materialization_state": "metadata_only",
        "dashboard_safe": "False",
    }
    payload = {"fields": [{"candidate_id": "candidate_underlying_icd_norm", "field": field}]}
    fields = materialized_fields_from_manifest(payload)
    if not fields or fields[0].get("support", {}).get("column") != "underlying_icd_norm":
        errors.append("materialized_fields_from_manifest does not flatten Slice 14B nested field wrappers")

    tmp = Path(".codecontext") / "slice15c_audit_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    run_dir = tmp / "run"
    manifest = tmp / "materialization.json"
    create_empty_output_bundle(run_dir)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = write_efg_promotion_plan(run_dir=run_dir, materialization_manifest=manifest)
    planned = result["plan"].get("planned_promotions", [])
    if not planned or not isinstance(planned[0].get("field"), dict):
        errors.append("promotion plan does not preserve full field payload under planned_promotions[].field")
    promoted, skipped, blocked = planned_promotion_fields(result["plan"])
    if skipped or blocked or not promoted:
        errors.append("promotion_apply cannot read planned full-field promotions")
    elif promoted[0].get("support", {}).get("column") != "underlying_icd_norm":
        errors.append("promotion_apply lost support metadata from promotion plan")

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, "planned_count": len(planned)}, indent=2, sort_keys=True))
    print("AUDIT PASSED: Slice 15C EFG promotion metadata preservation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
