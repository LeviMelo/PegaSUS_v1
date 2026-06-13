from __future__ import annotations

import ast
import py_compile
import re
import sys
import zipfile
from pathlib import Path
from textwrap import dedent

SLICE = "15C"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/efg/promotion_plan.py",
    "tests/unit/test_slice15c_efg_promotion_metadata_preservation.py",
    "tests/integration/test_slice15c_promotion_plan_apply_chain.py",
    "scripts/dev/audits/audit_slice15c_efg_promotion_metadata_preservation.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/source_registry.py",
    "src/pegasus/she/substrate.py",
    "src/pegasus/output/",
    "src/pegasus/pirs/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)


def fail(message: str) -> None:
    raise SystemExit(f"[slice15c] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_defs(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    for path in (
        "src/pegasus/efg/promotion_plan.py",
        "src/pegasus/efg/promotion_apply.py",
        "src/pegasus/workflows/efg_promotion.py",
        "src/pegasus/workflows/efg_apply.py",
        "src/pegasus/workflows/compile.py",
    ):
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_defs(compile_text, "run_compile") != 1 or top_level_defs(compile_text, "_run_compile_impl") != 1:
        fail("compile.py is not in the consolidated Slice 13C shape")
    plan_text = read("src/pegasus/efg/promotion_plan.py")
    for token in ("PromotionDecision", "materialized_fields_from_manifest", "build_efg_promotion_plan"):
        if token not in plan_text:
            fail(f"promotion_plan.py missing expected API: {token}")
    apply_text = read("src/pegasus/efg/promotion_apply.py")
    for token in ("planned_promotion_fields", "v_field_row", "apply_efg_promotion_plan_to_run"):
        if token not in apply_text:
            fail(f"promotion_apply.py missing expected API: {token}")


def patch_promotion_plan() -> None:
    path = "src/pegasus/efg/promotion_plan.py"
    text = read(path)

    if "field: dict[str, Any] | None = None" not in text:
        anchor = "    source_manifest_index: int | None = None\n"
        if anchor not in text:
            fail("PromotionDecision source_manifest_index anchor not found")
        text = text.replace(anchor, anchor + "    field: dict[str, Any] | None = None\n", 1)

    if '"field": dict(self.field) if self.field is not None else None,' not in text:
        anchor = '            "source_manifest_index": self.source_manifest_index,\n'
        if anchor not in text:
            fail("PromotionDecision.as_manifest source_manifest_index anchor not found")
        text = text.replace(
            anchor,
            anchor + '            "field": dict(self.field) if self.field is not None else None,\n',
            1,
        )

    replacement = dedent('''
    def materialized_fields_from_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract materialized FieldNode-like rows from Slice 14B manifest shapes.

        Slice 14B serializes materialized entries as wrappers of the form
        ``{"candidate_id": ..., "field": {...}}``.  Promotion planning must
        preserve the full nested FieldNode payload rather than collapsing to the
        small PromotionDecision summary.  Older tests and manual manifests may
        still provide direct field dictionaries; those remain accepted.
        """

        fields = _first_list(
            payload,
            (
                "materialized_fields",
                "field_nodes",
                "fields",
                "nodes",
                "materialized_field_nodes",
            ),
        )
        valid: list[dict[str, Any]] = []
        for item in fields:
            direct = dict(item)
            nested = None
            for key in ("field", "field_node", "materialized_field", "v_field"):
                value = item.get(key)
                if isinstance(value, dict):
                    nested = dict(value)
                    break
            if nested is not None:
                if item.get("candidate_id") is not None:
                    nested.setdefault("candidate_id", item.get("candidate_id"))
                if item.get("lineage_hash") is not None:
                    nested.setdefault("lineage_hash", item.get("lineage_hash"))
                if item.get("materialization_reason") is not None:
                    nested.setdefault("materialization_reason", item.get("materialization_reason"))
                if "field_id" in nested or "id" in nested or "name" in nested:
                    valid.append(nested)
                continue
            if "field_id" in direct or "id" in direct or "name" in direct:
                valid.append(direct)
        return valid
    ''')
    pattern = re.compile(
        r"def materialized_fields_from_manifest\(payload: dict\[str, Any\]\) -> list\[dict\[str, Any\]\]:\n.*?\n\ndef excluded_fields_from_manifest",
        re.DOTALL,
    )
    text, count = pattern.subn(replacement.rstrip() + "\n\ndef excluded_fields_from_manifest", text, count=1)
    if count != 1:
        fail("could not replace materialized_fields_from_manifest")

    old = """            source_manifest_index=index,\n        )\n"""
    new = """            source_manifest_index=index,\n            field=dict(field),\n        )\n"""
    if old in text and "field=dict(field)" not in text:
        text = text.replace(old, new, 1)
    elif "field=dict(field)" not in text:
        fail("could not install field=dict(field) in PromotionDecision construction")

    write(path, text)


def write_tests() -> None:
    unit = r'''
from __future__ import annotations

import json
from pathlib import Path

from pegasus.efg.promotion_apply import planned_promotion_fields
from pegasus.efg.promotion_plan import build_efg_promotion_plan, materialized_fields_from_manifest, write_efg_promotion_plan
from pegasus.output.bundle import create_empty_output_bundle


def _field() -> dict:
    return {
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


def test_slice15c_materialized_fields_accepts_slice14b_nested_shape() -> None:
    payload = {
        "fields": [
            {
                "candidate_id": "candidate_underlying_icd_norm",
                "field": _field(),
                "lineage_hash": "wrapper-lineage",
                "materialization_reason": "metadata_only",
            }
        ]
    }
    fields = materialized_fields_from_manifest(payload)
    assert len(fields) == 1
    assert fields[0]["field_id"] == "efg_substrate__SIM_DO__underlying_icd_norm"
    assert fields[0]["support"]["column"] == "underlying_icd_norm"
    assert fields[0]["candidate_id"] == "candidate_underlying_icd_norm"


def test_slice15c_promotion_plan_preserves_full_field_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    manifest = tmp_path / "efg_substrate_materialization.json"
    manifest.write_text(
        json.dumps({"fields": [{"candidate_id": "candidate_underlying_icd_norm", "field": _field()}]}),
        encoding="utf-8",
    )

    result = write_efg_promotion_plan(run_dir=run_dir, materialization_manifest=manifest)
    planned = result["plan"]["planned_promotions"]
    assert len(planned) == 1
    assert planned[0]["field"]["support"]["column"] == "underlying_icd_norm"
    assert planned[0]["field"]["role"] == ["diagnostic_topology", "source_field", "substrate_materialized"]

    promoted, skipped, blocked = planned_promotion_fields(result["plan"])
    assert not skipped
    assert not blocked
    assert promoted[0]["support"]["column"] == "underlying_icd_norm"
    assert promoted[0]["axes"]["diagnosis"] == "icd10"
'''

    integration = r'''
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run
from pegasus.efg.promotion_plan import write_efg_promotion_plan
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle


def _field() -> dict:
    return {
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


def test_slice15c_real_promotion_apply_chain_preserves_support_and_axes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    materialization = tmp_path / "efg_substrate_materialization.json"
    materialization.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact": "efg_substrate_materialization",
                "fields": [{"candidate_id": "candidate_underlying_icd_norm", "field": _field()}],
                "excluded_source_fields": [{"column": "constant_col", "efg_materialized": False}],
            }
        ),
        encoding="utf-8",
    )

    plan_result = write_efg_promotion_plan(run_dir=run_dir, materialization_manifest=materialization)
    apply_result = apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_result["path"])
    assert apply_result.promoted_field_count == 1
    assert validate_output_bundle(run_dir=str(run_dir)).ok

    vd = pq.read_table(run_dir / "VariableDictionary.parquet").to_pylist()
    row = next(item for item in vd if item["field_id"] == "efg_substrate__SIM_DO__underlying_icd_norm")
    assert "underlying_icd_norm" in row["support_description"]
    assert "icd10" in row["axis_description"]
'''
    write("tests/unit/test_slice15c_efg_promotion_metadata_preservation.py", unit)
    write("tests/integration/test_slice15c_promotion_plan_apply_chain.py", integration)


def write_audit() -> None:
    audit = r'''
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
'''
    write("scripts/dev/audits/audit_slice15c_efg_promotion_metadata_preservation.py", audit)


def self_validate() -> None:
    for path in TOUCH_LIST:
        if path.endswith(".py"):
            py_compile.compile(str(ROOT / path), doraise=True)
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pegasus.efg.promotion_plan import materialized_fields_from_manifest
    payload = {"fields": [{"candidate_id": "c", "field": {"field_id": "f", "support": {"column": "x"}, "materialization_state": "metadata_only"}}]}
    fields = materialized_fields_from_manifest(payload)
    if not fields or fields[0].get("support", {}).get("column") != "x":
        fail("self-validate: nested field wrapper was not flattened")


def main() -> None:
    preflight()
    patch_promotion_plan()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 15C updater applied: EFG promotion metadata preservation hardened.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
