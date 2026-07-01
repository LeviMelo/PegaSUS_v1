from __future__ import annotations

import ast
import json
import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

SLICE = "15A"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/efg/promotion_plan.py",
    "src/pegasus/workflows/efg_promotion.py",
    "tests/unit/test_slice15a_efg_promotion_plan.py",
    "tests/integration/test_slice15a_attach_efg_promotion_plan_to_run.py",
    "scripts/dev/audits/audit_slice15a_efg_promotion_plan.py",
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
    raise SystemExit(f"[slice15a] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    for prefix in FORBIDDEN_PREFIXES:
        if path.startswith(prefix):
            fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_function_count(path: str, name: str) -> int:
    tree = ast.parse(read(path))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = (
        "src/pegasus/efg/materialize.py",
        "src/pegasus/efg/materialization_manifest.py",
        "src/pegasus/workflows/efg_materialize.py",
        "src/pegasus/workflows/compile.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_function_count("src/pegasus/workflows/compile.py", "run_compile") != 1:
        fail("compile.py must contain exactly one public run_compile before Slice 15A")
    if top_level_function_count("src/pegasus/workflows/compile.py", "_run_compile_impl") != 1:
        fail("compile.py must contain exactly one private _run_compile_impl before Slice 15A")
    if "efg_materialization_gate" not in read("src/pegasus/efg/materialization_manifest.py"):
        fail("Slice 14B materialization manifest attach API is not present")
    if "inspect-materialization" not in read("src/pegasus/cli.py"):
        fail("Slice 14C CLI boundary is not present")


def write_promotion_plan_module() -> None:
    module = r'''
"""EFG promotion planning for materialized substrate fields.

Slice 15A is intentionally non-mutating with respect to V_fields, E_DAG, and
Q_tensor.  It reads the Slice 14B EFG substrate-materialization manifest and the
existing run bundle, then writes an auditable promotion plan.  Promotion means
"eligible for a future controlled V_fields insertion", not actual insertion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl


class EFGPromotionPlanError(ValueError):
    """Raised when a materialization manifest cannot be promoted safely."""


@dataclass(frozen=True)
class PromotionDecision:
    field_id: str
    name: str
    column: str | None
    carrier: str | None
    unit: str | None
    kind: str | None
    dashboard_safe: str | None
    materialization_state: str | None
    status: str
    reason: str
    source_manifest_index: int | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "name": self.name,
            "column": self.column,
            "carrier": self.carrier,
            "unit": self.unit,
            "kind": self.kind,
            "dashboard_safe": self.dashboard_safe,
            "materialization_state": self.materialization_state,
            "status": self.status,
            "reason": self.reason,
            "source_manifest_index": self.source_manifest_index,
        }


@dataclass(frozen=True)
class EFGPromotionPlan:
    materialization_manifest_path: str
    run_dir: str
    planned: tuple[PromotionDecision, ...]
    conflicts: tuple[PromotionDecision, ...]
    excluded_source_fields: tuple[dict[str, Any], ...]
    existing_field_count: int
    existing_field_ids: tuple[str, ...]
    existing_field_names: tuple[str, ...]

    @property
    def planned_count(self) -> int:
        return len(self.planned)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)

    @property
    def excluded_source_field_count(self) -> int:
        return len(self.excluded_source_fields)

    @property
    def promotion_safe(self) -> bool:
        return self.planned_count > 0 and self.conflict_count == 0

    def summary(self, *, plan_path: str | None = None) -> dict[str, Any]:
        return {
            "status": "planned" if self.promotion_safe else "blocked",
            "promotion_safe": self.promotion_safe,
            "planned_count": self.planned_count,
            "conflict_count": self.conflict_count,
            "excluded_source_field_count": self.excluded_source_field_count,
            "existing_field_count": self.existing_field_count,
            "materialization_manifest_path": self.materialization_manifest_path,
            "promotion_plan_path": plan_path,
            "non_mutating": True,
            "writes_v_fields": False,
            "writes_e_dag": False,
            "writes_q_tensor": False,
        }

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "artifact": "efg_promotion_plan",
            "slice": "15A",
            "materialization_manifest_path": self.materialization_manifest_path,
            "run_dir": self.run_dir,
            "promotion_safe": self.promotion_safe,
            "existing_field_count": self.existing_field_count,
            "existing_field_ids": list(self.existing_field_ids),
            "existing_field_names": list(self.existing_field_names),
            "planned_count": self.planned_count,
            "conflict_count": self.conflict_count,
            "excluded_source_field_count": self.excluded_source_field_count,
            "planned_promotions": [decision.as_manifest() for decision in self.planned],
            "conflicts": [decision.as_manifest() for decision in self.conflicts],
            "excluded_source_fields": list(self.excluded_source_fields),
            "contract": {
                "non_mutating": True,
                "writes_v_fields": False,
                "writes_e_dag": False,
                "writes_q_tensor": False,
                "future_stage": "controlled_v_fields_insertion",
            },
        }


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise EFGPromotionPlanError(f"JSON manifest does not exist: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EFGPromotionPlanError(f"JSON manifest must be an object: {p}")
    return payload


def _iter_nested_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_nested_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _first_list(payload: dict[str, Any], names: tuple[str, ...]) -> list[dict[str, Any]]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for node in _iter_nested_dicts(payload):
        for name in names:
            value = node.get(name)
            if isinstance(value, list):
                items = [item for item in value if isinstance(item, dict)]
                if items:
                    return items
    return []


def materialized_fields_from_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract materialized FieldNode-like rows from known Slice 14B manifest shapes."""

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
    for field in fields:
        if "field_id" in field or "id" in field or "name" in field:
            valid.append(dict(field))
    return valid


def excluded_fields_from_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _first_list(
        payload,
        (
            "excluded_source_fields",
            "excluded_fields",
            "exclusions",
            "source_exclusions",
        ),
    )


def _support_column(field: dict[str, Any]) -> str | None:
    support = field.get("support")
    if isinstance(support, str):
        try:
            support = json.loads(support)
        except json.JSONDecodeError:
            support = {}
    if isinstance(support, dict):
        value = support.get("column") or support.get("source_column") or support.get("column_name")
        if value is not None:
            return str(value)
    for key in ("column", "source_column", "column_name"):
        value = field.get(key)
        if value is not None:
            return str(value)
    return None


def _json_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _existing_v_fields(run_dir: str | Path) -> tuple[set[str], set[str], int]:
    path = Path(run_dir) / "V_fields.parquet"
    if not path.exists():
        return set(), set(), 0
    df = pl.read_parquet(path)
    ids: set[str] = set()
    names: set[str] = set()
    if "field_id" in df.columns:
        ids.update(str(v) for v in df.get_column("field_id").drop_nulls().to_list())
    if "name" in df.columns:
        names.update(str(v) for v in df.get_column("name").drop_nulls().to_list())
    return ids, names, df.height


def _field_id(field: dict[str, Any], index: int) -> str:
    value = field.get("field_id") or field.get("id")
    if value:
        return str(value)
    column = _support_column(field) or f"field_{index}"
    return f"efg_substrate::{column}"


def _field_name(field: dict[str, Any], field_id: str) -> str:
    value = field.get("name") or field.get("label")
    return str(value) if value is not None else field_id


def _text_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def build_efg_promotion_plan(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path,
) -> EFGPromotionPlan:
    payload = _load_json(materialization_manifest)
    fields = materialized_fields_from_manifest(payload)
    if not fields:
        raise EFGPromotionPlanError("materialization manifest does not contain materialized fields")

    excluded = tuple(dict(item, efg_materialized=False) for item in excluded_fields_from_manifest(payload))
    existing_ids, existing_names, existing_count = _existing_v_fields(run_dir)

    planned: list[PromotionDecision] = []
    conflicts: list[PromotionDecision] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    for index, field in enumerate(fields):
        field_id = _field_id(field, index)
        name = _field_name(field, field_id)
        column = _support_column(field)
        support = _json_cell(field.get("support"))
        carrier = _text_or_none(field.get("carrier"))
        unit = _text_or_none(field.get("unit"))
        kind = _text_or_none(field.get("kind"))
        dashboard_safe = _text_or_none(field.get("dashboard_safe"))
        materialization_state = _text_or_none(field.get("materialization_state"))

        reason = "eligible_for_future_v_fields_insertion"
        status = "planned"
        if field_id in existing_ids:
            status = "conflict"
            reason = "field_id_already_present_in_v_fields"
        elif name in existing_names:
            status = "conflict"
            reason = "field_name_already_present_in_v_fields"
        elif field_id in seen_ids:
            status = "conflict"
            reason = "duplicate_materialized_field_id"
        elif name in seen_names:
            status = "conflict"
            reason = "duplicate_materialized_field_name"
        elif materialization_state and materialization_state != "metadata_only":
            status = "conflict"
            reason = f"unsupported_materialization_state:{materialization_state}"
        elif support and support.get("efg_materialized") is False:
            status = "conflict"
            reason = "field_support_marks_efg_materialized_false"

        decision = PromotionDecision(
            field_id=field_id,
            name=name,
            column=column,
            carrier=carrier,
            unit=unit,
            kind=kind,
            dashboard_safe=dashboard_safe,
            materialization_state=materialization_state,
            status=status,
            reason=reason,
            source_manifest_index=index,
        )
        if status == "planned":
            planned.append(decision)
        else:
            conflicts.append(decision)
        seen_ids.add(field_id)
        seen_names.add(name)

    return EFGPromotionPlan(
        materialization_manifest_path=str(Path(materialization_manifest)),
        run_dir=str(Path(run_dir)),
        planned=tuple(planned),
        conflicts=tuple(conflicts),
        excluded_source_fields=excluded,
        existing_field_count=existing_count,
        existing_field_ids=tuple(sorted(existing_ids)),
        existing_field_names=tuple(sorted(existing_names)),
    )


def write_efg_promotion_plan(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    plan = build_efg_promotion_plan(run_dir=run_dir, materialization_manifest=materialization_manifest)
    output_path = Path(output) if output is not None else Path(run_dir) / "Tables" / "efg_promotion_plan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.as_manifest(), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return {
        "plan": plan.as_manifest(),
        "summary": plan.summary(plan_path=str(output_path)),
        "path": str(output_path),
    }


def _update_json(path: Path, key: str, value: dict[str, Any]) -> None:
    payload: dict[str, Any]
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}
    payload[key] = value
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def attach_efg_promotion_plan_to_run(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = Path(materialization_manifest) if materialization_manifest is not None else root / "Tables" / "efg_substrate_materialization.json"
    result = write_efg_promotion_plan(run_dir=root, materialization_manifest=manifest_path)
    summary = dict(result["summary"])
    for relative in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        _update_json(root / relative, "efg_promotion_gate", summary)
    return summary
'''
    write("src/pegasus/efg/promotion_plan.py", module)


def write_workflow_module() -> None:
    module = r'''
"""Workflow wrappers for Slice 15A EFG promotion planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.efg.promotion_plan import attach_efg_promotion_plan_to_run, write_efg_promotion_plan


def run_plan_efg_promotion(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return write_efg_promotion_plan(run_dir=run_dir, materialization_manifest=materialization_manifest, output=output)


def run_attach_efg_promotion_plan_to_run(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path | None = None,
) -> dict[str, Any]:
    return attach_efg_promotion_plan_to_run(run_dir=run_dir, materialization_manifest=materialization_manifest)
'''
    write("src/pegasus/workflows/efg_promotion.py", module)


def write_tests() -> None:
    unit_test = r'''
from __future__ import annotations

import json
from pathlib import Path

from pegasus.efg.promotion_plan import build_efg_promotion_plan, materialized_fields_from_manifest


def _manifest() -> dict:
    return {
        "schema_version": "1.0",
        "materialized_fields": [
            {
                "field_id": "efg_substrate::age_years",
                "name": "SIM-DO.age_years",
                "kind": "extensive_measure",
                "carrier": "Deaths",
                "unit": "years",
                "dashboard_safe": "true",
                "materialization_state": "metadata_only",
                "support": {"column": "age_years"},
            },
            {
                "field_id": "efg_substrate::underlying_icd_norm",
                "name": "SIM-DO.underlying_icd_norm",
                "kind": "observer_proxy",
                "carrier": "Deaths",
                "unit": "ICD10",
                "dashboard_safe": "warning",
                "materialization_state": "metadata_only",
                "support": {"column": "underlying_icd_norm"},
            },
        ],
        "excluded_source_fields": [
            {"column": "constant_col", "reason": "zero_variance", "efg_materialized": False}
        ],
    }


def test_slice15a_extracts_materialized_fields_from_manifest() -> None:
    fields = materialized_fields_from_manifest(_manifest())
    assert len(fields) == 2
    assert fields[1]["unit"] == "ICD10"


def test_slice15a_builds_non_mutating_promotion_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "Tables").mkdir(parents=True)
    manifest_path = run_dir / "Tables" / "efg_substrate_materialization.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    plan = build_efg_promotion_plan(run_dir=run_dir, materialization_manifest=manifest_path)

    assert plan.planned_count == 2
    assert plan.conflict_count == 0
    assert plan.excluded_source_field_count == 1
    assert plan.promotion_safe is True
    payload = plan.as_manifest()
    assert payload["contract"]["non_mutating"] is True
    assert payload["contract"]["writes_v_fields"] is False
    assert payload["planned_promotions"][1]["kind"] == "observer_proxy"
    assert payload["excluded_source_fields"][0]["efg_materialized"] is False
'''
    integration_test = r'''
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.construct.efg_promotion import run_attach_efg_promotion_plan_to_run


def test_slice15a_attaches_efg_promotion_plan_without_mutating_first_class_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    materialization_path = run_dir / "Tables" / "efg_substrate_materialization.json"
    materialization_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "materialized_fields": [
                    {
                        "field_id": "efg_substrate::underlying_icd_norm",
                        "name": "SIM-DO.underlying_icd_norm",
                        "kind": "observer_proxy",
                        "carrier": "Deaths",
                        "unit": "ICD10",
                        "dashboard_safe": "warning",
                        "materialization_state": "metadata_only",
                        "support": {"column": "underlying_icd_norm"},
                    }
                ],
                "excluded_source_fields": [
                    {"column": "constant_col", "reason": "zero_variance", "efg_materialized": False}
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    before = {p.name for p in run_dir.iterdir()}
    summary = run_attach_efg_promotion_plan_to_run(run_dir=run_dir)
    after = {p.name for p in run_dir.iterdir()}

    assert before == after
    assert summary["status"] == "planned"
    assert summary["planned_count"] == 1
    assert summary["conflict_count"] == 0
    assert summary["writes_v_fields"] is False
    assert (run_dir / "Tables" / "efg_promotion_plan.json").exists()

    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["efg_promotion_gate"]["planned_count"] == 1
    assert validate_output_bundle(run_dir=str(run_dir)).ok
'''
    write("tests/unit/test_slice15a_efg_promotion_plan.py", unit_test)
    write("tests/integration/test_slice15a_attach_efg_promotion_plan_to_run.py", integration_test)


def write_audit() -> None:
    audit = r'''
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.construct.efg_promotion import run_attach_efg_promotion_plan_to_run


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
        errors.append("compile.py must contain exactly one public run_compile")
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "_run_compile_impl") != 1:
        errors.append("compile.py must contain exactly one _run_compile_impl")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "run"
        create_empty_output_bundle(run_dir)
        materialization_path = run_dir / "Tables" / "efg_substrate_materialization.json"
        materialization_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "materialized_fields": [
                        {
                            "field_id": "efg_substrate::underlying_icd_norm",
                            "name": "SIM-DO.underlying_icd_norm",
                            "kind": "observer_proxy",
                            "carrier": "Deaths",
                            "unit": "ICD10",
                            "dashboard_safe": "warning",
                            "materialization_state": "metadata_only",
                            "support": {"column": "underlying_icd_norm"},
                        }
                    ],
                    "excluded_source_fields": [
                        {"column": "constant_col", "reason": "zero_variance", "efg_materialized": False}
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        before = {p.name for p in run_dir.iterdir()}
        summary = run_attach_efg_promotion_plan_to_run(run_dir=run_dir)
        after = {p.name for p in run_dir.iterdir()}
        if before != after:
            errors.append("EFG promotion attach changed first-class run keys")
        if summary.get("writes_v_fields") is not False:
            errors.append("EFG promotion summary must state writes_v_fields=False")
        if summary.get("planned_count") != 1:
            errors.append("EFG promotion planned_count did not equal 1")
        if not (run_dir / "Tables" / "efg_promotion_plan.json").exists():
            errors.append("EFG promotion plan was not written under Tables")
        if not validate_output_bundle(run_dir=str(run_dir)).ok:
            errors.append("EFG promotion attach invalidated output bundle")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 15A EFG promotion plan boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write("scripts/dev/audits/audit_slice15a_efg_promotion_plan.py", audit)


def self_validate() -> None:
    for path in TOUCH_LIST:
        if path.endswith(".py"):
            py_compile.compile(str(ROOT / path), doraise=True)
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from pegasus.efg.promotion_plan import build_efg_promotion_plan  # noqa: F401
    from pegasus.workflows.construct.efg_promotion import run_attach_efg_promotion_plan_to_run  # noqa: F401


def main() -> None:
    preflight()
    write_promotion_plan_module()
    write_workflow_module()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 15A updater applied: EFG promotion plan boundary added.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
