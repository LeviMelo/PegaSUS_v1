from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DAG = ROOT / "src/pegasus/efg/dag.py"


def run_audit() -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    if not DAG.exists():
        errors.append("src/pegasus/efg/dag.py missing")
        text = ""
    else:
        text = DAG.read_text(encoding="utf-8")

    required = {
        "_build_efg_base": "base build_efg preserved under _build_efg_base",
        "def build_efg(*args, **kwargs):": "public build_efg wrapper present",
        "core_seed_summary": "core seed manifest evidence emitted",
        "bridge_plan_summary": "bridge plan manifest evidence emitted",
        "semantic_manifest_schema": "semantic manifest schema marker emitted",
        "plan_bridge_candidates": "bridge planner imported/used",
        "build_core_seed_set": "core seed planner imported/used",
    }
    for needle, label in required.items():
        if needle not in text:
            errors.append(f"missing semantic DAG wiring: {label}")

    for rel in ("src/pegasus/efg/core_seed.py", "src/pegasus/efg/bridges.py"):
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing semantic module: {rel}")
            continue
        module_text = path.read_text(encoding="utf-8")
        if "slice0_scaffold_only" in module_text or "BlockedModuleError" in module_text:
            errors.append(f"semantic module remains scaffold blocked: {rel}")

    return {
        "audit": "slice28y_efg_semantic_manifest",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    result = run_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
