from __future__ import annotations

from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()
UPDATER_NAME = "apply_slice28y_efg_semantic_manifest_integration.py"


def _write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dedent(text).lstrip(), encoding="utf-8")
    print(f"wrote {path}")


def _append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        print(f"unchanged {path} ({marker})")
        return
    target.write_text(text.rstrip() + "\n\n" + dedent(block).lstrip(), encoding="utf-8")
    print(f"patched {path} ({marker})")


def patch_dag() -> None:
    path = ROOT / "src/pegasus/efg/dag.py"
    if not path.exists():
        raise RuntimeError("src/pegasus/efg/dag.py not found")
    text = path.read_text(encoding="utf-8")

    if "# ---- Slice 28Y EFG semantic manifest integration ----" in text:
        print("unchanged src/pegasus/efg/dag.py (slice28y already present)")
        return

    if "def _build_efg_base(" not in text:
        needle = "def build_efg("
        index = text.find(needle)
        if index < 0:
            raise RuntimeError("Could not find def build_efg( in src/pegasus/efg/dag.py")
        text = text[:index] + "def _build_efg_base(" + text[index + len(needle):]

    block = r'''
# ---- Slice 28Y EFG semantic manifest integration ----
# The autonomous DAG now records metadata-only core-seed and bridge-plan evidence
# without changing first-class output-bundle keys and without materializing tensors.
from functools import wraps as _slice28y_wraps

from pegasus.efg.bridges import bridge_summary as _slice28y_bridge_summary
from pegasus.efg.bridges import plan_bridge_candidates as _slice28y_plan_bridge_candidates
from pegasus.efg.core_seed import build_core_seed_set as _slice28y_build_core_seed_set
from pegasus.efg.core_seed import core_seed_summary as _slice28y_core_seed_summary


def _slice28y_empty_core_seed_summary() -> dict[str, object]:
    return {
        "registry_root": "config/registries",
        "seed_count": 0,
        "blocked_count": 0,
        "role_counts": {},
        "seeds": [],
        "blocked": [],
    }


def _slice28y_empty_bridge_plan_summary() -> dict[str, object]:
    return {
        "candidate_count": 0,
        "blocked_count": 0,
        "bridge_type_counts": {},
        "candidates": [],
        "blocked": [],
    }


if not hasattr(EFGResult, "_slice28y_base_as_manifest"):
    EFGResult._slice28y_base_as_manifest = EFGResult.as_manifest  # type: ignore[attr-defined]


def _slice28y_efgresult_as_manifest(self):
    payload = self._slice28y_base_as_manifest()  # type: ignore[attr-defined]
    payload.setdefault(
        "core_seed_summary",
        getattr(self, "_slice28y_core_seed_summary", _slice28y_empty_core_seed_summary()),
    )
    payload.setdefault(
        "bridge_plan_summary",
        getattr(self, "_slice28y_bridge_plan_summary", _slice28y_empty_bridge_plan_summary()),
    )
    payload.setdefault("semantic_manifest_schema", "28Y.1")
    return payload


EFGResult.as_manifest = _slice28y_efgresult_as_manifest  # type: ignore[method-assign]


@_slice28y_wraps(_build_efg_base)
def build_efg(*args, **kwargs):
    result = _build_efg_base(*args, **kwargs)
    fields = tuple(getattr(result, "fields", ()) or ())
    registry_root = kwargs.get("registry_root", "config/registries")
    intent = kwargs.get("intent")
    try:
        seed_set = _slice28y_build_core_seed_set(fields, registry_root=registry_root, intent=intent)
        bridge_plan = _slice28y_plan_bridge_candidates(fields, registry_root=registry_root, intent=intent)
        object.__setattr__(result, "_slice28y_core_seed_summary", _slice28y_core_seed_summary(seed_set))
        object.__setattr__(result, "_slice28y_bridge_plan_summary", _slice28y_bridge_summary(bridge_plan))
    except Exception as exc:  # pragma: no cover - defensive metadata guard only
        object.__setattr__(result, "_slice28y_core_seed_summary", _slice28y_empty_core_seed_summary())
        object.__setattr__(result, "_slice28y_bridge_plan_summary", {
            **_slice28y_empty_bridge_plan_summary(),
            "blocked": [{"reason": "semantic_manifest_failed", "error": str(exc)}],
            "blocked_count": 1,
        })
    return result
'''
    path.write_text(text.rstrip() + "\n\n" + dedent(block).lstrip(), encoding="utf-8")
    print("patched src/pegasus/efg/dag.py (slice28y semantic wrapper)")


AUDIT = r'''
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
'''

TEST_UNIT = r'''
from __future__ import annotations

from types import SimpleNamespace

import pegasus.efg.dag as dag


def _lineage():
    return SimpleNamespace(registry_versions={"test_registry": "v1"}, source_manifest_hashes=["hash_a"])


def _field(field_id: str, *, carrier: str, unit: str, source: str):
    return SimpleNamespace(
        id=field_id,
        field_id=field_id,
        name=field_id,
        kind="extensive_measure",
        carrier=carrier,
        unit=unit,
        aggregation="additive",
        role=[],
        source=[source],
        lineage=_lineage(),
        warnings=[],
    )


class _DummyEFGResult:
    def __init__(self):
        self.fields = (
            _field("sim_deaths", carrier="Deaths", unit="counts", source="SIM"),
            _field("sidra_population", carrier="Population", unit="persons", source="SIDRA"),
        )

    def as_manifest(self):
        return {"schema_version": "test"}


def test_slice28y_build_efg_wrapper_attaches_semantic_summaries(monkeypatch):
    monkeypatch.setattr(dag, "_build_efg_base", lambda *args, **kwargs: _DummyEFGResult())
    result = dag.build_efg(substrate=object(), registry_root="config/registries")

    core = getattr(result, "_slice28y_core_seed_summary")
    bridges = getattr(result, "_slice28y_bridge_plan_summary")

    assert core["seed_count"] == 2
    assert core["role_counts"]["death_event_seed"] == 1
    assert core["role_counts"]["population_denominator_seed"] == 1
    assert bridges["bridge_type_counts"]["mortality_rate_bridge"] == 1


def test_slice28y_efgresult_manifest_method_is_patched():
    assert hasattr(dag.EFGResult, "_slice28y_base_as_manifest")
    source_names = dag.EFGResult.as_manifest.__code__.co_names
    assert "core_seed_summary" in source_names or "_slice28y_core_seed_summary" in source_names
    assert "bridge_plan_summary" in source_names or "_slice28y_bridge_plan_summary" in source_names
'''

TEST_INTEGRATION = r'''
from __future__ import annotations

from scripts.dev.audits.audit_slice28y_efg_semantic_manifest import run_audit


def test_slice28y_semantic_manifest_audit_passes():
    result = run_audit()
    assert result["status"] == "passed"
    assert result["audit"] == "slice28y_efg_semantic_manifest"
'''

DOC_APPEND = r'''

## Slice 28Y semantic manifest evidence

The autonomous EFG build boundary records two metadata-only summaries on the
`EFGResult` manifest: `core_seed_summary` and `bridge_plan_summary`. These
summaries classify admitted fields into V_core-like seed roles and identify
cross-field bridge opportunities without materializing tensors or adding new
first-class output-bundle keys.
'''


def main() -> None:
    patch_dag()
    _write("scripts/dev/audits/audit_slice28y_efg_semantic_manifest.py", AUDIT)
    _write("tests/unit/test_slice28y_efg_semantic_manifest.py", TEST_UNIT)
    _write("tests/integration/test_slice28y_efg_semantic_manifest_integration.py", TEST_INTEGRATION)

    doc = ROOT / "docs/production_boundaries.md"
    if doc.exists() and "Slice 28Y semantic manifest evidence" not in doc.read_text(encoding="utf-8"):
        doc.write_text(doc.read_text(encoding="utf-8").rstrip() + "\n" + dedent(DOC_APPEND).lstrip(), encoding="utf-8")
        print("patched docs/production_boundaries.md")

    updater_target = ROOT / f"scripts/dev/updaters/{UPDATER_NAME}"
    updater_target.parent.mkdir(parents=True, exist_ok=True)
    updater_target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {updater_target.relative_to(ROOT)}")
    print("Slice 28Y updater completed. Run the quiet targeted validation block next.")


if __name__ == "__main__":
    main()
