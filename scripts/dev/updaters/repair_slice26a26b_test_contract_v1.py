from __future__ import annotations

import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] if "scripts/dev/updaters" in str(Path(__file__).resolve()) else Path.cwd()


def p(rel: str) -> Path:
    return ROOT / rel


def read(rel: str) -> str:
    path = p(rel)
    if not path.exists():
        raise SystemExit(f"[repair26a26b] required file missing: {rel}")
    return path.read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = p(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n").rstrip() + "\n", encoding="utf-8", newline="\n")


def patch_unit_test() -> None:
    rel = "tests/unit/test_slice26a26b_runtime_authority_stage_plan.py"
    text = read(rel)
    old = '    assert "legacy_bootstrap" not in str(metadata)\n    assert "pegasus.workflows.efg.run_build_sim_fixture" not in str(metadata)\n'
    new = '''    assert metadata["numerical_materialization"] != "legacy_bootstrap"
    assert metadata["legacy_bootstrap_builder"] is None
    assert metadata["legacy_bootstrap_status"] == "quarantined_fixture_only"
    assert "compatibility_materializer" not in str(metadata)
    assert "pegasus.workflows.efg.run_build_sim_fixture" not in str(metadata)
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif 'assert "legacy_bootstrap" not in str(metadata)' in text:
        text = text.replace('    assert "legacy_bootstrap" not in str(metadata)\n', new, 1)
    else:
        raise SystemExit("[repair26a26b] unit test legacy_bootstrap assertion anchor not found")
    write(rel, text)


def patch_slice25_integration_test() -> None:
    rel = "tests/integration/test_slice25a_final_compiler_unification.py"
    text = read(rel)
    if 'architecture["legacy_bootstrap_status"] == "compatibility_materializer"' not in text:
        # It may already be patched; still enforce modern expected assertions if absent.
        if 'architecture["legacy_bootstrap_status"] == "quarantined_fixture_only"' not in text:
            raise SystemExit("[repair26a26b] Slice 25 architecture status assertion anchor not found")
    text = text.replace(
        'architecture["legacy_bootstrap_status"] == "compatibility_materializer"',
        'architecture["legacy_bootstrap_status"] == "quarantined_fixture_only"',
    )
    # Add precise 26A assertion while preserving the Slice 25 test's role as compiler-unification regression.
    anchor = '    assert architecture["legacy_bootstrap_status"] == "quarantined_fixture_only"\n'
    insert = anchor + '    assert architecture["numerical_materialization"] == "autonomous_compiler_services"\n    assert architecture.get("legacy_bootstrap_builder") is None\n'
    if anchor in text and 'architecture["numerical_materialization"] == "autonomous_compiler_services"' not in text:
        text = text.replace(anchor, insert, 1)
    write(rel, text)


def patch_runtime_audit() -> None:
    rel = "scripts/dev/audits/audit_slice26a26b_runtime_authority_stage_plan.py"
    text = read(rel)
    old_set = '''FORBIDDEN_ARCHITECTURE_VALUES = {
    "legacy_bootstrap",
    "compatibility_materializer",
    "pegasus.workflows.efg.run_build_sim_fixture",
}
'''
    new_set = '''FORBIDDEN_ARCHITECTURE_VALUES = {
    "compatibility_materializer",
    "pegasus.workflows.efg.run_build_sim_fixture",
}
'''
    if old_set in text:
        text = text.replace(old_set, new_set, 1)
    elif '"legacy_bootstrap",' in text:
        text = text.replace('    "legacy_bootstrap",\n', '', 1)
    # Add field-specific checks so the audit forbids the old runtime authority value without forbidding
    # the explicit quarantine key name legacy_bootstrap_status.
    anchor = '''    if metadata.get("legacy_graph_authority") is not False:
        errors.append("legacy_graph_authority must be False")
'''
    addition = '''    if metadata.get("legacy_graph_authority") is not False:
        errors.append("legacy_graph_authority must be False")
    if metadata.get("numerical_materialization") == "legacy_bootstrap":
        errors.append("numerical_materialization must not be legacy_bootstrap")
    if metadata.get("legacy_bootstrap_builder") not in {None, ""}:
        errors.append("legacy_bootstrap_builder must be None/empty in quarantined architecture")
    if metadata.get("legacy_bootstrap_status") != "quarantined_fixture_only":
        errors.append("legacy_bootstrap_status must be quarantined_fixture_only")
'''
    if anchor in text and 'numerical_materialization must not be legacy_bootstrap' not in text:
        text = text.replace(anchor, addition, 1)
    write(rel, text)


def patch_updater_copy_if_present() -> None:
    rel = "scripts/dev/updaters/apply_slice26a26b_runtime_authority_stage_plan.py"
    path = p(rel)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    # Keep the historical updater consistent if it is tracked/committed.
    text = text.replace('    "legacy_bootstrap",\n', '')
    text = text.replace('    assert "legacy_bootstrap" not in str(metadata)\n', '    assert metadata["numerical_materialization"] != "legacy_bootstrap"\n    assert metadata["legacy_bootstrap_builder"] is None\n')
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    patch_unit_test()
    patch_slice25_integration_test()
    patch_runtime_audit()
    patch_updater_copy_if_present()
    print("Applied repair Slice 26A/26B v1: aligned test/audit contracts with quarantined legacy-bootstrap metadata.")
    print("Touched:")
    for rel in [
        "tests/unit/test_slice26a26b_runtime_authority_stage_plan.py",
        "tests/integration/test_slice25a_final_compiler_unification.py",
        "scripts/dev/audits/audit_slice26a26b_runtime_authority_stage_plan.py",
        "scripts/dev/updaters/apply_slice26a26b_runtime_authority_stage_plan.py (if present)",
    ]:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
