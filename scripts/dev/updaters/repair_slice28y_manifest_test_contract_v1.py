from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
TEST = ROOT / "tests" / "unit" / "test_slice28y_efg_semantic_manifest.py"
UPDATER_DST = ROOT / "scripts" / "dev" / "updaters" / "repair_slice28y_manifest_test_contract_v1.py"

OLD_NAME = "def test_slice28y_efgresult_manifest_method_is_patched"

NEW_FUNCTION = '''def test_slice28y_efgresult_manifest_method_is_patched() -> None:
    code = EFGResult.as_manifest.__code__
    source_names = set(code.co_names)
    string_constants = {value for value in code.co_consts if isinstance(value, str)}

    # ``core_seed_summary`` and ``bridge_plan_summary`` are payload keys passed
    # to ``dict.setdefault``.  In CPython bytecode they are string constants, not
    # attribute/function names.  The contract is that the patched manifest method
    # injects those keys and uses the Slice 28Y default-summary helpers.
    assert "setdefault" in source_names
    assert "core_seed_summary" in string_constants
    assert "bridge_plan_summary" in string_constants
    assert "semantic_manifest_schema" in string_constants
    assert "_slice28y_empty_core_seed_summary" in source_names
    assert "_slice28y_empty_bridge_plan_summary" in source_names
'''


def replace_function(text: str, function_name: str, replacement: str) -> str:
    start = text.find(function_name)
    if start == -1:
        raise RuntimeError(f"Could not find {function_name} in {TEST}")
    next_def = text.find("\ndef ", start + 1)
    if next_def == -1:
        end = len(text)
    else:
        end = next_def + 1
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def main() -> None:
    if not TEST.exists():
        raise RuntimeError(f"Missing test file: {TEST}")
    text = TEST.read_text(encoding="utf-8")
    updated = replace_function(text, OLD_NAME, NEW_FUNCTION)
    if updated == text:
        raise RuntimeError("Patch produced no changes")
    backup = TEST.with_suffix(TEST.suffix + ".slice28y_test_contract.bak")
    backup.write_text(text, encoding="utf-8")
    TEST.write_text(updated, encoding="utf-8")
    UPDATER_DST.parent.mkdir(parents=True, exist_ok=True)
    UPDATER_DST.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Patched {TEST}")
    print(f"Backup written to {backup}")
    print(f"Updater copied to {UPDATER_DST}")


if __name__ == "__main__":
    main()
