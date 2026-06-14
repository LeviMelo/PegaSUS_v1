from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
TEST_PATH = ROOT / "tests" / "unit" / "test_slice27a27b_registry_canonical_delta.py"
UPDATER_PATH = ROOT / "scripts" / "dev" / "updaters" / "repair_slice27a27b_test_fixture_v2.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def patch_test_fixture(text: str) -> str:
    if "materialization_state=SimpleNamespace(value=\"verified\")" in text:
        return text

    needle = "        state=SimpleNamespace(value=\"active\"),\n"
    replacement = (
        "        state=SimpleNamespace(value=\"active\"),\n"
        "        materialization_state=SimpleNamespace(value=\"verified\"),\n"
    )
    if needle not in text:
        raise RuntimeError(
            "Could not find the SimpleNamespace state anchor in "
            "tests/unit/test_slice27a27b_registry_canonical_delta.py"
        )
    return text.replace(needle, replacement, 1)


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    original = read(TEST_PATH)
    patched = patch_test_fixture(original)
    if patched != original:
        write(TEST_PATH, patched)

    # Preserve the repair script in the repository updater ledger when run from a copied path.
    try:
        source = Path(__file__).resolve()
        if source != UPDATER_PATH.resolve():
            write(UPDATER_PATH, source.read_text(encoding="utf-8"))
    except OSError:
        pass

    print("Applied repair_slice27a27b_test_fixture_v2: added materialization_state to Slice 27 test FieldNode-like fixture.")


if __name__ == "__main__":
    main()
