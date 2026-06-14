from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
TEST = ROOT / "tests" / "unit" / "test_slice28y_efg_semantic_manifest.py"
UPDATER_DST = ROOT / "scripts" / "dev" / "updaters" / "repair_slice28y_efgresult_reference_v2.py"

OLD_DIRECT = "code = EFGResult.as_manifest.__code__"
NEW_VIA_DAG = "code = dag.EFGResult.as_manifest.__code__"
DAG_IMPORT = "import pegasus.efg.dag as dag"


def ensure_dag_import(text: str) -> str:
    if DAG_IMPORT in text:
        return text
    lines = text.splitlines()
    insert_at = None
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    if insert_at is None:
        # Preserve future import at top if present by appending after it.
        insert_at = 1 if lines and lines[0].startswith("from __future__") else 0
    lines.insert(insert_at, DAG_IMPORT)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def main() -> None:
    if not TEST.exists():
        raise RuntimeError(f"Missing test file: {TEST}")

    text = TEST.read_text(encoding="utf-8")
    updated = ensure_dag_import(text)

    if OLD_DIRECT in updated:
        updated = updated.replace(OLD_DIRECT, NEW_VIA_DAG)
    elif NEW_VIA_DAG in updated:
        pass
    else:
        raise RuntimeError(
            "Could not find either direct EFGResult.as_manifest reference or dag.EFGResult.as_manifest reference "
            f"in {TEST}"
        )

    # If an explicit EFGResult import was previously added by hand, leave it alone;
    # it is harmless.  The canonical test contract uses the existing dag module alias
    # to avoid a second import path and to match the test file's current style.

    if updated == text:
        print(f"No changes needed: {TEST} already uses dag.EFGResult")
    else:
        backup = TEST.with_suffix(TEST.suffix + ".slice28y_efgresult_reference.bak")
        backup.write_text(text, encoding="utf-8")
        TEST.write_text(updated, encoding="utf-8")
        print(f"Patched {TEST}")
        print(f"Backup written to {backup}")

    UPDATER_DST.parent.mkdir(parents=True, exist_ok=True)
    UPDATER_DST.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Updater copied to {UPDATER_DST}")


if __name__ == "__main__":
    main()
