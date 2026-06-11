from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = ROOT / "tests" / "integration" / "test_slice11a_acceptance_integration.py"


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def patch_test() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"missing expected Slice 11A integration test: {TEST_PATH}")
    text = TEST_PATH.read_text(encoding="utf-8")

    old = "    assert inspected.validation_ok is True\n"
    new = '    assert inspected["validation_ok"] is True\n'

    if old in text:
        text = text.replace(old, new, 1)
    elif new in text:
        pass
    else:
        # More defensive fallback for slightly reformatted test files.
        marker = "inspected = inspect_run(run_dir=run_dir)"
        if marker not in text:
            raise RuntimeError("Slice 11A dashboard inspection test does not contain expected inspect_run call.")
        text2 = text.replace("assert inspected.validation_ok is True", 'assert inspected["validation_ok"] is True', 1)
        if text2 == text:
            raise RuntimeError(
                "Slice 11A dashboard inspection test does not contain the stale inspected.validation_ok assertion."
            )
        text = text2

    _write(TEST_PATH, text)


def main() -> None:
    patch_test()
    print("Slice 11A repair v2 applied: dashboard inspect_run assertion now matches dict-returning API.")


if __name__ == "__main__":
    main()
