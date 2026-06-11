"""Repair Slice 11A acceptance integration test call against dashboard read-only API.

The dashboard read-only service exposes inspect_run as a keyword-only API
(`inspect_run(*, run_dir=...)`). Slice 11A's integration test accidentally called
it positionally. This repair keeps the production API strict and fixes the test.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = ROOT / "tests" / "integration" / "test_slice11a_acceptance_integration.py"


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing expected file: {path}")
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    text = _read(TEST_PATH)
    old = "inspected = inspect_run(run_dir)"
    new = "inspected = inspect_run(run_dir=run_dir)"
    if new in text:
        print("Slice 11A repair v1 already applied: dashboard inspect_run test uses keyword API.")
        return
    if old not in text:
        raise RuntimeError(
            "Could not find positional inspect_run test call. "
            "Inspect tests/integration/test_slice11a_acceptance_integration.py manually."
        )
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one positional inspect_run call, found {count}")
    text = text.replace(old, new, 1)
    _write(TEST_PATH, text)
    print("Slice 11A repair v1 applied: dashboard inspect_run test now uses keyword-only API.")


if __name__ == "__main__":
    main()
