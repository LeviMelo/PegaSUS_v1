from __future__ import annotations

from pathlib import Path
import re


TEST_PATH = Path("tests/unit/test_slice27a27b_registry_canonical_delta.py")


def _insert_materialization_state(text: str) -> str:
    if "materialization_state=" in text:
        return text

    # Preferred target: the _field(...) helper returns a SimpleNamespace with
    # state=SimpleNamespace(value="active"). Insert the corresponding
    # materialization_state immediately after it, preserving indentation.
    pattern = re.compile(
        r"(?P<line>^(?P<indent>\s*)state\s*=\s*SimpleNamespace\s*\(\s*value\s*=\s*(?P<quote>['\"])active(?P=quote)\s*\)\s*,\s*$)",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            match.group("line")
            + "\n"
            + f'{indent}materialization_state=SimpleNamespace(value="verified"),'
        )

    patched, count = pattern.subn(repl, text, count=1)
    if count:
        return patched

    # Fallback: insert before role_json= in the first SimpleNamespace fixture
    # helper. This is robust to a different state variable representation while
    # preserving the object shape observed in the failing pytest traceback.
    role_pattern = re.compile(r"^(?P<indent>\s*)role_json\s*=", re.MULTILINE)
    match = role_pattern.search(text)
    if match:
        indent = match.group("indent")
        return (
            text[: match.start()]
            + f'{indent}materialization_state=SimpleNamespace(value="verified"),\n'
            + text[match.start():]
        )

    # Final fallback: insert before source_json= if role_json is absent.
    source_pattern = re.compile(r"^(?P<indent>\s*)source_json\s*=", re.MULTILINE)
    match = source_pattern.search(text)
    if match:
        indent = match.group("indent")
        return (
            text[: match.start()]
            + f'{indent}materialization_state=SimpleNamespace(value="verified"),\n'
            + text[match.start():]
        )

    raise RuntimeError(
        "Could not patch tests/unit/test_slice27a27b_registry_canonical_delta.py: "
        "no materialization_state anchor, role_json anchor, or source_json anchor found."
    )


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    original = TEST_PATH.read_text(encoding="utf-8")
    patched = _insert_materialization_state(original)

    if patched == original:
        print("Slice 27A/27B test fixture already includes materialization_state; no changes.")
        return

    TEST_PATH.write_text(patched, encoding="utf-8")
    print("Patched Slice 27A/27B test fixture with FieldNode-like materialization_state.")


if __name__ == "__main__":
    main()
