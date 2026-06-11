from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _strip_trailing_blank_lines(text: str) -> str:
    # Preserve exactly one POSIX newline at EOF and remove blank EOF lines that
    # git diff --check reports as "new blank line at EOF".
    return text.rstrip() + "\n"


def patch_cli() -> None:
    path = ROOT / "src" / "pegasus" / "cli.py"
    text = _read(path)

    if "import json\n" not in text:
        if "from pathlib import Path\n" in text:
            text = text.replace("from pathlib import Path\n", "import json\nfrom pathlib import Path\n", 1)
        elif "from __future__ import annotations\n\n" in text:
            text = text.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nimport json\n", 1)
        else:
            raise RuntimeError("src/pegasus/cli.py: could not locate import insertion point")

    _write(path, _strip_trailing_blank_lines(text))


def trim_eof(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"missing expected file: {path}")
    _write(path, _strip_trailing_blank_lines(_read(path)))


def main() -> None:
    patch_cli()
    trim_eof(ROOT / "src" / "pegasus" / "dashboard" / "contracts.py")
    trim_eof(ROOT / "src" / "pegasus" / "dashboard" / "read_only.py")
    print("Slice 10A repair v1 applied: dashboard CLI JSON import restored and EOF blank lines removed.")


if __name__ == "__main__":
    main()
