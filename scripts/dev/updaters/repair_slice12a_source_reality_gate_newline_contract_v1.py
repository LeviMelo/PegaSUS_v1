from __future__ import annotations

from pathlib import Path
import py_compile
import zipfile

ROOT = Path.cwd()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _replace_or_confirm(path: Path, old: str, new: str, confirm: str) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    text = _read(path)
    if old in text:
        _write(path, text.replace(old, new))
        return True
    if confirm in text:
        return False
    raise RuntimeError(
        f"Could not find expected broken or fixed newline contract in {path}. "
        "Upload the file/log if this occurs."
    )


def main() -> None:
    contracts = ROOT / "src/pegasus/source_artifacts/contracts.py"
    updater = ROOT / "scripts/dev/updaters/apply_slice12a_source_reality_gate.py"

    broken_contracts = 'out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")'
    fixed_contracts = r'out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")'
    _replace_or_confirm(
        contracts,
        old=broken_contracts,
        new=fixed_contracts,
        confirm=fixed_contracts,
    )

    if updater.exists():
        updater_old = r'out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")'
        updater_new = r'out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")'
        _replace_or_confirm(
            updater,
            old=updater_old,
            new=updater_new,
            confirm=updater_new,
        )

    compile_targets = [
        contracts,
        ROOT / "src/pegasus/source_artifacts/__init__.py",
        ROOT / "src/pegasus/source_artifacts/resolver.py",
        ROOT / "src/pegasus/workflows/source_artifacts.py",
        ROOT / "tests/unit/test_source_artifact_contract.py",
        ROOT / "tests/integration/test_slice12a_source_artifact_integration.py",
        ROOT / "scripts/dev/audits/audit_slice12a_source_artifacts.py",
    ]
    for target in compile_targets:
        if target.exists():
            py_compile.compile(str(target), doraise=True)

    print("Slice 12A repair v1 applied: source artifact manifest newline serialization fixed in live module and updater.")


if __name__ == "__main__":
    main()
