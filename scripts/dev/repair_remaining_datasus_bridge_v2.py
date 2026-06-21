from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
CONTRACT = "datasus_r_bridge_v2_raw_canonical_plus_microdatasus_sidecar"

FILES = {
    "subprocess": ROOT / "src" / "pegasus" / "datasus" / "subprocess.py",
    "manifests": ROOT / "src" / "pegasus" / "datasus" / "manifests.py",
    "workflow_datasus": ROOT / "src" / "pegasus" / "workflows" / "datasus.py",
    "config_datasus": ROOT / "config" / "datasus.yaml",
    "bridge_r": ROOT / "src" / "pegasus" / "datasus" / "r_scripts" / "fetch_process_microdatasus.R",
    "test_contract": ROOT / "tests" / "unit" / "test_datasus_bridge_v2_contract.py",
}

BACKUP_ROOT = ROOT / ".codex-tmp" / "datasus_bridge_v2_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = BACKUP_ROOT / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")


def patch_subprocess() -> None:
    path = FILES["subprocess"]
    text = read(path)

    if "r_library_path: str | None = None" not in text:
        text = text.replace(
            '    rscript_path: str = "Rscript"\n',
            '    rscript_path: str = "Rscript"\n'
            '    r_library_path: str | None = None\n',
            1,
        )

    start_marker = '    @classmethod\n    def from_mapping(cls, payload: dict[str, Any]) -> "DatasusConfig":\n'
    end_marker = '\n    @classmethod\n    def from_file'
    if start_marker not in text:
        raise RuntimeError("Could not find DatasusConfig.from_mapping start marker.")
    start = text.index(start_marker)
    end = text.index(end_marker, start)

    new_from_mapping = '''    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasusConfig":
        r_library_path = payload.get("r_library_path")
        return cls(
            rscript_path=str(payload.get("rscript_path", "Rscript")),
            r_library_path=None if r_library_path in {None, ""} else str(r_library_path),
            r_timeout_seconds=int(payload.get("r_timeout_seconds", 7200)),
            heartbeat_timeout_seconds=int(payload.get("heartbeat_timeout_seconds", 900)),
        )
'''
    text = text[:start] + new_from_mapping + text[end:]

    if '"--timeout-seconds"' not in text:
        anchor = '        "--out-dir",\n        str(raw_path.parent),\n'
        if anchor not in text:
            raise RuntimeError("Could not find R command --out-dir anchor in subprocess.py.")
        text = text.replace(
            anchor,
            anchor + '        "--timeout-seconds",\n        str(timeout_seconds),\n',
            1,
        )

    cache_anchor = '            cached_payload = json.loads(manifest_path.read_text(encoding="utf-8"))\n'
    if cache_anchor in text and "cached R manifest uses an obsolete DATASUS bridge contract" not in text:
        text = text.replace(
            cache_anchor,
            cache_anchor
            + '            if cached_payload.get("status") != "success":\n'
            + '                raise ValueError("cached R manifest is not successful")\n'
            + f'            if cached_payload.get("processing_contract_version") != "{CONTRACT}":\n'
            + '                raise ValueError("cached R manifest uses an obsolete DATASUS bridge contract")\n',
            1,
        )
    elif cache_anchor not in text:
        raise RuntimeError("Could not find cached_payload assignment in subprocess.py. Inspect cache block manually.")

    write(path, text)


def patch_manifests() -> None:
    path = FILES["manifests"]
    text = read(path)

    old = '''        "fetch_function": "fetch_datasus",
        "process_function": "process_datasus",
'''
    new = f'''        "fetch_function": "fetch_datasus",
        "process_function": "pegasus_process_datasus_dispatch",
        "processing_contract_version": "{CONTRACT}",
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif '"process_function": "pegasus_process_datasus_dispatch"' in text:
        pass
    else:
        raise RuntimeError("Could not find _request_identity fetch/process function block in manifests.py.")

    write(path, text)


def patch_workflow_datasus() -> None:
    path = FILES["workflow_datasus"]
    text = read(path)

    replacements = [
        (
            'elif result.status != "success":\n            failed = True',
            'elif result.status not in {"success", "cached"}:\n            failed = True',
        ),
        (
            'elif manifest.status != "success":\n            failed = True',
            'elif manifest.status not in {"success", "cached"}:\n            failed = True',
        ),
    ]

    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True

    if not changed and 'not in {"success", "cached"}' not in text:
        raise RuntimeError(
            "Could not find status failure block in workflows/datasus.py. "
            "Search manually for result.status != \"success\" or manifest.status != \"success\"."
        )

    write(path, text)


def patch_datasus_yaml() -> None:
    path = FILES["config_datasus"]
    if not path.exists():
        return

    text = read(path)
    if "r_library_path:" in text:
        return

    lines = text.splitlines()
    datasus_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "datasus:":
            datasus_idx = i
            break

    if datasus_idx is not None:
        insert_at = datasus_idx + 1
        lines.insert(insert_at, '  r_library_path: ".r-library"')
        write(path, "\n".join(lines) + "\n")
        return

    # Flat config form: load_datasus_config() accepts this too when no datasus: root exists.
    lines.append('r_library_path: ".r-library"')
    write(path, "\n".join(lines) + "\n")


def write_contract_tests() -> None:
    path = FILES["test_contract"]
    text = f'''from __future__ import annotations

from pathlib import Path

from pegasus.datasus.manifests import _request_identity
from pegasus.datasus.subprocess import DatasusConfig

CONTRACT = "{CONTRACT}"


def test_datasus_config_from_mapping_preserves_local_r_library() -> None:
    config = DatasusConfig.from_mapping({{"r_library_path": ".r-library"}})
    assert config.r_library_path == ".r-library"


def test_request_identity_uses_pegasus_dispatch_and_bridge_contract() -> None:
    identity = _request_identity(
        system="SIM-DO",
        uf="AL",
        year_start=2022,
        year_end=2022,
        month_start=None,
        month_end=None,
        config={{"backend": "microdatasus"}},
    )

    assert identity["fetch_function"] == "fetch_datasus"
    assert identity["process_function"] == "pegasus_process_datasus_dispatch"
    assert identity["processing_contract_version"] == CONTRACT


def test_r_bridge_restores_microdatasus_dispatch_without_removed_api() -> None:
    script = Path("src/pegasus/datasus/r_scripts/fetch_process_microdatasus.R").read_text(encoding="utf-8")

    assert "library(microdatasus)" in script
    assert "process_datasus_dispatch" in script
    assert "process_sim" in script
    assert "process_sinasc" in script
    assert "microdatasus::process_datasus" not in script
    assert "read.dbc::read.dbc(dbc_path, as.is = TRUE)" in script or "read.dbc(dbc_path, as.is = TRUE)" in script
    assert CONTRACT in script


def test_python_subprocess_passes_timeout_and_rejects_obsolete_cached_manifests() -> None:
    source = Path("src/pegasus/datasus/subprocess.py").read_text(encoding="utf-8")

    assert "\\"--timeout-seconds\\"" in source
    assert "str(timeout_seconds)" in source
    assert "cached R manifest uses an obsolete DATASUS bridge contract" in source
    assert CONTRACT in source
'''
    write(path, text)


def verify_r_bridge_already_rewritten() -> None:
    path = FILES["bridge_r"]
    text = read(path)

    required = [
        "library(microdatasus)",
        "process_datasus_dispatch",
        "processing_contract_version",
        CONTRACT,
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(
            "fetch_process_microdatasus.R does not look like the restored v2 bridge. "
            f"Missing: {missing}. Replace that R script first, then rerun this updater."
        )

    if "microdatasus::process_datasus" in text:
        raise RuntimeError("fetch_process_microdatasus.R still calls removed microdatasus::process_datasus().")


def main() -> None:
    verify_r_bridge_already_rewritten()
    patch_subprocess()
    patch_manifests()
    patch_workflow_datasus()
    patch_datasus_yaml()
    write_contract_tests()

    print("Patched remaining DATASUS bridge v2 files.")
    print(f"Backups written under: {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
