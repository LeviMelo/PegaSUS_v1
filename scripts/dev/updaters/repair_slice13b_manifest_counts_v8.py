from __future__ import annotations

import importlib
import inspect
import py_compile
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path.cwd()
TARGET = REPO_ROOT / "src" / "pegasus" / "she" / "source_registry.py"

NEW_FUNCTION = """
def source_registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    \"\"\"Return SHE-facing source-field registry metadata.

    Slice 13B tests and CLI callers consume summary counters at the top level.
    The full registry manifest is still preserved, and the same summary is also
    retained under ``summary`` for structured consumers.
    \"\"\"
    summary = source_field_registry_summary(registry_root=registry_root)
    payload = dict(source_field_registry_manifest(registry_root=registry_root))
    payload.update(summary)
    payload["summary"] = dict(summary)
    payload["she_source_registry_api"] = {
        "registry_backed": True,
        "carrier_surface": "canonical_registry_id_with_legacy_equality",
        "batch_signature": "resolve_source_fields(source_system, columns, allow_heuristic=True, registry_root=...)",
    }
    return payload
"""


def _compile(path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        py_compile.compile(str(path), cfile=str(Path(tmp) / (path.name + ".pyc")), doraise=True)


def patch_source_registry() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Missing target file: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\ndef source_registry_manifest\(registry_root: str \| Path = \"config/registries\"\) -> dict\[str, Any\]:\n"
        r".*?\n    return payload\n?$",
        re.DOTALL,
    )
    replacement = "\n" + NEW_FUNCTION.lstrip()
    new_text, count = pattern.subn(replacement, text)

    if count != 1:
        raise RuntimeError(
            "Could not replace exactly one source_registry_manifest() definition. "
            f"Matched {count} definitions in {TARGET}."
        )

    TARGET.write_text(new_text.rstrip() + "\n", encoding="utf-8")


def validate_live_module() -> None:
    _compile(TARGET)

    src_root = str(REPO_ROOT / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    module = importlib.import_module("pegasus.she.source_registry")
    module = importlib.reload(module)

    manifest = module.source_registry_manifest()
    assert "entry_count" in manifest, "source_registry_manifest() must expose entry_count at top level"
    assert manifest["entry_count"] >= 40, manifest
    assert "admissible_entry_count" in manifest, "source_registry_manifest() must expose admissible_entry_count"
    assert "audit_or_excluded_entry_count" in manifest, "source_registry_manifest() must expose audit_or_excluded_entry_count"
    assert "by_source_system" in manifest, "source_registry_manifest() must expose by_source_system"
    assert "summary" in manifest and manifest["summary"]["entry_count"] == manifest["entry_count"], manifest
    assert "entries" in manifest, "full registry manifest entries must remain present"
    assert "patterns" in manifest, "full registry manifest patterns must remain present"

    sig = inspect.signature(module.resolve_source_fields)
    assert "allow_heuristic" in sig.parameters, "resolve_source_fields must keep allow_heuristic compatibility"

    sim = module.resolve_source_fields(
        source_system="SIM-DO",
        columns=["underlying_icd_norm", "age_years", "raw_json"],
        allow_heuristic=True,
    )
    specs = {spec.column: spec for spec in sim.specs}
    assert specs["underlying_icd_norm"].unit == "ICD10"
    assert specs["underlying_icd_norm"].aggregation == "non_aggregable"
    assert "diagnostic_topology" in specs["underlying_icd_norm"].role
    assert specs["age_years"].carrier == "Deaths"
    assert specs["age_years"].carrier == "deaths"
    assert specs["raw_json"].admissible is False


def main() -> None:
    patch_source_registry()
    validate_live_module()
    print("repair_slice13b_manifest_counts_v8: patched source_registry_manifest top-level summary counters")


if __name__ == "__main__":
    main()
