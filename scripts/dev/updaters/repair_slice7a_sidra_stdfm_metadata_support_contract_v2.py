from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one occurrence, found {count}.")
    return text.replace(old, new, 1)


def patch_sidra_stdfm_bundle() -> None:
    path = ROOT / "src/pegasus/output/sidra_stdfm_bundle.py"
    if not path.exists():
        raise RuntimeError("src/pegasus/output/sidra_stdfm_bundle.py not found; apply Slice 7A first.")
    text = _read(path)

    if "schema_support = dict(support)" not in text:
        old = '''    lineage = {
        "field_id": field_id,
        "operator": operator,
        "support": support,
        "axes": axes,
        "metadata": metadata,
    }
'''
        new = '''    schema_support = dict(support)
    if metadata:
        schema_support["field_metadata"] = metadata
    lineage = {
        "field_id": field_id,
        "operator": operator,
        "support": schema_support,
        "axes": axes,
        "metadata": metadata,
    }
'''
        text = _replace_once(text, old, new, label="sidra_stdfm_bundle.py _field lineage metadata support")

    text = text.replace('        "support": _json(support),\n', '        "support": _json(schema_support),\n', 1)
    text = text.replace('        "support_json": _json(support),\n', '        "support_json": _json(schema_support),\n', 1)
    _write(path, text)


def _insert_field_metadata_helper(text: str) -> str:
    if "def _field_metadata(row):" in text:
        return text
    marker = '''def _loads(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value
'''
    helper = marker + '''

def _field_metadata(row):
    support = _loads(row.get("support_json"))
    if isinstance(support, dict) and isinstance(support.get("field_metadata"), dict):
        return support["field_metadata"]
    return _loads(row.get("metadata_json") or row.get("metadata"))
'''
    return _replace_once(text, marker, helper, label="insert _field_metadata helper")


def patch_integration_test() -> None:
    path = ROOT / "tests/integration/test_slice7a_sidra_stdfm_integration.py"
    if not path.exists():
        raise RuntimeError("tests/integration/test_slice7a_sidra_stdfm_integration.py not found; apply Slice 7A first.")
    text = _insert_field_metadata_helper(_read(path))
    replacements = {
        'projected_meta = _loads(by_id["sidra_projected_labor_context"].get("metadata_json") or by_id["sidra_projected_labor_context"].get("metadata"))':
        'projected_meta = _field_metadata(by_id["sidra_projected_labor_context"])',
        'highdim_meta = _loads(by_id["sidra_highdim_bounded_context"].get("metadata_json") or by_id["sidra_highdim_bounded_context"].get("metadata"))':
        'highdim_meta = _field_metadata(by_id["sidra_highdim_bounded_context"])',
        'stdfm_meta = _loads(stdfm.get("metadata_json") or stdfm.get("metadata"))':
        'stdfm_meta = _field_metadata(stdfm)',
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
    # Preserve the v1 dashboard_safe schema correction if the repo still has the original assertion.
    text = text.replace('assert stdfm["dashboard_safe"] is False', 'assert str(stdfm["dashboard_safe"]) == "False"')
    _write(path, text)


def patch_audit() -> None:
    path = ROOT / "scripts/dev/audits/audit_slice7a_sidra_stdfm.py"
    if not path.exists():
        raise RuntimeError("scripts/dev/audits/audit_slice7a_sidra_stdfm.py not found; apply Slice 7A first.")
    text = _insert_field_metadata_helper(_read(path))
    replacements = {
        'meta = _loads(by_id["sidra_projected_labor_context"].get("metadata_json") or by_id["sidra_projected_labor_context"].get("metadata"))':
        'meta = _field_metadata(by_id["sidra_projected_labor_context"])',
        'meta = _loads(by_id["sidra_highdim_bounded_context"].get("metadata_json") or by_id["sidra_highdim_bounded_context"].get("metadata"))':
        'meta = _field_metadata(by_id["sidra_highdim_bounded_context"])',
        'meta = _loads(row.get("metadata_json") or row.get("metadata"))':
        'meta = _field_metadata(row)',
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
    text = text.replace('row.get("dashboard_safe") is not False', 'str(row.get("dashboard_safe")) != "False"')
    _write(path, text)


def patch_cli_eof() -> None:
    path = ROOT / "src/pegasus/cli.py"
    if path.exists():
        _write(path, _read(path))


def main() -> None:
    if not (ROOT / "pyproject.toml").exists():
        raise RuntimeError("Run this repair from the PegaSUS repository root.")
    patch_sidra_stdfm_bundle()
    patch_integration_test()
    patch_audit()
    patch_cli_eof()
    print("Slice 7A repair v2 applied: schema-native support_json now carries Slice 7A field metadata; tests/audit read metadata from support_json.")


if __name__ == "__main__":
    main()
