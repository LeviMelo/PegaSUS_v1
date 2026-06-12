from __future__ import annotations

from pegasus.she.source_registry import resolve_source_field, source_registry_manifest


def test_slice13b_she_source_registry_is_yaml_backed_for_known_field() -> None:
    result = resolve_source_field(source_system="SIM-DO", column_name="race_color_admin")
    assert result.registry_backed is True
    assert result.known is True
    assert result.spec.carrier == "Deaths"
    assert result.spec.axes["race_axis_type"] == "administrative_death_declaration"
    assert result.spec.dashboard_safe == "warning"
    assert result.spec.registry_hash


def test_slice13b_she_source_registry_marks_unknown_as_not_admissible() -> None:
    result = resolve_source_field(source_system="SINASC", column_name="MYSTERY")
    assert result.registry_backed is True
    assert result.known is False
    assert result.spec.admissible is False
    assert "unknown_source_field_requires_registry_entry" in result.warnings
    assert "source_field_not_substrate_admissible:audit_only" in result.warnings


def test_slice13b_she_source_registry_manifest_exposes_counts() -> None:
    manifest = source_registry_manifest()
    assert manifest["entry_count"] >= 40
    assert manifest["registry_hash"]
    assert "SIM-DO" in manifest["by_source_system"]
