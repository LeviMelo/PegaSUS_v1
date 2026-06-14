from __future__ import annotations

from pathlib import Path


TARGET = Path("tests/unit/test_slice27a27b_registry_canonical_delta.py")
EXPECTED_IMPORT = "from pegasus.efg.legality import evaluate_delta"
REPLACEMENT = 'from __future__ import annotations\n\nfrom pathlib import Path\nfrom types import SimpleNamespace\n\nimport yaml\n\nfrom pegasus.efg.legality import evaluate_delta\nfrom pegasus.registries.cnes_capacity import capacity_evidence\nfrom pegasus.registries.diagnostic_topology import diagnostic_evidence\nfrom pegasus.registries.semantic import registry_is_scaffold_only\nfrom pegasus.registries.sih_cost import cost_evidence\n\n\nCRITICAL_REGISTRIES = (\n    "diagnostic_topology.yaml",\n    "cnes_capacity_registry.yaml",\n    "sih_cost_registry.yaml",\n    "clinical_event_definitions.yaml",\n    "bridge_grammars.yaml",\n    "race_axis_registry.yaml",\n    "icd_catalog.yaml",\n    "icd_quality_groups.yaml",\n    "join_affordances.yaml",\n    "municipality_crosswalk_sources.yaml",\n    "model_registry.yaml",\n    "residual_registry.yaml",\n)\n\n\ndef _enum_value(value: str):\n    return SimpleNamespace(value=value)\n\n\ndef _field(**kwargs):\n    """Return a FieldNode-like fixture while preserving raw registry identity.\n\n    This test file intentionally does not construct a real FieldNode because the\n    registry evidence helpers need the caller-supplied raw ``field_id`` values\n    such as ``sim_CAUSABAS`` and ``cnes_QTLEIT``.  The fixture nevertheless must\n    expose every FieldNode surface currently consumed by ``evaluate_delta``:\n    state, materialization_state, provenance, support, axes, role, source,\n    warnings, and id/field_id aliases.\n    """\n\n    payload = {\n        "field_id": "field",\n        "id": None,\n        "name": "field",\n        "kind": "extensive_measure",\n        "carrier": "Deaths",\n        "unit": "counts",\n        "aggregation": "additive",\n        "support": {\n            "years": [2022],\n            "municipalities": ["2704302"],\n            "n_events": 1.0,\n            "missingness": 0.0,\n            "denom_fragility": 0.0,\n        },\n        "axes": {\n            "time": "year",\n            "geography": "municipality",\n            "race_axis_type": None,\n        },\n        "role": ["test"],\n        "source": ["test_fixture"],\n        "operator": "raw_field",\n        "provenance": ["official"],\n        "state": _enum_value("verified"),\n        "warnings": [],\n        "lineage": SimpleNamespace(\n            parent_ids=[],\n            operator_type="raw_field",\n            operator_params={},\n            registry_versions={"registry_set": "slice27a27b_test"},\n            source_manifest_hashes=["slice27a27b_test_fixture"],\n            code_version="test",\n        ),\n        "materialization_state": _enum_value("metadata_only"),\n        "path": None,\n        "dashboard_safe": False,\n        "role_json": "[]",\n        "source_json": "[]",\n        "metadata_json": "{}",\n    }\n    payload.update(kwargs)\n\n    if payload["id"] is None:\n        payload["id"] = payload["field_id"]\n\n    # Allow callers to pass bare enum strings while keeping evaluate_delta\'s\n    # field.state.value / field.materialization_state.value contract intact.\n    if isinstance(payload.get("state"), str):\n        payload["state"] = _enum_value(payload["state"])\n    if isinstance(payload.get("materialization_state"), str):\n        payload["materialization_state"] = _enum_value(payload["materialization_state"])\n\n    return SimpleNamespace(**payload)\n\n\ndef test_slice27a_critical_registries_are_not_scaffold_only() -> None:\n    root = Path("config/registries")\n    for name in CRITICAL_REGISTRIES:\n        path = root / name\n        assert path.exists(), name\n        payload = yaml.safe_load(path.read_text(encoding="utf-8"))\n        assert payload["registry_version"] == "v2.0", name\n        assert not registry_is_scaffold_only(name), name\n        entries = payload.get("entries") or []\n        assert len(entries) >= 1, name\n        assert all("scaffold_only" not in set(entry.get("warnings") or []) for entry in entries), name\n\n\ndef test_slice27a_semantic_registry_helpers_classify_special_fields() -> None:\n    diagnostic = _field(\n        field_id="sim_CAUSABAS",\n        name="CAUSABAS",\n        unit="ICD10",\n        aggregation="non_aggregable",\n        kind="observer_proxy",\n    )\n    cnes = _field(\n        field_id="cnes_QTLEIT",\n        name="QTLEIT bed capacity",\n        carrier="Facilities",\n        unit="beds",\n    )\n    sih = _field(\n        field_id="sih_VAL_SH",\n        name="VAL_SH hospital service cost",\n        carrier="HospitalAdmissions",\n        unit="BRL",\n    )\n    assert diagnostic_evidence(diagnostic)["topology_role"] == "underlying_cause"\n    assert capacity_evidence(cnes)["capacity_family"] == "bed_capacity"\n    assert cost_evidence(sih)["cost_component"] == "hospital_service"\n\n\ndef test_slice27b_delta_warnings_carry_registry_evidence() -> None:\n    operator = SimpleNamespace(name="raw_field")\n    diagnostic = _field(\n        field_id="sim_CAUSABAS",\n        name="CAUSABAS",\n        unit="ICD10",\n        aggregation="non_aggregable",\n        kind="observer_proxy",\n    )\n    result = evaluate_delta(parents=[diagnostic], operator=operator)\n    warnings = "\\n".join(str(value) for value in result.warnings)\n    assert "registry_evidence_attached" in warnings\n    assert "diagnostic_topology=sim_underlying_cause_causabas" in warnings\n'


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"missing target test file: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")
    if EXPECTED_IMPORT not in original:
        raise RuntimeError(
            f"{TARGET} does not look like the Slice 27 registry-delta test file; "
            "refusing to overwrite."
        )

    if original == REPLACEMENT:
        print("repair already applied")
        return

    backup = TARGET.with_suffix(TARGET.suffix + ".slice27_fixture_contract.bak")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(REPLACEMENT, encoding="utf-8")
    print(f"Rewrote {TARGET} with complete FieldNode-like registry-evidence fixture.")
    print(f"Backup written to {backup}")


if __name__ == "__main__":
    main()
