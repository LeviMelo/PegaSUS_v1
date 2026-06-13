from __future__ import annotations

from pegasus.efg.equivalence import precompress_fields
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node


def _field(
    *,
    name: str,
    operator: str = "she_substrate_materialization",
    axes: dict | None = None,
    role: list[str] | None = None,
    aggregation: str = "additive",
    carrier: str = "HospitalAdmissions",
    unit: str = "counts",
    source_hash: str = "source-a",
):
    lineage = make_lineage(
        parent_ids=["parent-a"] if operator != "she_substrate_materialization" else [],
        operator_type=operator,
        operator_params={"drop_axes": ["sex"]} if operator == "pi_*" else {"name": name},
        registry_versions={"registry": "v1"},
        source_manifest_hashes=[source_hash],
        code_version="slice20a",
    )
    return make_field_node(
        name=name,
        kind="observer_proxy" if "diagnostic_topology" in (role or []) else "extensive_measure",
        carrier=carrier,
        unit=unit,
        support={"artifact_path": "same.parquet", "column": name},
        axes=axes or {},
        aggregation=aggregation,
        role=role or ["source_field"],
        source=["SIH-RD", "same.parquet", name],
        operator=operator,
        provenance=["source_normalized"],
        state="verified",
        warnings=[],
        lineage=lineage,
        materialization_state="metadata_only",
    )


def test_slice20a_projection_identity_uses_parent_operator_support_and_registry_metadata() -> None:
    left = _field(name="projection-a", operator="pi_*", source_hash="source-a")
    right = _field(name="projection-b", operator="pi_*", source_hash="source-b")
    compressed, report = precompress_fields([left, right])

    assert len(compressed) == 1
    assert report.suppressed_count == 1
    item = report.suppressed[0]
    assert item.equivalence_kind == "exact_projection_redundancy"
    assert "parent_ids" in item.compared_metadata
    assert "exact equality" in item.proof
    assert report.as_manifest()["uses_numerical_arrays"] is False


def test_slice20a_diagnostic_aliases_compress_only_with_same_topology_and_code_set() -> None:
    left = _field(
        name="diagnostic-a",
        axes={"icd_topology_role": "underlying_cause", "code_set": ["A00", "A01"]},
        role=["diagnostic_topology", "underlying_cause"],
        aggregation="non_aggregable",
        unit="ICD10",
    )
    alias = _field(
        name="diagnostic-alias",
        axes={"icd_topology_role": "underlying_cause", "code_set": ["A00", "A01"]},
        role=["diagnostic_topology", "underlying_cause"],
        aggregation="non_aggregable",
        unit="ICD10",
        source_hash="source-b",
    )
    secondary = _field(
        name="diagnostic-secondary",
        axes={"icd_topology_role": "sih_secondary_diagnosis", "code_set": ["A00", "A01"]},
        role=["diagnostic_topology", "sih_secondary_diagnosis"],
        aggregation="non_aggregable",
        unit="ICD10",
        source_hash="source-c",
    )
    compressed, report = precompress_fields([left, alias, secondary])

    assert len(compressed) == 2
    assert report.suppressed[0].equivalence_kind == "nested_icd_exact_identity"
    assert any(
        item.protection_kind == "diagnostic_topology_identity"
        for item in report.protected_non_equivalences
    )


def test_slice20a_cost_components_and_capacity_indices_are_protected() -> None:
    cost_sh = _field(name="cost-sh", axes={"cost_component": "VAL_SH"}, unit="BRL")
    cost_sp = _field(name="cost-sp", axes={"cost_component": "VAL_SP"}, unit="BRL", source_hash="b")
    bed_a = _field(
        name="bed-a", axes={"capacity_vector_index": "QTLEITP1"}, carrier="Facilities", unit="beds",
    )
    bed_b = _field(
        name="bed-b", axes={"capacity_vector_index": "QTLEITP3"}, carrier="Facilities", unit="beds", source_hash="b",
    )
    compressed, report = precompress_fields([cost_sh, cost_sp, bed_a, bed_b])

    assert len(compressed) == 4
    protections = {item.protection_kind for item in report.protected_non_equivalences}
    assert "cost_component_identity" in protections
    assert "capacity_vector_identity" in protections
    assert report.as_manifest()["protected_non_equivalence_count"] >= 2
