"""Proof-carrying metadata-level topological precompression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import FieldNode


@dataclass(frozen=True)
class SuppressedEquivalent:
    suppressed_field_id: str
    canonical_field_id: str
    equivalence_kind: str
    signature: str
    proof: str = "metadata_identity"
    compared_metadata: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compared_metadata"] = list(self.compared_metadata)
        return payload


@dataclass(frozen=True)
class ProtectedNonEquivalence:
    left_field_id: str
    right_field_id: str
    protection_kind: str
    reason: str

    def as_manifest(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PrecompressionReport:
    input_count: int
    output_count: int
    suppressed: tuple[SuppressedEquivalent, ...]
    canonical_by_field_id: dict[str, str]
    protected_non_equivalences: tuple[ProtectedNonEquivalence, ...] = ()

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed)

    @property
    def protected_count(self) -> int:
        return len(self.protected_non_equivalences)

    def as_manifest(self) -> dict[str, Any]:
        class_counts: dict[str, int] = {}
        for item in self.suppressed:
            class_counts[item.equivalence_kind] = class_counts.get(item.equivalence_kind, 0) + 1
        return {
            "stage": "topological_precompression",
            "uses_numerical_arrays": False,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "suppressed_count": self.suppressed_count,
            "protected_non_equivalence_count": self.protected_count,
            "equivalence_class_counts": dict(sorted(class_counts.items())),
            "suppressed": [item.as_manifest() for item in self.suppressed],
            "protected_non_equivalences": [
                item.as_manifest() for item in self.protected_non_equivalences
            ],
            "canonical_by_field_id": dict(self.canonical_by_field_id),
        }


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set)):
        return sorted((_stable(item) for item in value), key=repr)
    return value


def _support_signature(field: FieldNode, *, include_column: bool = True) -> dict[str, Any]:
    ignored = {
        "row_count", "non_null_count", "unique_non_null_count", "missing_rate",
        "numeric_min", "numeric_max",
    }
    if not include_column:
        ignored |= {"column", "source_column", "column_name"}
    return {key: value for key, value in field.support.items() if key not in ignored}


def _component(field: FieldNode) -> str | None:
    return field.axes.get("cost_component") or next(
        (role for role in field.role if role in {"VAL_SH", "VAL_SP", "VAL_UTI", "VAL_TOT"}),
        None,
    )


def _capacity_index(field: FieldNode) -> str | None:
    return (
        field.axes.get("capacity_vector_index")
        or field.axes.get("capacity_index")
        or field.axes.get("capacity_family")
    )


def _diagnostic_topology(field: FieldNode) -> str | None:
    return field.axes.get("icd_topology_role") or field.axes.get("diagnostic_role")


def _protection(left: FieldNode, right: FieldNode) -> ProtectedNonEquivalence | None:
    if left.carrier == right.carrier and left.unit == right.unit:
        left_cost, right_cost = _component(left), _component(right)
        if left_cost and right_cost and left_cost != right_cost:
            return ProtectedNonEquivalence(
                left.id, right.id, "cost_component_identity",
                f"SIH cost components differ: {left_cost} != {right_cost}",
            )
        left_capacity, right_capacity = _capacity_index(left), _capacity_index(right)
        if left_capacity and right_capacity and left_capacity != right_capacity:
            return ProtectedNonEquivalence(
                left.id, right.id, "capacity_vector_identity",
                f"CNES capacity indices differ: {left_capacity} != {right_capacity}",
            )
    left_topology, right_topology = _diagnostic_topology(left), _diagnostic_topology(right)
    if left_topology and right_topology and left_topology != right_topology:
        return ProtectedNonEquivalence(
            left.id, right.id, "diagnostic_topology_identity",
            f"Diagnostic roles differ: {left_topology} != {right_topology}",
        )
    return None


def _signature_payload(field: FieldNode) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
    lineage = field.lineage.model_dump(mode="json")
    if field.operator in {"pi_*", "pi_bound_*", "Pi_Clsf_to_Axis"}:
        kind = "exact_projection_redundancy"
        compared = ("operator", "parent_ids", "operator_params", "axes", "support", "unit", "carrier")
        payload = {
            "operator": field.operator,
            "parent_ids": lineage["parent_ids"],
            "operator_params": lineage["operator_params"],
            "axes": field.axes,
            "support": _support_signature(field, include_column=False),
            "unit": field.unit,
            "carrier": field.carrier,
            "provenance": sorted(field.provenance),
            "registry_versions": lineage["registry_versions"],
        }
    elif field.carrier == "Population" or "denominator" in field.role:
        kind = "shared_denominator_identity"
        compared = ("source", "support", "axes", "unit", "carrier", "provenance", "registry_versions")
        payload = {
            "source": sorted(field.source[:2]),
            "support": _support_signature(field, include_column=False),
            "axes": field.axes,
            "unit": field.unit,
            "carrier": field.carrier,
            "provenance": sorted(field.provenance),
            "registry_versions": lineage["registry_versions"],
        }
    elif field.kind == "observer_proxy" and "diagnostic_topology" in field.role:
        kind = "nested_icd_exact_identity"
        compared = ("diagnostic_topology", "code_set", "support", "source", "provenance")
        payload = {
            "diagnostic_topology": _diagnostic_topology(field),
            "code_set": field.axes.get("code_set") or field.axes.get("icd_code_set") or field.name,
            "support": _support_signature(field, include_column=False),
            "source": sorted(field.source[:2]),
            "provenance": sorted(field.provenance),
        }
    elif field.aggregation == "compositional":
        kind = "compositional_closure_identity"
        compared = ("parent_ids", "operator_params", "axes", "support", "provenance")
        payload = {
            "parent_ids": lineage["parent_ids"],
            "operator_params": lineage["operator_params"],
            "axes": field.axes,
            "support": _support_signature(field, include_column=False),
            "provenance": sorted(field.provenance),
        }
    elif _component(field):
        kind = "cost_component_exact_identity"
        compared = ("cost_component", "source", "support", "axes", "unit", "carrier")
        payload = {
            "cost_component": _component(field),
            "source": sorted(field.source),
            "support": _support_signature(field),
            "axes": field.axes,
            "unit": field.unit,
            "carrier": field.carrier,
        }
    elif _capacity_index(field):
        kind = "capacity_vector_exact_identity"
        compared = ("capacity_index", "source", "support", "axes", "unit", "carrier")
        payload = {
            "capacity_index": _capacity_index(field),
            "source": sorted(field.source),
            "support": _support_signature(field),
            "axes": field.axes,
            "unit": field.unit,
            "carrier": field.carrier,
        }
    else:
        kind = "same_field_semantics"
        compared = (
            "name", "source", "operator", "support", "axes", "unit", "carrier",
            "aggregation", "provenance", "role", "registry_versions",
        )
        payload = {
            "name": field.name,
            "source": sorted(field.source),
            "operator": field.operator,
            "support": _support_signature(field),
            "axes": field.axes,
            "unit": field.unit,
            "carrier": field.carrier,
            "aggregation": field.aggregation,
            "provenance": sorted(field.provenance),
            "role": sorted(field.role),
            "registry_versions": lineage["registry_versions"],
        }
    return kind, content_hash(_stable(payload)), compared, payload


def equivalence_signature(field: FieldNode) -> tuple[str, str]:
    kind, signature, _compared, _payload = _signature_payload(field)
    return kind, signature


def precompress_fields(fields: Iterable[FieldNode]) -> tuple[tuple[FieldNode, ...], PrecompressionReport]:
    source = tuple(fields)
    canonical: list[FieldNode] = []
    by_id: dict[str, FieldNode] = {}
    by_signature: dict[str, FieldNode] = {}
    canonical_by_id: dict[str, str] = {}
    suppressed: list[SuppressedEquivalent] = []
    protected: list[ProtectedNonEquivalence] = []

    for field in source:
        kind, signature, compared, _payload = _signature_payload(field)
        if field.id in by_id:
            target = by_id[field.id]
            kind = "identical_field_id"
            signature = field.id
            compared = ("field_id",)
        else:
            target = by_signature.get(signature)
        if target is not None:
            barrier = _protection(target, field)
            if barrier is not None:
                protected.append(barrier)
                target = None
        if target is not None:
            canonical_by_id[field.id] = target.id
            suppressed.append(SuppressedEquivalent(
                field.id,
                target.id,
                kind,
                signature,
                proof=f"exact equality over {', '.join(compared)}",
                compared_metadata=compared,
            ))
            continue

        # Record safety barriers among semantically adjacent fields even when
        # their signatures already differ because the protected axis differs.
        for other in canonical:
            barrier = _protection(other, field)
            if barrier is not None:
                protected.append(barrier)
        canonical.append(field)
        by_id[field.id] = field
        by_signature[signature] = field
        canonical_by_id[field.id] = field.id

    unique_protected = {
        (item.left_field_id, item.right_field_id, item.protection_kind): item
        for item in protected
    }
    report = PrecompressionReport(
        input_count=len(source),
        output_count=len(canonical),
        suppressed=tuple(suppressed),
        canonical_by_field_id=canonical_by_id,
        protected_non_equivalences=tuple(unique_protected.values()),
    )
    return tuple(canonical), report
