"""Metadata support and axis alignment for autonomous EFG expansion."""

from __future__ import annotations

from typing import Any

from pegasus.core.schemas import AlignmentResult, FieldNode
from pegasus.efg.declaration import OperatorSpec


ALIGNMENT_AXES = ("geography", "time", "age", "sex", "race", "diagnostic")

AXIS_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "geography": ("geography", "geo", "municipality", "locality"),
    "time": ("time", "period", "year", "month"),
    "age": ("age", "age_axis", "age_group"),
    "sex": ("sex", "sex_axis"),
    "race": ("race_axis_type", "race_axis", "declaration_process", "race"),
    "diagnostic": ("icd_topology_role", "diagnostic_role", "topology"),
}


def _value(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def _canonical(axis: str, value: Any) -> Any:
    if value is None:
        return None
    token = str(value).strip().lower()
    if axis == "time" and token in {"year", "birth_year", "admission_year", "annual"}:
        return "year"
    if axis == "geography" and token in {
        "mun_residence_cod6", "mun_facility_cod6", "municipality_cod6", "municipality",
    }:
        return "municipality_cod6"
    return value


def _axis_values(field: FieldNode) -> dict[str, Any]:
    support_axes = field.support.get("axes") if isinstance(field.support.get("axes"), dict) else {}
    merged = {**support_axes, **field.axes}
    return {
        axis: _canonical(axis, _value(merged, aliases))
        for axis, aliases in AXIS_KEY_ALIASES.items()
    }


def _support_value(field: FieldNode, *names: str) -> Any:
    value = _value(field.support, tuple(names))
    if value is not None:
        return value
    return _value(field.axes, tuple(names))


def _support_descriptor(left: FieldNode, right: FieldNode, relations: dict[str, str]) -> dict[str, Any]:
    return {
        "support_kind": "efg_aligned_metadata",
        "left_support": left.support,
        "right_support": right.support,
        "relations": relations,
        "geography": _support_value(left, "geography_support", "geography", "geo"),
        "time": _support_value(left, "time_support", "time", "period", "year"),
    }


def align_fields(
    *,
    left: FieldNode,
    right: FieldNode,
    operator: OperatorSpec,
    intent: Any = None,
    geo_support: Any = None,
    registries: Any = None,
) -> AlignmentResult:
    """Compare support and formal axes without fabricating transformed tensors.

    Legal marginalizations are reported as required operations.  Bridge and
    geospatial requirements are explicit failures because 19A does not execute
    those modules.
    """

    del geo_support, registries
    left_axes = _axis_values(left)
    right_axes = _axis_values(right)
    relations: dict[str, str] = {}
    operations: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []

    left_geo = _support_value(left, "geography_support", "geo_mode")
    right_geo = _support_value(right, "geography_support", "geo_mode")
    if left_geo is not None and right_geo is not None and left_geo != right_geo:
        relations["support"] = "geospatial_transform_required"
        failures.append("geospatial_transform_required")
        warnings.append("required_module:geo_support_calculus")
    else:
        relations["support"] = "exact_or_unspecified"

    left_time = _support_value(left, "time_support", "period_range")
    right_time = _support_value(right, "time_support", "period_range")
    if left_time is not None and right_time is not None and left_time != right_time:
        relations["time_support"] = "incompatible"
        failures.append("time_support_incompatible")

    for axis in ALIGNMENT_AXES:
        left_value = left_axes[axis]
        right_value = right_axes[axis]
        if left_value == right_value:
            relations[axis] = "exact"
            continue
        if left_value is None and right_value is not None:
            if axis == "race":
                relations[axis] = "race_bridge_required"
                failures.append("race_bridge_required")
                warnings.append("required_module:Bridge_R")
            elif axis == "diagnostic":
                relations[axis] = "diagnostic_topology_only"
                warnings.append("diagnostic_topology_preserved")
            elif right.aggregation == "additive":
                relations[axis] = "denominator_extra_projectable"
                operations.append(f"project_right:{axis}")
            else:
                relations[axis] = "denominator_extra_incompatible"
                failures.append(f"denominator_axis_not_projectable:{axis}")
            continue
        if left_value is not None and right_value is None:
            if axis == "race":
                relations[axis] = "race_bridge_required"
                failures.append("race_bridge_required")
                warnings.append("required_module:Bridge_R")
            elif axis == "diagnostic":
                relations[axis] = "diagnostic_topology_only"
                warnings.append("diagnostic_topology_preserved")
            elif left.aggregation == "additive":
                relations[axis] = "numerator_extra_projectable"
                operations.append(f"project_left:{axis}")
            else:
                relations[axis] = "numerator_extra_incompatible"
                failures.append(f"numerator_axis_not_projectable:{axis}")
            continue

        if axis == "geography":
            relations[axis] = "geospatial_transform_required"
            failures.append("geospatial_transform_required")
            warnings.append("required_module:geo_support_calculus")
        elif axis == "race":
            relations[axis] = "race_bridge_required"
            failures.append("race_bridge_required")
            warnings.append("required_module:Bridge_R")
        elif axis == "diagnostic":
            relations[axis] = "diagnostic_topology_incompatible"
            failures.append("diagnostic_topology_incompatible")
        else:
            relations[axis] = "incompatible"
            failures.append(f"axis_incompatible:{axis}")

    # Geneallocation cannot be used as an implicit repair for an RN field.
    geo_mode = getattr(intent, "geo_mode", None)
    if isinstance(intent, dict):
        geo_mode = intent.get("geo_mode", geo_mode)
    if failures and geo_mode == "geneallocated" and operator.name == "RN":
        warnings.append("direct_rate_geneallocation_forbidden")

    failures = list(dict.fromkeys(failures))
    warnings = list(dict.fromkeys(warnings))
    return AlignmentResult(
        ok=not failures,
        aligned_left_id=left.id if not failures else None,
        aligned_right_id=right.id if not failures else None,
        operations_applied=operations,
        support_after_alignment=_support_descriptor(left, right, relations),
        warnings=warnings,
        failure_reason=";".join(failures) if failures else None,
    )
