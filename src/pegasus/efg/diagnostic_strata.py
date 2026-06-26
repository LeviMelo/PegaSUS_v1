"""ICD diagnostic traversal (MSD §3.11) and health-seed enforcement (§3.10 intent contract).

The autonomous EFG admits the underlying-cause / principal-diagnosis ICD field as a
*non-aggregable diagnostic-topology observer* (``unit == "ICD10"``).  That field must
never be summed: an ICD code is a label, not a quantity.  The legal derived measure is
the **σ_C restriction** — a *cause-specific event count* whose carrier is the event
carrier (``Deaths`` / ``HospitalAdmissions``) and whose support declares an
``icd_chapter`` / ``icd_block`` stratification axis.  Counting records *within* an ICD
group is what turns the forbidden observer into a legal additive measure, after which a
Radon–Nikodym ratio against the population denominator yields cause-specific mortality
(or cause-specific hospitalization) rates.

This module is metadata-only.  It plans which strata to construct (gated by the intent's
``health_seeds``) and audits whether the produced field set satisfies the requested
intent contract.  The physical realization of the restriction lives in the executor
(``efg.executor._count_tensor``), which maps each record's ICD code to its chapter/block
via the registry-backed grouper (``datasus.icd_groups``) and counts within strata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


# health_seed token -> physical stratification level used by the executor grouper.
_ICD_SEED_LEVELS: dict[str, str] = {
    "icd_chapter": "chapter",
    "icd_block": "block",
    "icd_curated": "curated",
    "curated_cause": "curated",
}

# stratification level -> declared support/axis name on the derived count.
ICD_AXIS_BY_LEVEL: dict[str, str] = {
    "chapter": "icd_chapter",
    "block": "icd_block",
    "curated": "curated_cause_group",
}

# Only *primary, single-valued* diagnostic positions are restrictable into a partition
# of the event population.  Associated/secondary chains are multi-valued (one record can
# carry several codes) and would double count, so they are deliberately excluded here.
PRIMARY_DIAGNOSTIC_ROLES: frozenset[str] = frozenset({
    "underlying_cause",
    "sih_principal_diagnosis",
})


def _intent_attr(intent: Any, name: str, default: Any) -> Any:
    if intent is None:
        return default
    if isinstance(intent, dict):
        return intent.get(name, default)
    return getattr(intent, name, default)


def health_seeds(intent: Any) -> list[str]:
    seeds = _intent_attr(intent, "health_seeds", []) or []
    return [str(seed) for seed in seeds]


def requested_icd_levels(intent: Any) -> list[tuple[str, str]]:
    """Return ``[(level, seed_token), ...]`` for each ICD stratification the intent requests."""
    seeds = set(health_seeds(intent))
    seen: set[str] = set()
    levels: list[tuple[str, str]] = []
    for token in ("icd_chapter", "icd_block", "icd_curated", "curated_cause"):
        if token in seeds:
            level = _ICD_SEED_LEVELS[token]
            if level not in seen:
                seen.add(level)
                levels.append((level, token))
    return levels


def _field_attr(field: Any, name: str, default: Any = None) -> Any:
    if isinstance(field, dict):
        return field.get(name, default)
    return getattr(field, name, default)


def diagnostic_columns_by_event(roots: Iterable[Any]) -> dict[tuple[str, str], str]:
    """Map ``(artifact_path, carrier) -> icd_column`` for primary diagnostic observers.

    These are the ICD10-unit roots the event-count loop excludes; we use their source
    column to drive the σ_C restriction over the *same* artifact and carrier.
    """
    mapping: dict[tuple[str, str], str] = {}
    for field in roots:
        unit = str(_field_attr(field, "unit", ""))
        axes = dict(_field_attr(field, "axes", {}) or {})
        role = _field_attr(field, "axes", {}) or {}
        topology_role = axes.get("icd_topology_role") or axes.get("diagnostic_role")
        if unit != "ICD10" and "diagnostic_topology" not in set(_field_attr(field, "role", []) or []):
            continue
        if topology_role not in PRIMARY_DIAGNOSTIC_ROLES:
            continue
        support = dict(_field_attr(field, "support", {}) or {})
        column = support.get("column")
        artifact = support.get("artifact_path")
        carrier = str(_field_attr(field, "carrier", ""))
        if not column or not artifact or not carrier:
            continue
        # First primary observer per (artifact, carrier) wins; underlying_cause and
        # sih_principal_diagnosis never share an (artifact, carrier) key.
        mapping.setdefault((str(artifact), carrier), str(column))
    return mapping


# ---------------------------------------------------------------------------
# Health-seed / intent-contract enforcement
# ---------------------------------------------------------------------------

class HealthSeedContractError(ValueError):
    """Raised when an intent's ``health_seeds`` are not satisfied by the compiled EFG."""


def _field_roles(field: Any) -> set[str]:
    return {str(role) for role in (_field_attr(field, "role", []) or [])}


def _field_support(field: Any) -> dict[str, Any]:
    return dict(_field_attr(field, "support", {}) or {})


def _field_axes(field: Any) -> dict[str, Any]:
    return dict(_field_attr(field, "axes", {}) or {})


def _has_event_count(fields: Iterable[Any], carrier: str) -> bool:
    for field in fields:
        if str(_field_attr(field, "carrier", "")) != carrier:
            continue
        if "source_event_count" in _field_roles(field):
            return True
    return False


def _has_icd_stratum(fields: Iterable[Any], level: str) -> bool:
    axis = ICD_AXIS_BY_LEVEL[level]
    for field in fields:
        support = _field_support(field)
        if support.get("stratify_icd") == level:
            return True
        if axis in _field_axes(field):
            return True
    return False


def _has_capacity(fields: Iterable[Any]) -> bool:
    for field in fields:
        carrier = str(_field_attr(field, "carrier", ""))
        roles = _field_roles(field)
        if carrier == "Facilities" and "source_event_count" in roles:
            return True
        if "cnes_capacity_component" in roles:
            return True
    return False


def seed_satisfaction(intent: Any, fields: Iterable[Any]) -> dict[str, bool]:
    """Compute, per requested ``health_seed``, whether a producing field exists."""
    field_list = list(fields)
    checks: dict[str, Any] = {
        "all_cause_mortality": lambda: _has_event_count(field_list, "Deaths"),
        "icd_chapter": lambda: _has_icd_stratum(field_list, "chapter"),
        "icd_block": lambda: _has_icd_stratum(field_list, "block"),
        "icd_curated": lambda: _has_icd_stratum(field_list, "curated"),
        "curated_cause": lambda: _has_icd_stratum(field_list, "curated"),
        "hospitalization": lambda: _has_event_count(field_list, "HospitalAdmissions"),
        "capacity": lambda: _has_capacity(field_list),
    }
    result: dict[str, bool] = {}
    for seed in health_seeds(intent):
        check = checks.get(seed)
        # Seeds without a registered capability check are treated as satisfied here
        # rather than silently failing the compile on an unmodelled token.
        result[seed] = bool(check()) if check is not None else True
    return result


def enforce_health_seeds(intent: Any, fields: Iterable[Any]) -> dict[str, Any]:
    """Fail the compile loudly when requested ``health_seeds`` are not produced.

    This ends the "hollow success" regime where the EFG reported success while
    delivering none of the analytical surface the intent contracted for.
    """
    satisfaction = seed_satisfaction(intent, fields)
    unsatisfied = sorted(seed for seed, ok in satisfaction.items() if not ok)
    summary = {
        "requested_seeds": sorted(satisfaction),
        "satisfied_seeds": sorted(seed for seed, ok in satisfaction.items() if ok),
        "unsatisfied_seeds": unsatisfied,
        "satisfaction": satisfaction,
    }
    if unsatisfied:
        raise HealthSeedContractError(
            "EFG did not satisfy intent health_seeds "
            f"{unsatisfied}; produced fields cover {summary['satisfied_seeds']}. "
            "Refusing to report a hollow compile success."
        )
    return summary


__all__ = [
    "HealthSeedContractError",
    "ICD_AXIS_BY_LEVEL",
    "PRIMARY_DIAGNOSTIC_ROLES",
    "diagnostic_columns_by_event",
    "enforce_health_seeds",
    "health_seeds",
    "requested_icd_levels",
    "seed_satisfaction",
]
