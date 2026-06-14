"""Metadata-only bridge planning for autonomous EFG expansion.

Bridge planning identifies cross-field semantic opportunities and blockers.  It
never materializes tensors; numeric realization remains the responsibility of
explicit compiler stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pegasus.core.hashing import content_hash
from pegasus.efg.core_seed import CoreSeed, build_core_seed_set


@dataclass(frozen=True)
class BridgeCandidate:
    bridge_id: str
    bridge_type: str
    numerator_id: str
    denominator_id: str | None
    required_operator: str
    support_relation: str
    registry_evidence: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "bridge_type": self.bridge_type,
            "numerator_id": self.numerator_id,
            "denominator_id": self.denominator_id,
            "required_operator": self.required_operator,
            "support_relation": self.support_relation,
            "registry_evidence": list(self.registry_evidence),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BridgePlan:
    candidates: tuple[BridgeCandidate, ...]
    blocked: tuple[dict[str, Any], ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "candidate_count": len(self.candidates),
            "blocked_count": len(self.blocked),
            "bridge_type_counts": _counts(candidate.bridge_type for candidate in self.candidates),
            "candidates": [candidate.as_manifest() for candidate in self.candidates],
            "blocked": list(self.blocked),
        }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _bridge_id(kind: str, left: str, right: str | None) -> str:
    suffix = content_hash({"bridge_type": kind, "numerator": left, "denominator": right})[:16]
    return f"bridge::{kind}::{suffix}"


def _candidate(kind: str, numerator: CoreSeed, denominator: CoreSeed | None, *, operator: str = "ratio") -> BridgeCandidate:
    evidence = tuple(sorted(set(numerator.registry_evidence + ((denominator.registry_evidence if denominator else ())))))
    return BridgeCandidate(
        bridge_id=_bridge_id(kind, numerator.field_id, denominator.field_id if denominator else None),
        bridge_type=kind,
        numerator_id=numerator.field_id,
        denominator_id=denominator.field_id if denominator else None,
        required_operator=operator,
        support_relation="requires_alignment_or_transform",
        registry_evidence=evidence,
        warnings=("metadata_only_bridge_candidate",),
    )


def _by_role(seeds: Iterable[CoreSeed], role: str) -> list[CoreSeed]:
    return [seed for seed in seeds if seed.seed_role == role]


def plan_bridge_candidates(
    fields: Iterable[Any],
    *,
    registry_root: str | Path = "config/registries",
    intent: Any = None,
) -> BridgePlan:
    seed_set = build_core_seed_set(fields, registry_root=registry_root, intent=intent)
    seeds = list(seed_set.seeds)
    candidates: list[BridgeCandidate] = []
    blocked: list[dict[str, Any]] = list(seed_set.blocked)

    populations = _by_role(seeds, "population_denominator_seed")
    deaths = _by_role(seeds, "death_event_seed")
    births = _by_role(seeds, "birth_event_seed")
    admissions = _by_role(seeds, "admission_event_seed")
    capacities = _by_role(seeds, "facility_capacity_seed")
    costs = _by_role(seeds, "cost_component_seed")

    for death in deaths:
        for population in populations:
            candidates.append(_candidate("mortality_rate_bridge", death, population))
    for birth in births:
        for population in populations:
            candidates.append(_candidate("birth_rate_bridge", birth, population))
    for admission in admissions:
        for population in populations:
            candidates.append(_candidate("admission_rate_bridge", admission, population))
        for capacity in capacities:
            candidates.append(_candidate("capacity_pressure_bridge", admission, capacity))
    for cost in costs:
        for admission in admissions:
            candidates.append(_candidate("cost_per_admission_bridge", cost, admission))

    if not populations and (deaths or births or admissions):
        blocked.append({"reason": "population_denominator_seed_missing", "affected_event_seed_count": len(deaths) + len(births) + len(admissions)})
    if admissions and not capacities:
        blocked.append({"reason": "facility_capacity_seed_missing_for_admission_bridge", "admission_seed_count": len(admissions)})

    return BridgePlan(candidates=tuple(candidates), blocked=tuple(blocked))


def bridge_summary(plan: BridgePlan) -> dict[str, Any]:
    return plan.as_manifest()


__all__ = ["BridgeCandidate", "BridgePlan", "bridge_summary", "plan_bridge_candidates"]
