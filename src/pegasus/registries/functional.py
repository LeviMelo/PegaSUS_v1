"""Statistical-functional field registry (MSD §3.10.4–§3.10.6 Ψ operators).

Read boundary for ``fields/functional_fields.yaml`` — the single source of truth for which
per-record marks (length of stay, costs, reporting delay) are summarized by a mean/median
functional over a support cell. The EFG builds these from the registry; no per-source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pegasus.registries.generic import active_entries


REGISTRY_FILE = "fields/functional_fields.yaml"
ALLOWED_FUNCTIONALS = frozenset({"mean", "median"})


@dataclass(frozen=True)
class FunctionalFieldSpec:
    field_id: str
    source_system: str
    carrier: str
    mark_column: str
    functional: str
    unit: str
    role: str


@lru_cache(maxsize=8)
def _specs_cached(root: str) -> tuple[FunctionalFieldSpec, ...]:
    specs: list[FunctionalFieldSpec] = []
    for entry in active_entries(REGISTRY_FILE, root=root, required=False):
        p = entry.payload
        functional = str(p.get("functional") or "mean")
        carrier = p.get("carrier")
        mark = p.get("mark_column")
        if not carrier or not mark or functional not in ALLOWED_FUNCTIONALS:
            continue
        specs.append(FunctionalFieldSpec(
            field_id=str(p.get("id") or entry.id),
            source_system=str(p.get("source_system") or ""),
            carrier=str(carrier),
            mark_column=str(mark),
            functional=functional,
            unit=str(p.get("unit") or "value"),
            role=str(p.get("role") or "statistical_functional"),
        ))
    return tuple(specs)


def functional_field_specs(*, registry_root: str | Path = "config/registries") -> tuple[FunctionalFieldSpec, ...]:
    return _specs_cached(str(registry_root))


__all__ = ["FunctionalFieldSpec", "functional_field_specs", "ALLOWED_FUNCTIONALS"]
