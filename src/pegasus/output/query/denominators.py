"""Multi-denominator declaration + query-time resolution (FEAT-P4).

Reads ``config/registries/health/denominators.yaml``: each quantity declares the denominators that
may normalize it, exactly one flagged ``default``. Resolution is deterministic and recorded — a
per-query override must be admissible, else a typed refusal (never a silent fallback, §V).
See PEGASUS_OUTPUT_QUERY_LAYER.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.registries.loader import load_registry_file

REGISTRY_FILE = "health/denominators.yaml"


class DenominatorRegistryError(ValueError):
    """Raised when a denominator registry is malformed or a resolution is illegal."""


@dataclass(frozen=True)
class DenominatorOption:
    id: str
    default: bool
    strata: tuple[str, ...]
    offset_semantics: str
    legality_class: str

    def as_provenance(self) -> dict[str, Any]:
        return {
            "denominator_id": self.id,
            "denominator_strata": list(self.strata),
            "offset_semantics": self.offset_semantics,
            "legality_class": self.legality_class,
            "was_default": self.default,
        }


@dataclass(frozen=True)
class QuantityDenominators:
    quantity: str
    extensive: bool
    options: tuple[DenominatorOption, ...]

    def default(self) -> DenominatorOption:
        for opt in self.options:
            if opt.default:
                return opt
        raise DenominatorRegistryError(f"quantity {self.quantity!r} has no default denominator")

    def get(self, denominator_id: str) -> DenominatorOption | None:
        return next((o for o in self.options if o.id == denominator_id), None)


def _parse_quantity(quantity: str, spec: dict[str, Any]) -> QuantityDenominators:
    admissible = spec.get("admissible")
    if not isinstance(admissible, list) or not admissible:
        raise DenominatorRegistryError(f"quantity {quantity!r}: 'admissible' must be a non-empty list")
    options: list[DenominatorOption] = []
    seen: set[str] = set()
    for raw in admissible:
        if not isinstance(raw, dict) or "id" not in raw:
            raise DenominatorRegistryError(f"quantity {quantity!r}: each admissible entry needs an 'id'")
        did = str(raw["id"])
        if did in seen:
            raise DenominatorRegistryError(f"quantity {quantity!r}: duplicate denominator id {did!r}")
        seen.add(did)
        options.append(DenominatorOption(
            id=did,
            default=bool(raw.get("default", False)),
            strata=tuple(str(s) for s in raw.get("strata", [])),
            offset_semantics=str(raw.get("offset_semantics", "log_exposure")),
            legality_class=str(raw.get("legality_class", "structural")),
        ))
    n_default = sum(1 for o in options if o.default)
    if n_default != 1:
        raise DenominatorRegistryError(
            f"quantity {quantity!r}: exactly one denominator must be default (found {n_default})"
        )
    return QuantityDenominators(quantity=quantity, extensive=bool(spec.get("extensive", True)), options=tuple(options))


def load_denominator_registry(*, root: str | Path = "config/registries") -> dict[str, QuantityDenominators]:
    """Load + validate the denominator registry. Raises on malformed entries (one-default rule,
    non-empty admissible, unique ids)."""
    payload = load_registry_file(Path(root) / REGISTRY_FILE)
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DenominatorRegistryError("denominators.yaml: 'entries' must be a non-empty list")
    out: dict[str, QuantityDenominators] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            raise DenominatorRegistryError("denominators.yaml: each entry needs an 'id'")
        if str(entry.get("status", "active")) != "active":
            continue  # inactive quantities are declared but not resolvable
        quantity = str(entry["id"])
        if quantity in out:
            raise DenominatorRegistryError(f"denominators.yaml: duplicate quantity id {quantity!r}")
        out[quantity] = _parse_quantity(quantity, entry)
    return out


def resolve_denominator(quantity: str, spec, *, root: str | Path = "config/registries") -> DenominatorOption:
    """Resolve the denominator for a quantity: the per-query override if given (must be admissible),
    else the declared default. ``spec`` is a QuerySpec (or anything with a ``denominator`` attribute)."""
    registry = load_denominator_registry(root=root)
    if quantity not in registry:
        raise DenominatorRegistryError(
            f"no denominator declaration for quantity {quantity!r}; have {sorted(registry)}"
        )
    q = registry[quantity]
    override = getattr(spec, "denominator", None)
    if override is not None:
        opt = q.get(str(override))
        if opt is None:
            raise DenominatorRegistryError(
                f"denominator {override!r} is not admissible for {quantity!r}; "
                f"admissible: {[o.id for o in q.options]}"
            )
        return opt
    return q.default()


__all__ = [
    "REGISTRY_FILE",
    "DenominatorRegistryError",
    "DenominatorOption",
    "QuantityDenominators",
    "load_denominator_registry",
    "resolve_denominator",
]
