"""Output Query Layer (FEAT-P3 + FEAT-P4).

A first-class, user-invokable reader over an emitted run bundle that serves derived,
provenance-carrying datasets — rates (count / resolved denominator), directly age-standardized
rates, certified-edge tables, raw fields — in a chosen format. Additive over the bundle; it does
not touch the inference core. See PEGASUS_OUTPUT_QUERY_LAYER.md.
"""

from __future__ import annotations

from pegasus.output.query.denominators import (
    DenominatorOption,
    DenominatorRegistryError,
    QuantityDenominators,
    load_denominator_registry,
    resolve_denominator,
)
from pegasus.output.query.spec import QuerySpec

__all__ = [
    "QuerySpec",
    "DenominatorOption",
    "QuantityDenominators",
    "DenominatorRegistryError",
    "load_denominator_registry",
    "resolve_denominator",
]
