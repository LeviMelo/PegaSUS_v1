"""SIDRA metadata ensurers (read-cached-or-fetch-live).

The ensurers physically live in :mod:`sidra_population` because population
acquisition and metadata acquisition are mutually dependent (the population
acquirers call the ensurers, and the ensurers are keyed by the population-table
constants). This module re-exports them so callers can address the metadata
layer by its own name without importing the whole population module surface.
"""

from __future__ import annotations

from pegasus.workflows.acquire.sidra_population import (
    _ensure_sidra_metadata,
    _ensure_sidra_metadata_tables,
)

__all__ = [
    "_ensure_sidra_metadata",
    "_ensure_sidra_metadata_tables",
]
