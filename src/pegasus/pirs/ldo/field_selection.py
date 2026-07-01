"""Analytical-variable selection for the LDO (MSD-II §II.6.1).

The EFG materializes both *analytical quantities* (counts, rates, shares,
functionals, divergences, capacity/cost components) and *raw source-column
passthroughs / support axes* (identifiers like ``municipality_cod6``/``year``,
demographic axes like ``sex``/``race``/``age``, and raw per-record marks). Only the
former are variables the dependency operator should correlate — the latter are
support/stratifier axes, not epidemiological signals, and feeding them to the
precision estimator both pollutes the graph with meaningless edges and inflates the
``O(p^3)`` eigensolves that dominate the LDO's cost.

A field is *analytical* iff it carries at least one role token that is not a purely
structural marker (``source_field``, ``substrate_materialized``, a ``*_axis`` /
``*_mark`` / ``*_candidate`` role, or a geography/identifier role). This is a
role-registry decision, not a name heuristic, so adding a new analytical field is a
registry edit rather than an engine edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

# Role tokens that mark a field as a raw passthrough / support axis, not an
# analytical quantity. A field whose roles are *entirely* structural is excluded.
_STRUCTURAL_ROLES: frozenset[str] = frozenset({
    "source_field",
    "substrate_materialized",
    "time_axis_candidate",
    "stratifier",
    "geography_axis",
    "sex_axis",
    "newborn_sex_axis",
    "administrative_race_axis",
    "residence",
    "occurrence",
})
_STRUCTURAL_SUFFIXES: tuple[str, ...] = ("_mark", "_axis", "_candidate")


def _roles(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value]
    try:
        parsed = json.loads(str(value))
        return [str(t) for t in parsed] if isinstance(parsed, list) else [str(value)]
    except Exception:
        return [str(value)]


def is_analytical_role(role: object) -> bool:
    """True iff at least one role token is a genuine analytical quantity."""
    tokens = _roles(role)
    return any(
        (t not in _STRUCTURAL_ROLES) and not t.endswith(_STRUCTURAL_SUFFIXES)
        for t in tokens
    )


def analytical_variable_ids(run_dir: str | Path) -> set[str] | None:
    """Read a compiled run's ``V_fields.parquet`` and return the analytical field ids.

    Returns ``None`` if the field table is unavailable (caller keeps all variables).
    """
    path = Path(run_dir) / "V_fields.parquet"
    if not path.exists():
        return None
    vf = pl.read_parquet(path)
    if "field_id" not in vf.columns or "role" not in vf.columns:
        return None
    keep: set[str] = set()
    for row in vf.select(["field_id", "role"]).iter_rows(named=True):
        if is_analytical_role(row["role"]):
            keep.add(str(row["field_id"]))
    return keep or None


__all__ = ["analytical_variable_ids", "is_analytical_role"]
