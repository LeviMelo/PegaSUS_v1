"""Disease variable grammar — the concept grammar that instantiates the LDO variable
set by crossing carriers with disease concepts (§5.1 / MSD-III §II.6).

Today's σ_C restricts one field at a time; the LDO consumes the *unfolded* view: a
carrier (Deaths, HospitalAdmissions, LiveBirths, …) crossed with a disease selector
at a chosen resolution becomes a distinct variable ``X_1..X_p``. Resolution is a
first-class parameter (the disease axis of §II.7 multi-resolution):

    variable = carrier × disease_selector(resolution) × topology_role

Contracts honoured here:
- **SIM chain semantics.** ``topology_role ∈ {underlying_cause, mention, associated}``
  is part of the variable identity — a disease as an underlying cause is NOT the same
  variable as the same disease as a mention (§5.1).
- **Anti-silence.** A code that resolves to no concept at the resolution is routed to an
  explicit ``…_OTHER`` residual variable, never dropped.
- **Provenance/projection typing.** Each variable carries ``code_system`` and the worst
  ``projection_status`` across its member concepts (CCSR on CID-10 ⇒ ``approximate`` ⇒
  the downstream field enters at ``state ≤ fragile``, §II.13.1). Multi-label is preserved:
  at ``concept:<family>`` resolution a code may feed several variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import polars as pl

from pegasus.disease import concept_registry, icd_adapter

_HIERARCHY = {"chapter", "block", "category"}

# projection severity — a variable inherits the WORST of its members (anti-false-precision).
_PROJECTION_SEVERITY = {
    "exact": 0, "source_system_specific": 1, "parent_projection": 2,
    "approximate": 3, "unmappable": 4,
}


@dataclass(frozen=True)
class DiseaseVariable:
    """One instantiated LDO variable = carrier × disease concept × topology role."""

    variable_id: str            # "Deaths|underlying_cause|chapter_IX"
    carrier: str
    topology_role: str          # underlying_cause | mention | associated
    concept_id: str             # chapter_IX | block_I20-I25 | category_I21 | ccsr_… | …_OTHER
    resolution: str             # chapter | block | category | concept:<family>
    code_system: str
    projection_status: str      # worst across member concepts
    member_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class DiseaseStratification:
    """The result of stratifying a carrier's events across the disease axis."""

    variables: tuple[DiseaseVariable, ...]
    counts: pl.DataFrame        # long form: [variable_id, <geo_col>, <time_col>, count]


@lru_cache(maxsize=200_000)
def _concepts_for_code(code: str, resolution: str, code_system: str, registry_root: str
                       ) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """(concept_id, projection_status, warnings) for a code at ``resolution``.

    Hierarchy resolutions return exactly one concept (or none → the caller routes to
    OTHER); ``concept:<family>`` returns zero-or-more (multi-label preserved).
    """
    if resolution in _HIERARCHY:
        info = icd_adapter.code_info(code)
        value = {"chapter": info.chapter, "block": info.block, "category": info.category}[resolution]
        if not value:
            return ()
        status = "exact" if (resolution != "category" or info.status == "who_icd10") else "source_system_specific"
        return ((f"{resolution}_{value}", status, ()),)
    if resolution.startswith("concept:"):
        family = resolution.split(":", 1)[1]
        out: list[tuple[str, str, tuple[str, ...]]] = []
        for a in concept_registry.assertions_for_code(code, code_system=code_system, registry_root=registry_root):
            if family in ("*", a.concept_family):
                out.append((a.concept_id, a.projection_status, tuple(a.warnings)))
        return tuple(out)
    raise ValueError(f"unknown disease resolution {resolution!r} (expected chapter|block|category|concept:<family>)")


def _worst(statuses: list[str]) -> str:
    return max(statuses, key=lambda s: _PROJECTION_SEVERITY.get(s, 0)) if statuses else "exact"


def stratify_events(
    events: pl.DataFrame,
    *,
    carrier: str,
    topology_role: str,
    resolution: str,
    code_col: str,
    geo_col: str = "municipality_cod6",
    time_col: str = "time",
    code_system: str = "CID-10",
    registry_root: str = "config/registries",
    other_label: str = "OTHER",
) -> DiseaseStratification:
    """Cross ``carrier`` events with disease concepts at ``resolution`` → variables + counts.

    Each event row (an occurrence carrying an ICD ``code_col`` at ``geo_col``×``time_col``)
    is routed to every concept its code holds at the resolution (multi-label ⇒ several),
    or to the ``…_OTHER`` residual. Returns the long-form count series and the typed
    variable set. The count of a variable at a cell is the number of qualifying events.
    """
    def vid(concept_id: str) -> str:
        return f"{carrier}|{topology_role}|{concept_id}"

    other_concept = f"{resolution.replace('concept:', '')}_{other_label}"
    unique_codes = [c for c in events[code_col].unique().to_list() if c is not None]

    # code → member concepts (cached per (code, resolution)); build the explode map + specs.
    map_rows: list[dict] = []
    spec: dict[str, dict] = {}
    for code in unique_codes:
        concepts = _concepts_for_code(code, resolution, code_system, registry_root)
        targets = [(c, s, w) for c, s, w in concepts] or [(other_concept, "exact", ())]
        for concept_id, status, warns in targets:
            map_rows.append({code_col: code, "concept_id": concept_id})
            s = spec.setdefault(concept_id, {"codes": set(), "statuses": [], "warnings": set()})
            s["codes"].add(code)
            s["statuses"].append(status)
            s["warnings"].update(warns)

    variables = tuple(
        DiseaseVariable(
            variable_id=vid(cid), carrier=carrier, topology_role=topology_role,
            concept_id=cid, resolution=resolution, code_system=code_system,
            projection_status=_worst(info["statuses"]),
            member_codes=tuple(sorted(info["codes"])),
            warnings=tuple(sorted(info["warnings"])),
        )
        for cid, info in sorted(spec.items())
    )

    if not map_rows:
        empty = pl.DataFrame(schema={"variable_id": pl.Utf8, geo_col: pl.Utf8, time_col: pl.Int64, "count": pl.UInt32})
        return DiseaseStratification(variables=variables, counts=empty)

    map_df = pl.DataFrame(map_rows).with_columns(pl.col(code_col).cast(events.schema[code_col]))
    counts = (
        events.join(map_df, on=code_col)  # multi-label → row duplication (correct: overlap is known structure)
        .with_columns((pl.lit(f"{carrier}|{topology_role}|") + pl.col("concept_id")).alias("variable_id"))
        .group_by(["variable_id", geo_col, time_col]).len()
        .rename({"len": "count"})
        .sort(["variable_id", geo_col, time_col])
    )
    return DiseaseStratification(variables=variables, counts=counts)


def assemble_disease_field(
    strat: DiseaseStratification, *, geo_col: str = "municipality_cod6",
    time_col: str = "time", resolution_label: str = "month",
):
    """Pivot the long-form stratified counts into an ``LDOField`` (X, W) ready for ``run_ldo``.

    Every observed cell carries full weight; unobserved cells are NaN/zero-weight (the
    latent-field convention — sparse disease variables inform only where seen).
    """
    from pegasus.ldo.assemble import LDOField  # lazy: keep disease/ import-light of the LDO
    import numpy as np

    counts = strat.counts
    variables = tuple(v.variable_id for v in strat.variables)
    var_index = {v: i for i, v in enumerate(variables)}
    space_ids = tuple(sorted(str(s) for s in counts[geo_col].unique().to_list() if s is not None))
    time_ids = tuple(sorted(int(t) for t in counts[time_col].unique().to_list() if t is not None))
    s_index = {s: i for i, s in enumerate(space_ids)}
    t_index = {t: i for i, t in enumerate(time_ids)}
    p, S, T = len(variables), len(space_ids), len(time_ids)

    X = np.full((p, S, T), np.nan, dtype=np.float64)
    for row in counts.iter_rows(named=True):
        vi = var_index.get(row["variable_id"])
        si = s_index.get(str(row[geo_col]))
        ti = t_index.get(int(row[time_col]))
        if vi is not None and si is not None and ti is not None:
            X[vi, si, ti] = float(row["count"])
    W = np.where(np.isnan(X), 0.0, 1.0)
    return LDOField(variables=variables, space_ids=space_ids, time_ids=time_ids, X=X, W=W, resolution=resolution_label)


def variable_meta(variables) -> dict[str, dict]:
    """Build the ``run_ldo(variable_meta=...)`` map from generated variables.

    ``variable_id -> {code_set, code_system, topology_role, projection_status}`` — feeds
    the LDO's disease-provenance annotation and the shared-code overlap guard (§5.3).
    """
    return {
        v.variable_id: {
            "code_set": frozenset(v.member_codes),
            "code_system": v.code_system,
            "topology_role": v.topology_role,
            "projection_status": v.projection_status,
        }
        for v in variables
    }


__all__ = [
    "DiseaseVariable", "DiseaseStratification", "stratify_events",
    "assemble_disease_field", "variable_meta",
]
