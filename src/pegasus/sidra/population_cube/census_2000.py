"""2000-census demographic strata from SIDRA table 2093, disaggregated to the single-year axis.

FAL-POP #2/#3 (MSD-III §II.4). SIDRA 9606 carries only the 2010 and 2022 censuses; the 2000 census
age×sex×race breakdown lives only in table **2093**, which reports age as overlapping *brackets*
(0-4, 5-9, …, plus roll-ups 0-14, 15-64, 65+, 70+ …). §II.4: "its overlapping age brackets are a CTR
disaggregation instance (a clean partition selected, roll-ups treated as derived, never summed)."

Two steps here:
  1. Select a **clean, non-overlapping partition** of 2093's brackets covering 0..100+ — the roll-up
     categories are excluded (never summed).
  2. **CTR disaggregation** — distribute each 2000 bracket total across its single-year ages in
     proportion to a reference single-year *shape* (the 2010 census single-year distribution from
     9606, within the same bracket), preserving the bracket total exactly. This yields a 2000
     single-year age×sex×race anchor on the SAME axis as 9606's 2010/2022, so all three censuses
     cohort-project on one lattice.

The disaggregation is a shape prior, not new information: it never invents a bracket total, only its
within-bracket single-year profile, borrowed from the nearest census that resolves single years.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# The clean, contiguous, non-overlapping partition of SIDRA 2093 clsf-58 age brackets covering 0..100+.
# {category_id: (age_low, age_high)}; age_high >= 100 denotes the open-ended top bracket (80 anos ou
# mais). The excluded ids (0 Total, 100402 0-14, 6797 15-64, 1143 15-19, 6798 65+, 3244 70+) are
# roll-ups of these and are NEVER selected or summed (§II.4).
CLEAN_AGE_BRACKETS_2093: dict[str, tuple[int, int]] = {
    "1140": (0, 4),      # 0 a 4 anos
    "1141": (5, 9),      # 5 a 9 anos
    "1142": (10, 14),    # 10 a 14 anos
    "2792": (15, 17),    # 15 a 17 anos
    "92982": (18, 19),   # 18 e 19 anos
    "1144": (20, 24),    # 20 a 24 anos
    "1145": (25, 29),    # 25 a 29 anos
    "3299": (30, 39),    # 30 a 39 anos
    "3300": (40, 49),    # 40 a 49 anos
    "3301": (50, 59),    # 50 a 59 anos
    "3520": (60, 69),    # 60 a 69 anos
    "95252": (70, 79),   # 70 a 79 anos
    "2503": (80, 200),   # 80 anos ou mais (open-ended top)
}

_TOP_AGE = 100  # canonical axis tops out at age_100_plus

# SIDRA 2093 clsf-86 category for "Cor ou raça: Sem declaração" (undeclared race). Enumerated people
# who did not declare race -- missing data on the race axis, reallocated by local composition
# (§II.5 FAL-POP-RECON), NEVER dropped.
SIDRA_2093_UNDECLARED_RACE = "2781"


def reconcile_undeclared_race(
    declared_profiles: dict[tuple[str, str, str], dict[str, float]],
    undeclared_profiles: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """Reallocate undeclared-race census mass into the declared races by local composition (§II.5).

    ``declared_profiles[(locality, sex, race)] = {bracket_id: count}`` for the declared races;
    ``undeclared_profiles[(locality, sex)] = {bracket_id: undeclared_count}`` for "Sem declaração".

    For each ``(locality, sex, bracket)`` the undeclared count is distributed across the declared races
    in proportion to the **local declared composition** `π_local[race | locality, sex, bracket]` — the
    max-entropy allocation given the margins, preserving the person's declared sex and age bracket and
    imputing only race. Where a cell is entirely undeclared (no declared mass to form `π`), a
    **hierarchical fallback** borrows the composition from `(locality, sex)`, then `(locality)`, then
    the whole panel (small-area strength-borrowing, §III.3). Mutates ``declared_profiles`` in place so
    it now sums to the complete enumerated total; returns telemetry. The undeclared bin is thereby
    reconciled, never deleted (the prime directive — missingness is never silence)."""
    declared_races = sorted({race for (_l, _s, race) in declared_profiles})
    if not declared_races:
        return {"undeclared_reallocated": 0.0, "undeclared_dropped_no_declared": float(
            sum(v for prof in undeclared_profiles.values() for v in prof.values())
        )}

    # Fallback compositions over the declared races, precomputed once.
    ls_comp: dict[tuple[str, str], dict[str, float]] = {}
    l_comp: dict[str, dict[str, float]] = {}
    g_comp: dict[str, float] = {race: 0.0 for race in declared_races}
    for (loc, sex, race), prof in declared_profiles.items():
        tot = float(sum(prof.values()))
        ls_comp.setdefault((loc, sex), {}); ls_comp[(loc, sex)][race] = ls_comp[(loc, sex)].get(race, 0.0) + tot
        l_comp.setdefault(loc, {}); l_comp[loc][race] = l_comp[loc].get(race, 0.0) + tot
        g_comp[race] += tot

    def _norm(counts: dict[str, float]) -> dict[str, float] | None:
        s = float(sum(counts.get(r, 0.0) for r in declared_races))
        return {r: counts.get(r, 0.0) / s for r in declared_races} if s > 0.0 else None

    g_weights = _norm(g_comp) or {r: 1.0 / len(declared_races) for r in declared_races}

    reallocated = 0.0
    fallback_cells = 0
    for (loc, sex), uprof in undeclared_profiles.items():
        for bracket, u in uprof.items():
            if u <= 0.0:
                continue
            local = {r: declared_profiles.get((loc, sex, r), {}).get(bracket, 0.0) for r in declared_races}
            weights = _norm(local)
            if weights is None:
                weights = _norm(ls_comp.get((loc, sex), {})) or _norm(l_comp.get(loc, {})) or g_weights
                fallback_cells += 1
            for r in declared_races:
                cell = declared_profiles.setdefault((loc, sex, r), {})
                cell[bracket] = cell.get(bracket, 0.0) + u * weights[r]
            reallocated += float(u)
    return {"undeclared_reallocated": reallocated, "fallback_cells": float(fallback_cells)}


def bracket_single_year_labels(age_low: int, age_high: int) -> list[str]:
    """Canonical single-year ``age_group`` labels spanned by a bracket. An open-ended top bracket
    (``age_high >= 100``) spans ``age_low..age_99`` plus ``age_100_plus``."""
    if age_high >= _TOP_AGE:
        return [f"age_{a}" for a in range(age_low, _TOP_AGE)] + ["age_100_plus"]
    return [f"age_{a}" for a in range(age_low, age_high + 1)]


@dataclass(frozen=True)
class BracketDisaggregation:
    """One bracket's single-year split. ``by_age`` sums to ``bracket_total`` (closure preserved)."""
    by_age: dict[str, float]
    used_uniform_fallback: bool


def disaggregate_bracket(
    bracket_total: float, age_low: int, age_high: int, reference_shape: dict[str, float]
) -> BracketDisaggregation:
    """CTR-disaggregate one 2000 bracket total to single-year ages, proportional to
    ``reference_shape`` (single-year populations from the nearest single-year census, e.g. 2010)
    within the bracket. The bracket total is preserved exactly; if the reference has no positive
    mass in the bracket (a tiny municipality), fall back to a uniform split (flagged)."""
    labels = bracket_single_year_labels(age_low, age_high)
    weights = np.array([max(float(reference_shape.get(lbl, 0.0)), 0.0) for lbl in labels], dtype=np.float64)
    total_w = float(weights.sum())
    uniform = total_w <= 0.0
    if uniform:
        weights = np.ones(len(labels), dtype=np.float64)
        total_w = float(len(labels))
    split = {lbl: float(bracket_total) * float(w) / total_w for lbl, w in zip(labels, weights)}
    return BracketDisaggregation(by_age=split, used_uniform_fallback=uniform)


def disaggregate_2000_strata_to_single_year(
    *,
    bracket_counts: dict[str, float],
    reference_single_year: dict[str, float],
) -> dict[str, float]:
    """Disaggregate a full 2000 (age-bracket) profile — for one (locality, sex, race) — to the
    single-year axis. ``bracket_counts`` maps clean-partition bracket category ids → 2000 counts;
    ``reference_single_year`` maps canonical single-year labels → the 2010 shape for that same
    (locality, sex, race). Returns canonical single-year label → 2000 count; each bracket's total is
    preserved, so summing the output equals summing the (clean-partition) bracket inputs."""
    out: dict[str, float] = {}
    for cat_id, total in bracket_counts.items():
        span = CLEAN_AGE_BRACKETS_2093.get(cat_id)
        if span is None:  # a roll-up or unknown id — never summed into the partition
            continue
        dis = disaggregate_bracket(float(total), span[0], span[1], reference_single_year)
        for lbl, v in dis.by_age.items():
            out[lbl] = out.get(lbl, 0.0) + v
    return out


def assemble_2000_single_year_records(
    *,
    bracket_profiles: dict[tuple[str, str, str], dict[str, float]],
    reference_2010: dict[tuple[str, str, str], dict[str, float]],
) -> list[dict]:
    """Assemble 2000-census single-year age×sex×race records from parsed 2093 bracket profiles.

    ``bracket_profiles[(locality, sex, race)] = {bracket_category_id: 2000_count}`` (clean partition).
    ``reference_2010[(locality, sex, race)] = {single_year_label: 2010_count}`` supplies the shape.
    Each (locality, sex, race) is disaggregated independently; the shape falls back — specific cell →
    locality-marginal (summed over sex/race) → uniform — so a cell with no matching 2010 mass still
    resolves. Returns records shaped exactly like the build's 9606 records (period ``"2000"``), so the
    caller appends them to the census record set and the existing solver cohort-ages across 2000/2010/2022.
    """
    # locality-marginal 2010 shape (summed over sex/race) as the first fallback.
    loc_marginal: dict[str, dict[str, float]] = {}
    for (loc, _sex, _race), shape in reference_2010.items():
        acc = loc_marginal.setdefault(loc, {})
        for age_label, v in shape.items():
            acc[age_label] = acc.get(age_label, 0.0) + float(v)

    records: list[dict] = []
    for (loc, sex, race), brackets in bracket_profiles.items():
        shape = reference_2010.get((loc, sex, race)) or loc_marginal.get(loc, {})
        single = disaggregate_2000_strata_to_single_year(bracket_counts=brackets, reference_single_year=shape)
        for age_label, value in single.items():
            records.append({
                "municipality_cod6": loc, "period": "2000",
                "age_group": age_label, "sex": sex, "race": race, "value": float(value),
            })
    return records


__all__ = [
    "CLEAN_AGE_BRACKETS_2093",
    "SIDRA_2093_UNDECLARED_RACE",
    "BracketDisaggregation",
    "bracket_single_year_labels",
    "disaggregate_bracket",
    "disaggregate_2000_strata_to_single_year",
    "assemble_2000_single_year_records",
    "reconcile_undeclared_race",
]
