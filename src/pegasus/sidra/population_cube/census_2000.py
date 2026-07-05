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


__all__ = [
    "CLEAN_AGE_BRACKETS_2093",
    "BracketDisaggregation",
    "bracket_single_year_labels",
    "disaggregate_bracket",
    "disaggregate_2000_strata_to_single_year",
]
