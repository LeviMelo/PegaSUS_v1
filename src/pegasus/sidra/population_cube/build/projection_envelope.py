"""FAL-POP-PROJ native-projection envelope: per-year anchor class + horizon-growing uncertainty (§II.4)."""

from __future__ import annotations


# FAL-POP-PROJ (§II.4 native-projection contract): honesty envelope for years the tensor covers by
# projection rather than enumeration. The projection VALUES already come from the solver's cohort-
# component dynamics (loss.py aging/birth/death terms) run on the SV-extrapolated closure past the
# last census -- the same operators that reconstruct intercensal years. What PROJ adds is the
# prime-directive discipline: classify each year by its distance from the census anchors and attach a
# per-cell uncertainty that GROWS monotonically with that horizon, typed fragile/unreliable, so a rate
# on a projected denominator carries visibly wider bars than one on a census denominator.
PROJECTION_H_SOFT = 5  # years: fragile within this horizon, unreliable (dashboard-blocked) beyond
PROJECTION_UNCERTAINTY_PER_YEAR = 0.015  # per-year sigma contribution (variance accumulates in quadrature)


def _classify_projection_years(
    periods: tuple[str, ...],
    census_years: frozenset[str],
    base_uncertainty: float,
    *,
    h_soft: int = PROJECTION_H_SOFT,
    per_year: float = PROJECTION_UNCERTAINTY_PER_YEAR,
) -> tuple[dict[str, tuple[str, int, str, float]], tuple[int | None, int | None], int]:
    """Per-year (anchor_class, horizon, state, uncertainty) + (anchored_range, max_horizon).

    anchor_class:
      - ``census``            a year with a real enumeration (2000/2010/2022); horizon 0, verified.
      - ``interpolated``      strictly between two census anchors; bounded on both sides, verified.
      - ``projected_forward`` past the latest census; horizon = years beyond, uncertainty accumulates.
      - ``projected_backward`` before the earliest census; symmetric.
      - ``unanchored``        no census anchor at all (degenerate single-window build); fragile.

    uncertainty(h) = sqrt(base^2 + h * per_year^2) -- variance accumulates one per-year step per horizon
    year (§II.4). state is ``fragile`` for 1<=h<=h_soft and ``unreliable`` beyond.
    """
    census_ints = sorted(int(y) for y in census_years)
    if not census_ints:
        return ({p: ("unanchored", 0, "fragile", base_uncertainty) for p in periods}, (None, None), 0)
    lo, hi = census_ints[0], census_ints[-1]
    census_set = set(census_ints)
    out: dict[str, tuple[str, int, str, float]] = {}
    max_h = 0
    for p in periods:
        t = int(p)
        if t in census_set:
            out[p] = ("census", 0, "verified", base_uncertainty)
        elif lo < t < hi:
            out[p] = ("interpolated", 0, "verified", base_uncertainty * 1.5)
        else:
            if t > hi:
                h, cls = t - hi, "projected_forward"
            else:
                h, cls = lo - t, "projected_backward"
            max_h = max(max_h, h)
            unc = (base_uncertainty ** 2 + h * per_year ** 2) ** 0.5
            state = "fragile" if h <= h_soft else "unreliable"
            out[p] = (cls, h, state, unc)
    return out, (lo, hi), max_h


__all__ = [
    "PROJECTION_H_SOFT",
    "PROJECTION_UNCERTAINTY_PER_YEAR",
    "_classify_projection_years",
]
