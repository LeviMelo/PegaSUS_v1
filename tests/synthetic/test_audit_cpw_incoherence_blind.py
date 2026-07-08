"""ADVERSARIAL AUDIT (W5) — the CPW incoherence measure is blind to the exact condition
it claims to detect.

lowrank._cpw_incoherence documents itself as a CPW identifiability score whose LOW value
means "S's mass sits inside L's span" (the unidentifiable/degenerate split). But the score
is a product of two INDEPENDENT MARGINALS:

    score = spread(L only) * (1 - deg(S only))

`spread` is the participation number of L's column-space projector diagonal — a function of
L ALONE. `deg` is the off-diagonal edge density of S — a function of S ALONE. Neither term,
nor their product, ever computes any overlap between S's support and L's column space. So the
measure CANNOT distinguish:

  (identifiable)  a spread L + a sparse S whose edges lie OUTSIDE L's span, from
  (degenerate)    the SAME spread L + the SAME-sparsity S whose edges lie INSIDE L's span,

even though the second is precisely the "S mass inside L span" case the docstring says a low
score flags. This test constructs that adversarial pair and asserts the guard should tell them
apart. It will FAIL while the measure remains a marginal product — confirming the flaw.

It also drives the full public fit entry point (`fit_sparse_plus_lowrank`) on a maximally-spread
global factor that co-moves every variable together with a strong direct edge sitting on two of
those same (in-span) variables, and asserts the guard does NOT wave that degenerate split through
as `well_identified`.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.lowrank import _cpw_incoherence, fit_sparse_plus_lowrank


def _make_pair(p: int = 12):
    """Two (S, V) splits with IDENTICAL marginal spread(L) and identical S sparsity, differing
    ONLY in whether S's single strong edge lands inside or outside L's column span."""
    # Spread low-rank factor on coordinates {0..5}.
    V = np.zeros((p, 1))
    V[0:6, 0] = 1.0 / np.sqrt(6)

    # Identifiable: S's strong edge on {10, 11} — DISJOINT from L's span {0..5}.
    S_identifiable = np.eye(p) * 2.0
    S_identifiable[10, 11] = S_identifiable[11, 10] = -1.2

    # Degenerate: S's strong edge on {0, 1} — INSIDE L's span (docstring's unidentifiable case).
    S_degenerate = np.eye(p) * 2.0
    S_degenerate[0, 1] = S_degenerate[1, 0] = -1.2

    return (S_identifiable, V), (S_degenerate, V)


def test_incoherence_distinguishes_s_inside_vs_outside_l_span():
    """The score must separate an identifiable split from a degenerate one whose sparse mass
    sits inside L's span. If it cannot, it is not measuring CPW identifiability at all."""
    (S_id, V_id), (S_dg, V_dg) = _make_pair()

    score_id, spread_id, deg_id = _cpw_incoherence(S_id, V_id, edge_threshold=0.05)
    score_dg, spread_dg, deg_dg = _cpw_incoherence(S_dg, V_dg, edge_threshold=0.05)

    # The two splits share spread(L) and deg(S) exactly — proving the score's two inputs are
    # blind to the S-in-L-span overlap that actually governs CPW identifiability.
    assert np.isclose(spread_id, spread_dg)
    assert np.isclose(deg_id, deg_dg)

    # A genuine identifiability measure must rank the S-inside-L-span split as STRICTLY LESS
    # identifiable than the S-disjoint split. This is the flaw: the measure returns them equal.
    assert score_dg < score_id - 1e-6, (
        "CPW incoherence gives the degenerate (S-inside-L-span) split the SAME score as the "
        f"identifiable one (id={score_id:.4f}, degenerate={score_dg:.4f}) — the measure is a "
        "product of two independent marginals and cannot detect S-mass inside L's span."
    )


def test_fit_does_not_pass_a_maximally_confounded_split_as_well_identified():
    """End-to-end: a single global factor that co-moves EVERY variable (spread L ~ 1) plus a
    strong direct edge on two of those same in-span variables is the maximally-confounded S/L
    split. The guard should not certify it as well_identified."""
    rng = np.random.default_rng(7)
    p, n = 14, 8000
    f = rng.standard_normal(n)
    X = rng.standard_normal((n, p)) * 0.4
    for v in range(p):            # a global driver on EVERY variable → maximally spread L
        X[:, v] += 0.85 * f
    X[:, 1] += 1.5 * X[:, 0]       # a strong direct edge on {0,1}, inside the global span
    C = np.corrcoef(X, rowvar=False)

    fit = fit_sparse_plus_lowrank(C, lambda1=0.08, lambda2=0.05)

    # The direct edge {0,1} lives inside L's (global) span → S mass inside L span → the
    # docstring's degenerate case. A guard that "flags a poorly-identified S/L split" must not
    # report this as well_identified.
    assert fit.incoherence is not None
    assert fit.well_identified is False, (
        f"maximally-confounded global-factor split passed the CPW guard "
        f"(incoherence={fit.incoherence:.3f}, well_identified=True) — the guard rubber-stamps a "
        "degenerate split because a dense/spread L drives 'spread' to 1 regardless of S overlap."
    )
