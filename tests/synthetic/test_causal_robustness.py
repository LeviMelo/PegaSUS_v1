"""Theme-15 — causal-ladder robustness guards (MSD-III §IV).

Three fragilities, each guarded default-on but result-preserving on a correct fit:
  (1) LiNGAM is identifiable only from NON-Gaussianity; on ~Gaussian inputs the direction is
      provably unrecoverable, so orientation is REFUSED (tagged ``lingam_gaussian_unidentifiable``)
      instead of emitting a coin-flip. A genuinely non-Gaussian cause→effect still orients correctly.
  (2) A near-threshold (weak/ambiguous) v-structure — where the CI calls could flip on sampling
      noise — is left undirected; a clean, well-separated collider is still oriented.
  (3) An ITS whose residuals are autocorrelated (AR(1)) violates the single-clean-break assumption;
      the iid t-stat is anticonservative, so the effect is flagged ``its_autocorrelated_effect_uncertain``
      with a Newey–West-deflated t. A clean iid break is unaffected (same estimate, no flag).
"""

from __future__ import annotations

import numpy as np

from pegasus.causal.orient import collider_stability, is_collider, orient_links
from pegasus.causal.quasi import (
    detect_structural_break,
    escalate_rung2_its,
    interrupted_time_series,
)
from pegasus.ldo.records import LinkRecord


# ---- (1) LiNGAM validity gate --------------------------------------------------------------

def test_gaussian_inputs_refuse_orientation_nongaussian_still_orients():
    rng = np.random.default_rng(0)
    n = 3000
    # Gaussian pair: direction NOT identifiable → refuse, flag the Gaussian licence failure.
    xg = rng.standard_normal(n)
    yg = 0.7 * xg + rng.standard_normal(n)
    out_g = orient_links([LinkRecord("X", "Y", "contemporaneous")], {"X": xg, "Y": yg})[0]
    assert out_g.causal_rung == 0
    assert "lingam_gaussian_unidentifiable" in out_g.warnings
    assert "oriented_non_gaussian_lingam" not in out_g.warnings

    # Genuinely non-Gaussian cause→effect: orientation is recovered toward the cause (unchanged).
    xn = rng.exponential(1.0, n) - 1.0
    yn = 0.8 * xn + (rng.exponential(1.0, n) - 1.0)
    out_n = orient_links([LinkRecord("Y", "X", "contemporaneous")], {"X": xn, "Y": yn})[0]
    assert out_n.causal_rung == 1
    assert out_n.source_var == "X" and out_n.target_var == "Y"  # cause=X, swapped back
    assert "oriented_non_gaussian_lingam" in out_n.warnings
    assert "lingam_gaussian_unidentifiable" not in out_n.warnings


# ---- (2) faithfulness/stability guard on collider orientation ------------------------------

def _collider(seed, coef, noise, n=4000):
    rng = np.random.default_rng(seed)
    a = rng.exponential(1.0, n) - 1.0
    b = rng.exponential(1.0, n) - 1.0
    c = coef * a + coef * b + noise * (rng.exponential(1.0, n) - 1.0)
    return {"A": a, "B": b, "C": c}


def test_weak_collider_left_undirected_strong_collider_oriented():
    recs = [LinkRecord("A", "C", "contemporaneous"), LinkRecord("B", "C", "contemporaneous")]

    # WEAK: passes the raw CI test but the conditional signal barely clears the threshold
    # (stability margin not met) → left undirected with the instability flag.
    weak = _collider(11, coef=0.5, noise=1.0)
    assert is_collider(weak["A"], weak["B"], weak["C"])              # raw CI would fire …
    assert collider_stability(weak["A"], weak["B"], weak["C"]) <= 0.02   # … but it is not stable
    out_w = orient_links([LinkRecord("A", "C", "contemporaneous"),
                          LinkRecord("B", "C", "contemporaneous")], weak)
    assert all("oriented_collider" not in r.warnings for r in out_w)
    assert all("collider_v_structure" not in r.causal_assumptions for r in out_w)
    assert any("collider_unstable_left_undirected" in r.warnings for r in out_w)

    # STRONG, well-separated collider: still oriented into C (default behavior preserved).
    strong = _collider(3, coef=1.0, noise=0.15)
    out_s = orient_links(recs, strong)
    assert all(r.target_var == "C" for r in out_s)
    assert all(r.causal_rung == 1 for r in out_s)
    assert all("oriented_collider" in r.warnings for r in out_s)


# ---- (3) ITS robustness to residual autocorrelation ----------------------------------------

def _ar1_break_series(T=80, t0=40, rho=0.85, jump=6.0, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T) * 0.5
    x = np.zeros(T)
    for i in range(1, T):
        x[i] = rho * x[i - 1] + e[i]          # strongly autocorrelated disturbance
    return 1.0 + 0.02 * np.arange(T) + jump * (np.arange(T) >= t0) + x


def test_its_autocorrelation_flagged_clean_break_unchanged():
    # AR(1) residuals around a real break: DW well below 2, autocorrelation flagged, HAC t < OLS t.
    y = _ar1_break_series()
    brk = detect_structural_break(y)
    its = interrupted_time_series(y, brk)
    assert its.autocorrelated
    assert its.durbin_watson < 1.4
    assert abs(its.level_t_hac) < abs(its.level_t)   # serial correlation deflates the effect t

    edge = LinkRecord("S", "Y", "lagged_directed", lag_k=1, causal_rung=1,
                      causal_assumptions=("time_precedence",))
    out = escalate_rung2_its([edge], {"Y": y})[0]
    assert out.causal_rung == 2
    assert "its_autocorrelated_effect_uncertain" in out.warnings
    assert any(w.startswith("rung2_its_level_t_hac_") for w in out.warnings)

    # DEFAULT-PRESERVING: a clean iid break is NOT flagged and the OLS estimate is unchanged.
    rng = np.random.default_rng(0)
    Tc, t0 = 60, 30
    yc = 2.0 + 0.1 * np.arange(Tc) + 5.0 * (np.arange(Tc) >= t0) + rng.normal(0, 0.5, Tc)
    itsc = interrupted_time_series(yc, t0)
    assert not itsc.autocorrelated
    assert abs(itsc.level_change - 5.0) < 1.0 and itsc.level_t > 3.0
    clean = escalate_rung2_its(
        [LinkRecord("S", "Y", "lagged_directed", lag_k=1, causal_rung=1)], {"Y": yc})[0]
    assert clean.causal_rung == 2
    assert "its_autocorrelated_effect_uncertain" not in clean.warnings
