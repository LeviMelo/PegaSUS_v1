"""ADVERSARIAL AUDIT (W3) — matrix-free Moran + AR(1) temporal whitening.

These tests are hostile refutations, not friendly confirmations. Each is written to
FAIL if the flaw it targets is real, so a green run means the flaw does NOT exist and
a red run pins it. They must not be weakened to make the suite green.

Targets (commit aa10a8b):
  A. pegasus.ldo.temporal._ar1_phi / temporal_whiten — the AR(1) coefficient estimator.
  B. pegasus.geo.spatial.moran_i — "== the dense row-standardized Moran" claim.
"""

from __future__ import annotations

import numpy as np

from pegasus.geo.spatial import moran_i
from pegasus.geo.spatial_graph import SpatialWeightGraph
from pegasus.ldo.temporal import _ar1_phi, temporal_whiten


# ---------------------------------------------------------------------------
# A. AR(1) phi estimation must be a *temporal* AR(1) coefficient. The whitening's
#    stated job is "remove AR(1) so a shared trend does not read as a lagged edge".
#    On a panel that is temporally WHITE (iid over time within each unit) but has
#    per-unit fixed effects (different baseline levels — the norm in municipality
#    panels), the true temporal AR(1) is 0. A correct estimator returns ~0 and the
#    whitening is a near no-op. The pooled-mean estimator instead centers by the
#    single global mean over all (space,time) pairs, so the between-unit variance
#    masquerades as temporal persistence and phi_hat blows up toward 1.
# ---------------------------------------------------------------------------

def test_ar1_phi_not_inflated_by_spatial_fixed_effects():
    rng = np.random.default_rng(0)
    S, T = 200, 20
    levels = rng.standard_normal(S) * 5.0          # strong per-unit fixed effects
    noise = rng.standard_normal((S, T))            # iid over time -> zero temporal AR(1)
    Z_var = levels[:, None] + noise                # (S, T)

    phi_hat = _ar1_phi(Z_var)

    # A valid temporal AR(1) estimate on temporally-white data is ~0. Allow generous
    # slack (0.3) for finite-sample noise. The pooled-mean estimator returns ~0.96 here.
    assert abs(phi_hat) < 0.3, (
        f"_ar1_phi conflates spatial heterogeneity with temporal persistence: "
        f"temporally-white data with per-unit fixed effects yields phi_hat={phi_hat:.3f} "
        f"(should be ~0). The whitener will over-difference genuine panels."
    )


def test_temporal_whiten_is_near_noop_on_temporally_white_data():
    """A field with NO temporal autocorrelation (only spatial FE) should be left
    essentially unchanged by temporal whitening — the whitening should not manufacture
    a large phi and difference the series. Compares the whitened t>=1 residuals to the
    raw t>=1 values; a near-no-op keeps them highly correlated."""
    rng = np.random.default_rng(7)
    S, T = 200, 20
    levels = rng.standard_normal(S) * 5.0
    noise = rng.standard_normal((S, T))
    Z = (levels[:, None] + noise)[None]            # (1, S, T), temporally white

    Zw, phi = temporal_whiten(Z)

    # If phi ~ 0 the whitened series ~ the raw series (shifted/scaled trivially) and
    # the transform is near-identity in correlation terms. With the pooled-mean bias
    # phi ~ 0.95, so z_t - phi z_{t-1} differences out the level and the whitened
    # residual decorrelates from the raw value.
    raw = Z[0, :, 1:].ravel()
    wht = Zw[0, :, 1:].ravel()
    m = np.isfinite(raw) & np.isfinite(wht)
    a = raw[m] - raw[m].mean()
    b = wht[m] - wht[m].mean()
    corr = float((a @ b) / np.sqrt((a @ a) * (b @ b)))
    assert corr > 0.7, (
        f"temporal_whiten materially distorted temporally-white data "
        f"(raw<->whitened corr={corr:.3f}, phi={float(phi[0]):.3f}); the AR(1) estimator "
        f"fired on between-unit variance, not temporal structure."
    )


# ---------------------------------------------------------------------------
# B. moran_i "== the dense version". True on FULL support (verified separately).
#    Under PARTIAL observation the claim is only true for the specific dense
#    variant the code implements (full-graph-degree row weights restricted to
#    observed-observed edges). It does NOT equal the statistically-standard Moran
#    computed over the observed subgraph (neighbours re-counted among observed
#    units), which is what a statistician / PySAL returns when you subset to the
#    observed support. This test pins the divergence: if the two ever coincide the
#    assertion (that they differ) fails and we learn the implementation changed.
# ---------------------------------------------------------------------------

def _rook_grid(side: int) -> SpatialWeightGraph:
    nodes = [f"{r}_{c}" for r in range(side) for c in range(side)]
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for r in range(side):
        for c in range(side):
            h = f"{r}_{c}"
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    adj[h].append(f"{rr}_{cc}")
    return SpatialWeightGraph(
        graph_id="grid", legality_class="structural", provenance=(),
        node_ids=tuple(nodes), _adjacency={k: tuple(v) for k, v in adj.items()},
    )


def _proper_subgraph_moran(obs: dict[str, float], graph: SpatialWeightGraph) -> float:
    """Reference: row-standardized Moran over the OBSERVED subgraph (neighbours
    re-standardized among observed units only)."""
    nodes = [n for n in graph.node_ids if n in obs]
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    for a in nodes:
        for b in graph.neighbors(a):
            if b in idx:
                W[idx[a], idx[b]] = 1.0
    deg = W.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    Wr = W / deg
    x = np.array([obs[n] for n in nodes])
    z = x - x.mean()
    return float((n / Wr.sum()) * ((z @ Wr @ z) / (z @ z)))


def test_moran_matches_dense_on_full_support():
    """The core claim, on FULL support: matrix-free == dense row-standardized Moran."""
    g = _rook_grid(4)
    vals = {f"{r}_{c}": float(r * 2 - c) for r in range(4) for c in range(4)}
    I_mf = moran_i(vals, g)

    nodes = list(g.node_ids)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    for a in nodes:
        for b in g.neighbors(a):
            W[idx[a], idx[b]] = 1.0
    Wr = W / W.sum(axis=1, keepdims=True)
    x = np.array([vals[n] for n in nodes])
    z = x - x.mean()
    I_dense = float((n / Wr.sum()) * ((z @ Wr @ z) / (z @ z)))
    assert abs(I_mf - I_dense) < 1e-9, f"matrix-free != dense on full support: {I_mf} vs {I_dense}"


def test_moran_diverges_from_proper_observed_subgraph_on_partial_support():
    """ADVERSARIAL: under partial observation the matrix-free Moran does NOT equal the
    proper observed-subgraph row-standardized Moran. Fails if they coincide — pinning
    that 'moran_i == the dense version' is support-conditional, not unconditional."""
    g = _rook_grid(4)
    full = {f"{r}_{c}": float(r + c) for r in range(4) for c in range(4)}
    missing = {"0_1", "1_1", "2_2", "3_0", "1_3"}     # scattered holes so degree-restriction bites
    obs = {k: v for k, v in full.items() if k not in missing}

    I_mf = moran_i(obs, g)
    I_proper = _proper_subgraph_moran(obs, g)

    assert I_mf is not None
    # If these ever agree, the implementation switched to the proper observed-subgraph
    # standardization; until then they must differ, so the "==dense" claim needs the
    # full-support caveat.
    assert abs(I_mf - I_proper) > 1e-3, (
        f"expected matrix-free Moran ({I_mf:.6f}) to differ from the proper observed-"
        f"subgraph Moran ({I_proper:.6f}) under partial support; if equal, the "
        f"full-degree-vs-observed-degree standardization was reconciled."
    )
