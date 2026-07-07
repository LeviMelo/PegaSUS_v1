"""WP2 / §III.3 — the spatial BYM varying-coefficient field for an LDO edge.

An edge X→Y whose strength varies by region must be recovered as a smooth per-locality slope
field β_s (BYM/ICAR prior shrinking neighbours together), and the §III.3 multiresolution
readout must attribute the variation to the REGION scale — not smear it into muni noise. The
proof plants a region-graded slope and checks recovery + the variance-share decomposition.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pegasus.ldo.margins import GaussianField
from pegasus.ldo.spatial_field import (
    fit_spatial_varying_coefficient,
    write_spatial_field,
)


def _planted_field(seed: int = 0):
    rng = np.random.default_rng(seed)
    # Two regions (cod6 first digit 1 vs 2), two states each, several munis per state.
    space_ids = []
    region_slope = {}
    for region in ("1", "2"):
        for st in range(2):
            uf = f"{region}{st}"
            for m in range(6):
                sid = f"{uf}{m:04d}"          # 6-char cod6: region=[0], state=[:2]
                space_ids.append(sid)
                # planted slope: region 1 → strong (+1.2), region 2 → weak (+0.2)
                region_slope[sid] = 1.2 if region == "1" else 0.2
    space_ids = tuple(space_ids)
    S = len(space_ids)
    T = 40
    X = rng.standard_normal((S, T))
    beta_true = np.array([region_slope[s] for s in space_ids])
    Y = beta_true[:, None] * X + 0.15 * rng.standard_normal((S, T))
    Z = np.stack([X, Y])                       # (p=2, S, T)
    field = GaussianField(variables=("X", "Y"), space_ids=space_ids,
                          time_ids=tuple(range(T)), Z=Z, W=np.ones((2, S, T)),
                          resolution="year")
    # A block Laplacian: connect munis within the same state (exercises BYM smoothing, since the
    # synthetic cod6 are not in the real adjacency).
    state = np.array([s[:2] for s in space_ids])
    A = np.zeros((S, S))
    for a in range(S):
        for b in range(a + 1, S):
            if state[a] == state[b]:
                A[a, b] = A[b, a] = 1.0
    L = sp.csr_matrix(np.diag(A.sum(1)) - A)
    return field, beta_true, L


def test_recovers_region_graded_slope_field():
    field, beta_true, L = _planted_field()
    fit = fit_spatial_varying_coefficient(field, "X", "Y", tau=0.5, laplacian=L)
    assert fit is not None
    # β_s recovers the planted region-graded slope (smoothed, so allow tolerance).
    assert np.corrcoef(fit.beta, beta_true)[0, 1] > 0.95
    # regions are separated: region-1 munis clearly higher than region-2.
    region = np.array([s[0] for s in field.space_ids])
    assert fit.beta[region == "1"].mean() - fit.beta[region == "2"].mean() > 0.7


def test_multiresolution_attributes_variation_to_region_scale():
    field, beta_true, L = _planted_field()
    fit = fit_spatial_varying_coefficient(field, "X", "Y", tau=0.5, laplacian=L)
    comp = fit.components
    # components sum exactly to β
    recon = comp["national"] + comp["region"] + comp["state"] + comp["muni"]
    assert np.allclose(recon, fit.beta, atol=1e-9)
    # the planted variation is BETWEEN regions → region scale carries the largest variance share
    h = fit.heterogeneity
    assert h["region_var_share"] > h["state_var_share"]
    assert h["region_var_share"] > h["muni_var_share"]
    assert h["beta_std"] > 0.3            # real spatial heterogeneity present


def test_returns_none_when_no_within_locality_variation():
    # X constant over time in every locality → no within-locality slope is identifiable.
    S, T = 12, 20
    space_ids = tuple(f"1{i:05d}" for i in range(S))
    X = np.ones((S, T))
    Y = np.random.default_rng(0).standard_normal((S, T))
    field = GaussianField(variables=("X", "Y"), space_ids=space_ids,
                          time_ids=tuple(range(T)), Z=np.stack([X, Y]),
                          W=np.ones((2, S, T)), resolution="year")
    assert fit_spatial_varying_coefficient(field, "X", "Y") is None


def test_write_spatial_field_sidecar(tmp_path):
    import polars as pl

    field, beta_true, L = _planted_field()
    fit = fit_spatial_varying_coefficient(field, "X", "Y", tau=0.5, laplacian=L)
    p = write_spatial_field(fit, tmp_path / "edge_X_Y.spatial_field.parquet")
    df = pl.read_parquet(p)
    assert df.height == len(field.space_ids)
    assert {"space_id", "beta", "comp_region", "comp_state", "comp_muni"} <= set(df.columns)
