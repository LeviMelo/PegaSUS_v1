"""LDO field assembly (MSD-II §II.6.1, MII-LDO-00).

Turn the CommonPanel (geo×time × fields, with per-cell provenance) into the LDO's
multivariate data tensor ``X ∈ R^{p×S×T}`` plus a per-cell reliability weight
tensor ``W`` drawn from the panel's provenance state (and, when available, the
§3.12 state tensor's ``n_eff``). ``X`` is what Layer 0 (copula margins) consumes;
``W`` down-weights broadcast/reconstructed cells relative to observed ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl

from pegasus.she.panel import CommonPanel

# Provenance state → default reliability weight.
_STATE_WEIGHT: dict[str, float] = {
    "observed": 1.0,
    "geo_invariant_broadcast": 0.5,
    "time_invariant_broadcast": 0.5,
    "domain_scalar_broadcast": 0.25,
    "reconstructed": 0.5,
    "bounded": 0.4,
    "projected": 0.5,
    "unavailable_on_panel": 0.0,
}


@dataclass
class LDOField:
    variables: tuple[str, ...]        # p field ids, axis 0
    space_ids: tuple[str, ...]        # S municipality_cod6, axis 1
    time_ids: tuple[int, ...]         # T years (or year*12+month at month grain), axis 2
    X: np.ndarray                     # (p, S, T) values (nan where absent)
    W: np.ndarray                     # (p, S, T) reliability weights in [0,1]
    resolution: str = "year"
    exposure: np.ndarray | None = None  # (p, S, T) denominator for count-with-exposure margin (§III.5)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.X.shape

    def variable_index(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.variables)}


def _time_key(cell_keys: tuple[str, ...]) -> str:
    return "year"


def assemble_ldo_tensor(
    panel: CommonPanel,
    *,
    field_weights: Mapping[str, float] | None = None,
    keep_variables: set[str] | frozenset[str] | None = None,
    exposure_field_by_variable: Mapping[str, str] | None = None,
) -> LDOField:
    """Assemble ``X`` and ``W`` from a compiled CommonPanel.

    ``field_weights`` optionally scales a whole variable's weights (e.g. a per-field
    ``n_eff``-derived reliability from the §3.12 state tensor). ``keep_variables``,
    if given, restricts the LDO variable set to those field ids (e.g. the analytical
    fields per ``field_selection.analytical_variable_ids``) — raw passthrough/support
    axes are dropped, shrinking the ``O(p^3)`` precision solve and removing meaningless
    edges.
    """
    values = panel.values
    time_col = _time_key(panel.cell_keys)
    if "municipality_cod6" not in values.columns or time_col not in values.columns:
        raise ValueError("CommonPanel values must carry municipality_cod6 and a time column")

    # LDO variables are the numeric fields; non-numeric fields (e.g. ICD-code
    # valued) are not continuous variables the precision operator can consume.
    numeric = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8}
    variables = tuple(
        c for c in values.columns
        if c not in panel.cell_keys and values.schema.get(c) in numeric
        and (keep_variables is None or c in keep_variables)
    )
    space_ids = tuple(sorted(str(s) for s in values["municipality_cod6"].unique() if s is not None))
    time_ids = tuple(sorted(int(t) for t in values[time_col].unique() if t is not None))
    p, S, T = len(variables), len(space_ids), len(time_ids)

    s_index = {s: i for i, s in enumerate(space_ids)}
    t_index = {t: i for i, t in enumerate(time_ids)}

    # Vectorized scatter: map each row's (space, time) to flat indices once, then
    # assign each variable column in one shot (was a per-row Python loop per field).
    si_col = values["municipality_cod6"].cast(pl.Utf8).replace_strict(s_index, default=-1)
    ti_col = values[time_col].cast(pl.Int64).replace_strict(t_index, default=-1)
    si_arr = si_col.to_numpy()
    ti_arr = ti_col.to_numpy()
    valid_cell = (si_arr >= 0) & (ti_arr >= 0)
    si_arr0, ti_arr0 = si_arr, ti_arr  # value-scatter cell indices (manifest block reuses the names)
    X = np.full((p, S, T), np.nan, dtype=np.float64)
    for vi, var in enumerate(variables):
        vals = values[var].cast(pl.Float64).to_numpy()
        ok = valid_cell & np.isfinite(vals)
        X[vi, si_arr[ok], ti_arr[ok]] = vals[ok]

    # Weights from the per-(field, cell) provenance manifest.
    W = np.zeros((p, S, T), dtype=np.float64)
    var_index = {v: i for i, v in enumerate(variables)}
    manifest = panel.manifest
    if manifest.height:
        # Vectorized scatter: map field/space/time/state to indices+weights in one polars
        # pass, then a single numpy fancy-index assignment. The old row-by-row
        # ``.to_dicts()`` loop was O(fields×cells) Python dict builds + per-row dict
        # lookups — at national scale the manifest is p·S·T ≈ tens of millions of rows.
        # ``default=-1``/``fill_null(-1)`` route missing field/space/time (and raw nulls,
        # which map outside the index) to -1 so ``ok`` drops them — mirroring the old
        # loop's ``is None``/``.get() is None`` skips. Null ``state`` → weight 0.0 (as
        # ``str(None)`` missed the dict before).
        m = manifest.select(["field_id", "municipality_cod6", time_col, "state"]).with_columns(
            pl.col("field_id").replace_strict(var_index, default=-1).fill_null(-1).alias("_vi"),
            pl.col("municipality_cod6").cast(pl.Utf8).replace_strict(s_index, default=-1).fill_null(-1).alias("_si"),
            pl.col(time_col).cast(pl.Int64).replace_strict(t_index, default=-1).fill_null(-1).alias("_ti"),
            pl.col("state").cast(pl.Utf8).replace_strict(_STATE_WEIGHT, default=0.0).fill_null(0.0).alias("_w"),
        )
        vi_arr = m["_vi"].to_numpy()
        si_arr = m["_si"].to_numpy()
        ti_arr = m["_ti"].to_numpy()
        w_arr = m["_w"].to_numpy()
        ok = (vi_arr >= 0) & (si_arr >= 0) & (ti_arr >= 0)
        # Last-write-wins on any duplicate (field,cell) rows, matching the old loop.
        W[vi_arr[ok], si_arr[ok], ti_arr[ok]] = w_arr[ok]
    else:
        W[~np.isnan(X)] = 1.0

    if field_weights:
        for var, scale in field_weights.items():
            vi = var_index.get(var)
            if vi is not None:
                W[vi] *= float(scale)

    # A NaN value can never carry positive weight.
    W[np.isnan(X)] = 0.0

    # §III.5 count-with-exposure: build the per-variable denominator tensor for the count
    # variables whose exposure field is declared. The exposure field is any numeric panel
    # column (typically the population denominator) — it need not itself be a kept variable.
    exposure = None
    if exposure_field_by_variable:
        exposure = np.full((p, S, T), np.nan, dtype=np.float64)
        for var, exp_field in exposure_field_by_variable.items():
            vi = var_index.get(var)
            if vi is None or exp_field not in values.columns:
                continue
            evals = values[exp_field].cast(pl.Float64).to_numpy()
            ok = valid_cell & np.isfinite(evals)
            exposure[vi, si_arr0[ok], ti_arr0[ok]] = evals[ok]
        if not np.isfinite(exposure).any():
            exposure = None
    return LDOField(variables=variables, space_ids=space_ids, time_ids=time_ids,
                    X=X, W=W, resolution=panel.resolution, exposure=exposure)


__all__ = ["LDOField", "assemble_ldo_tensor"]
