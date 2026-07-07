from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pegasus.core.enums import FieldState
from pegasus.core.schemas import FieldNode, QState, WarningRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def provenance_risk(provenance: list[str]) -> float:
    p = set(provenance)
    if "synthetic" in p:
        return 1.0
    if "latent" in p:
        return 0.8
    if "SIM_informed_denominator_prior" in p or "facility_linkage_filtered" in p:
        return 0.5
    if "reconstructed" in p or "geneallocated" in p:
        return 0.4
    if "classification_projected" in p or "longitudinally_stitched" in p:
        return 0.3
    if "harmonized" in p or "deflated" in p or "AMC_contracted" in p:
        return 0.2
    return 0.0


def classify_q_state(
    *,
    n_eff: float | None,
    denom_fragility: float | None,
    missingness: float | None,
    risk: float,
) -> FieldState:
    n_eff = 0.0 if n_eff is None else n_eff
    denom_fragility = 1.0 if denom_fragility is None else denom_fragility
    missingness = 1.0 if missingness is None else missingness

    if n_eff >= 100 and denom_fragility < 0.05 and missingness < 0.10 and risk < 0.5:
        return FieldState.verified
    if n_eff >= 30 and denom_fragility < 0.20:
        return FieldState.fragile
    return FieldState.quarantined_descriptive


def _numeric_values(tensor: Any) -> list[float]:
    if tensor is None:
        return []
    if isinstance(tensor, (list, tuple)):
        values = tensor
    elif hasattr(tensor, "to_series"):
        try:
            values = tensor.to_series().to_list()
        except Exception:
            values = []
    elif hasattr(tensor, "to_list"):
        try:
            values = tensor.to_list()
        except Exception:
            values = []
    else:
        values = []
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean == 0:
        return None
    # Sample variance (ddof=1), matching the original Σ(x-x̄)²/(n-1).
    var = float(arr.var(ddof=1))
    return float(math.sqrt(var) / abs(mean))


def _temporal_roughness(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=np.float64)
    diffs = np.diff(arr)
    denom = abs(float(arr.mean())) or 1.0
    return float(math.sqrt(float(np.mean(diffs * diffs))) / denom)


def _spatial_entropy(values: list[float]) -> float | None:
    if not values:
        return None
    nonnegative = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(nonnegative.sum())
    if total <= 0:
        return 0.0
    positive = nonnegative[nonnegative > 0]
    probs = positive / total
    entropy = float(-np.sum(probs * np.log(probs)))
    n = nonnegative.shape[0]
    max_entropy = math.log(n) if n > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _moran_contiguity(values: list[float]) -> float | None:
    """Moran's I under a 1-D rook-contiguity weight over the tensor cell order.

    A best-effort spatial-autocorrelation estimate used when no explicit
    municipality adjacency matrix is supplied to the Q-tensor (MSD §3.12): cells
    adjacent in the provided ordering are treated as neighbours (w_ij = 1 for
    |i - j| = 1, symmetric). This is the standard Moran's I statistic

        I = (N / W) * (Σ_i Σ_j w_ij (x_i - x̄)(x_j - x̄)) / Σ_i (x_i - x̄)²

    specialized to chain contiguity. It is an ordering proxy, not a
    geography-aware Moran's I; callers should record the proxy provenance.
    Returns None when undefined (fewer than 3 cells or zero variance).
    """
    n = len(values)
    if n < 3:
        return None
    arr = np.asarray(values, dtype=np.float64)
    deviations = arr - float(arr.mean())
    denom = float(deviations @ deviations)
    if denom <= 0:
        return None
    # Symmetric chain contiguity: each adjacent pair (i,i+1) contributes twice
    # (both (i,i+1) and (i+1,i)); W = 2*(n-1).
    cross = 2.0 * float(deviations[:-1] @ deviations[1:])
    weight_total = 2.0 * (n - 1)
    if weight_total <= 0:
        return None
    return float((n / weight_total) * (cross / denom))


def _kish_effective_n(weights: list[float]) -> float | None:
    """Kish effective sample size ``(Σw)² / Σw²`` (MSD-I §3.12.3).

    Down-weights the raw cell count when the per-cell weights are concentrated
    in a few cells. Under equal weights it equals the cell count. ``w`` is the
    numerator count for count variables and the denominator for rate/proportion
    variables (§3.12.3). Returns ``None`` when no positive weight exists.
    """
    if not weights:
        return None
    abs_w = np.abs(np.asarray(weights, dtype=np.float64))
    s1 = float(abs_w.sum())
    s2 = float(abs_w @ abs_w)
    if s2 <= 0:
        return None
    return (s1 * s1) / s2


def _moran_corrected_n_eff(base_n: float | None, moran_i: float | None) -> float | None:
    """Deflate an effective size by spatial autocorrelation (MSD-I §3.12.3).

    ``n_eff = base_n · 1/(1 + max(0, MoranI))`` — positive autocorrelation reduces
    the count of *independent* observations; non-positive autocorrelation leaves
    it unchanged.
    """
    if base_n is None:
        return None
    deflation = 1.0 / (1.0 + max(0.0, moran_i or 0.0))
    return float(max(1.0, base_n * deflation))


def compute_q_state(
    *,
    field: FieldNode,
    tensor=None,
    denominator=None,
    provenance: list[str] | None = None,
    warnings: list[WarningRecord] | None = None,
    registries=None,
    spatial_graph=None,
    spatial_values=None,
) -> QState:
    provenance = provenance or field.provenance
    risk = provenance_risk(provenance)

    n_events = field.support.get("n_events")
    n_denom = field.support.get("n_denom")
    n_eff = field.support.get("n_eff", n_events)
    missingness = min(1.0, field.support.get("missingness", 0.0) + field.support.get("invalid_flag_share", 0.0))
    denom_fragility = field.support.get("denom_fragility", 1.0 if n_denom is None else 0.0)
    zero_inflation = field.support.get("zero_inflation", 0.0)
    values = _numeric_values(tensor)
    denom_values = _numeric_values(denominator)
    cv = field.support.get("cv")
    moran_i = field.support.get("moran_i")
    temporal_roughness = field.support.get("temporal_roughness")
    spatial_entropy = field.support.get("spatial_entropy")
    q_warnings = list(field.warnings or [])
    geo_n_eff = None  # geography-aware spatial effective-n (single source of truth)
    if values or denom_values:
        if values:
            cv = cv if cv is not None else _cv(values)
            temporal_roughness = temporal_roughness if temporal_roughness is not None else _temporal_roughness(values)
            spatial_entropy = spatial_entropy if spatial_entropy is not None else _spatial_entropy(values)
            if moran_i is None:
                geo_vals = spatial_values if spatial_values is not None else values
                if spatial_graph is not None:
                    from pegasus.geo.spatial import effective_n as _geo_neff
                    from pegasus.geo.spatial import moran_i as _geo_moran
                    moran_i = _geo_moran(geo_vals, spatial_graph)
                    geo_n_eff = _geo_neff(geo_vals, spatial_graph)
                else:
                    moran_i = _moran_contiguity(values)
                    if moran_i is not None:
                        q_warnings.append("moran_i_ordering_contiguity_proxy")
        # §3.12.3 effective sample size: Kish (Σw)²/Σw² over per-cell weights, deflated by
        # spatial autocorrelation. With a real graph the deflation is the geography-aware
        # effective_n/n ratio; else the 1/(1+max(0,MoranI)) proxy correction.
        weights = denom_values if denom_values else values
        kish = _kish_effective_n(weights)
        if geo_n_eff is not None and kish is not None:
            n_obs = max(1, sum(1 for v in (spatial_values.values() if spatial_values is not None else values)))
            n_eff = float(max(1.0, kish * (geo_n_eff / n_obs)))
        else:
            n_eff = _moran_corrected_n_eff(kish, moran_i)
    elif n_eff is not None:
        n_eff = _moran_corrected_n_eff(float(n_eff), moran_i)

    state = classify_q_state(
        n_eff=n_eff,
        denom_fragility=denom_fragility,
        missingness=missingness,
        risk=risk,
    )

    race_axis = field.axes.get("race_axis_type") or field.axes.get("race_axis")

    return QState(
        field_id=field.id,
        n_events=n_events,
        n_denom=n_denom,
        n_eff=n_eff,
        cov_S=field.support.get("cov_S"),
        cov_T=field.support.get("cov_T"),
        missingness=missingness,
        zero_inflation=zero_inflation,
        denom_fragility=denom_fragility,
        cv=cv,
        moran_i=moran_i,
        temporal_roughness=temporal_roughness,
        spatial_entropy=spatial_entropy,
        provenance_risk=risk,
        race_axis_source=race_axis,
        race_axis_target=None,
        missing_race_share=field.support.get("missing_race_share"),
        emission_prior_strength=None,
        race_bridge_cv=None,
        sensitivity_width=None,
        bridge_mode=None,
        state=state,
        dashboard_safe=False if state != FieldState.verified else field.dashboard_safe,
        warnings=q_warnings,
        computed_at=_now(),
        q_schema_version="1.0",
    )
