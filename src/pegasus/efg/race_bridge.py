from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import numpy as np

from pegasus.core.hashing import content_hash, sha256_file

ADMIN_RACE_LABELS = {
    "1": "branca",
    "2": "preta",
    "3": "amarela",
    "4": "parda",
    "5": "indigena",
}

DEFAULT_TARGET_CATEGORIES = ["branca", "preta", "amarela", "parda", "indigena"]
# A DATASUS race code counts as a valid administrative observation for the bridge when its
# normalizer state marks it valid. The normalizers (datasus/normalize/primitives.py) emit the
# generic marker "valid" for an in-range 1-5 race code; earlier bridge code checked only for
# "valid_admin_race", so EVERY real record was silently treated as missing and reallocated
# uniformly via local-pi (the "uniform race" bug). Accept both markers.
VALID_ADMIN_RACE_STATES: frozenset[str] = frozenset({"valid_admin_race", "valid"})
RACE_BRIDGE_BOOTSTRAP_REPLICATES = 200
RACE_BRIDGE_MATRIX_CONCENTRATION = 250.0
RACE_BRIDGE_PI_CONCENTRATION = 250.0


class RaceBridgeValidationError(ValueError):
    """Raised when a Bridge_R emission-prior object is invalid."""


@dataclass(frozen=True)
class RaceBridgePrior:
    bridge_id: str
    mode: str
    source_axis: str
    target_axis: str
    source_categories: list[str]
    target_categories: list[str]
    matrix: dict[str, dict[str, float]]
    sensitivity_width: float
    metadata: dict[str, Any]
    prior_hash: str


@dataclass(frozen=True)
class RaceBridgeCounts:
    raw_admin_counts: dict[str, int]
    missing_count: int
    total_count: int
    support: dict[str, Any]

    @property
    def missing_share(self) -> float:
        if self.total_count <= 0:
            return 0.0
        return self.missing_count / float(self.total_count)


@dataclass(frozen=True)
class RaceBridgePosterior:
    posterior_counts: dict[str, float]
    lower_counts: dict[str, float]
    upper_counts: dict[str, float]
    raw_admin_counts: dict[str, int]
    missing_count: int
    missing_share: float
    sensitivity_width: float
    race_bridge_cv: float
    prior: RaceBridgePrior
    support: dict[str, Any]
    bridge_mode: str = "posterior_simulation"
    effective_bridge_mode: str = "localPi_posteriorC"

    def metadata(self) -> dict[str, Any]:
        return {
            "numerator_axis_source": self.prior.source_axis,
            "denominator_axis_target": self.prior.target_axis,
            "bridge_operator": f"Bridge_R_{self.effective_bridge_mode}",
            "emission_matrix_registry_version": self.prior.bridge_id,
            # Report the mode actually exercised: localPi_posteriorC only when a
            # local self-declared composition was supplied; otherwise the fast
            # fixedC_dynamic_weight path (uniform local prior ⇒ W = C). MSD §4.5.3/§4.6.
            "bridge_mode": self.effective_bridge_mode,
            "bridge_uncertainty_mode": self.bridge_mode,
            "missing_race_share": self.missing_share,
            "race_bridge_cv": self.race_bridge_cv,
            "sensitivity_width": self.sensitivity_width,
            "race_axis_warning": "Administrative race/color is declaration-process data and is not overwritten by IBGE self-declared race.",
            "bayesian_ecological_bridge_warning": "Posterior race counts are bridge-derived observer fields, not direct self-declared measurements.",
            "prior_hash": self.prior.prior_hash,
        }


def _canonical_code(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or None


def load_race_bridge_prior(path: str | Path) -> RaceBridgePrior:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_race_bridge_prior(payload, prior_hash=sha256_file(path))


def validate_race_bridge_prior(payload: dict[str, Any], *, prior_hash: str | None = None) -> RaceBridgePrior:
    if not isinstance(payload, dict):
        raise RaceBridgeValidationError("Bridge_R prior payload must be a JSON object.")
    bridge_id = str(payload.get("bridge_id") or "")
    mode = str(payload.get("mode") or "")
    source_axis = str(payload.get("source_axis") or "")
    target_axis = str(payload.get("target_axis") or "")
    source_categories = [str(x) for x in payload.get("source_categories") or []]
    target_categories = [str(x) for x in payload.get("target_categories") or []]
    matrix = payload.get("matrix") or {}
    sensitivity_width = float(payload.get("sensitivity_width", 0.0))
    metadata = dict(payload.get("metadata") or {})

    if not bridge_id:
        raise RaceBridgeValidationError("Bridge_R prior missing bridge_id.")
    if mode not in {"localPi_posteriorC", "fixedC_dynamic_weight"}:
        raise RaceBridgeValidationError(f"Unsupported Bridge_R mode: {mode!r}")
    if not source_axis or not target_axis:
        raise RaceBridgeValidationError("Bridge_R prior must declare source_axis and target_axis.")
    if not source_categories or not target_categories:
        raise RaceBridgeValidationError("Bridge_R prior must declare nonempty source and target categories.")
    if not isinstance(matrix, dict) or not matrix:
        raise RaceBridgeValidationError("Bridge_R prior matrix must be a nonempty object.")
    if sensitivity_width < 0:
        raise RaceBridgeValidationError("Bridge_R sensitivity_width must be nonnegative.")

    normalized: dict[str, dict[str, float]] = {}
    missing_rows = [src for src in source_categories if src not in matrix]
    if missing_rows:
        raise RaceBridgeValidationError(f"Bridge_R matrix missing source rows: {missing_rows}")

    for src in source_categories:
        row = matrix.get(src)
        if not isinstance(row, dict) or not row:
            raise RaceBridgeValidationError(f"Bridge_R matrix row is empty or invalid: {src}")
        clean_row: dict[str, float] = {}
        for tgt in target_categories:
            value = row.get(tgt, 0.0)
            try:
                weight = float(value)
            except Exception as exc:
                raise RaceBridgeValidationError(f"Bridge_R matrix value is nonnumeric at {src}->{tgt}: {value!r}") from exc
            if not math.isfinite(weight) or weight < 0:
                raise RaceBridgeValidationError(f"Bridge_R matrix value is negative/nonfinite at {src}->{tgt}: {weight}")
            clean_row[tgt] = weight
        extra_targets = sorted(set(row) - set(target_categories))
        if extra_targets:
            raise RaceBridgeValidationError(f"Bridge_R matrix row has undeclared target categories at {src}: {extra_targets}")
        total = sum(clean_row.values())
        if abs(total - 1.0) > 1e-6:
            raise RaceBridgeValidationError(f"Bridge_R matrix row must sum to 1.0: source={src} sum={total}")
        normalized[src] = clean_row

    return RaceBridgePrior(
        bridge_id=bridge_id,
        mode=mode,
        source_axis=source_axis,
        target_axis=target_axis,
        source_categories=source_categories,
        target_categories=target_categories,
        matrix=normalized,
        sensitivity_width=sensitivity_width,
        metadata=metadata,
        prior_hash=prior_hash or content_hash(payload),
    )


def summarize_sim_admin_race_counts(
    sim_events_path: str | Path,
    *,
    municipality_cod6: str | None = None,
    year: int | None = None,
) -> RaceBridgeCounts:
    df = pl.read_parquet(sim_events_path)
    if municipality_cod6 is not None and "mun_residence_cod6" in df.columns:
        df = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
    if year is not None and "year" in df.columns:
        df = df.filter(pl.col("year") == int(year))

    support = {
        "time": {"years": sorted(int(x) for x in df["year"].drop_nulls().unique().to_list()) if "year" in df.columns else []},
        "geography": {
            "municipality_cod6": sorted(str(x) for x in df["mun_residence_cod6"].drop_nulls().unique().to_list())
            if "mun_residence_cod6" in df.columns else []
        },
        "n_events": int(df.height),
    }

    raw_counts = {code: 0 for code in ADMIN_RACE_LABELS}
    missing = 0
    if "race_color_admin" not in df.columns:
        return RaceBridgeCounts(raw_admin_counts=raw_counts, missing_count=int(df.height), total_count=int(df.height), support=support)

    states = df["race_missingness_state"].to_list() if "race_missingness_state" in df.columns else [None] * df.height
    for code_value, state in zip(df["race_color_admin"].to_list(), states, strict=False):
        code = _canonical_code(code_value)
        if code in raw_counts and (state is None or state in VALID_ADMIN_RACE_STATES):
            raw_counts[code] += 1
        else:
            missing += 1
    return RaceBridgeCounts(raw_admin_counts=raw_counts, missing_count=missing, total_count=int(df.height), support=support)


def bridge_admin_race_group_counts(
    *,
    race_codes: list[Any],
    prior: RaceBridgePrior,
    race_states: list[Any] | None = None,
    support: dict[str, Any] | None = None,
) -> RaceBridgePosterior:
    """Bridge one group's raw administrative race/color codes to self-declared counts.

    Shared entry point for any consumer holding a flat list of administrative
    race codes for one (locality, time, ...) cell -- e.g. SIM deaths or SINASC
    newborn births feeding the population tensor (MSD §2.8.5/§2.8.6), or an EFG
    ``Bridge_R`` field. A code counts as valid administrative race only when its
    paired state (if given) is ``valid_admin_race``; everything else -- missing,
    unknown-sentinel, or an unrecognized code -- is folded into ``missing_count``
    and reallocated via the prior's local-pi, never dropped silently.
    """
    raw_counts = {code: 0 for code in ADMIN_RACE_LABELS}
    missing = 0
    states = race_states if race_states is not None else [None] * len(race_codes)
    for code_value, state in zip(race_codes, states, strict=False):
        code = _canonical_code(code_value)
        if code in raw_counts and (state is None or state in VALID_ADMIN_RACE_STATES):
            raw_counts[code] += 1
        else:
            missing += 1
    counts = RaceBridgeCounts(
        raw_admin_counts=raw_counts,
        missing_count=missing,
        total_count=len(race_codes),
        support=support or {},
    )
    return fixedc_dynamic_weight_bridge(counts, prior)


def fixedc_dynamic_weight_bridge(counts: RaceBridgeCounts, prior: RaceBridgePrior) -> RaceBridgePosterior:
    for source in counts.raw_admin_counts:
        if source not in prior.source_categories:
            raise RaceBridgeValidationError(f"Raw administrative race category not supported by prior: {source}")

    local_pi = _local_target_pi(counts=counts, prior=prior)
    crosswalk = _local_pi_crosswalk(prior=prior, local_pi=local_pi)
    posterior = _bridge_with_crosswalk(counts.raw_admin_counts, crosswalk, prior)
    if counts.missing_count:
        for target in prior.target_categories:
            posterior[target] += float(counts.missing_count) * local_pi[target]

    draws = _posterior_bridge_draws(counts=counts, prior=prior, local_pi=local_pi)
    lower, upper = _posterior_intervals(draws=draws, posterior=posterior, prior=prior)
    cv = _posterior_draw_cv(draws=draws, prior=prior)
    width = _sensitivity_width(posterior=posterior, lower=lower, upper=upper, prior=prior, missing_share=counts.missing_share)

    # Effective mode: localPi only when a local self-declared composition was
    # actually supplied; otherwise the uniform local prior reduces W to C exactly
    # (the fast fixedC_dynamic_weight path). MSD §4.5.3/§4.6.
    has_local_pi = _local_pi_source(counts.support) == "declared_target_population_shares"
    effective_mode = "localPi_posteriorC" if has_local_pi else "fixedC_dynamic_weight"

    return RaceBridgePosterior(
        posterior_counts=posterior,
        lower_counts=lower,
        upper_counts=upper,
        raw_admin_counts=dict(counts.raw_admin_counts),
        missing_count=counts.missing_count,
        missing_share=counts.missing_share,
        sensitivity_width=width,
        race_bridge_cv=cv,
        prior=prior,
        support=dict(counts.support) | {
            "bridge_operator": "Bridge_R_localPi_posteriorC",
            "bridge_uncertainty_mode": "posterior_simulation",
            "local_target_pi": local_pi,
            "local_pi_source": _local_pi_source(counts.support),
            "posterior_replicates": RACE_BRIDGE_BOOTSTRAP_REPLICATES,
        },
        bridge_mode="posterior_simulation",
        effective_bridge_mode=effective_mode,
    )


def _local_pi_source(support: dict[str, Any]) -> str:
    shares = support.get("target_population_shares") if isinstance(support, dict) else None
    return "declared_target_population_shares" if isinstance(shares, dict) and shares else "uniform_missing_local_calibration"


def _local_target_pi(*, counts: RaceBridgeCounts, prior: RaceBridgePrior) -> dict[str, float]:
    shares = counts.support.get("target_population_shares") if isinstance(counts.support, dict) else None
    if isinstance(shares, dict) and shares:
        clean = {target: max(float(shares.get(target, 0.0) or 0.0), 0.0) for target in prior.target_categories}
    else:
        clean = {target: 1.0 for target in prior.target_categories}
    total = sum(clean.values())
    if total <= 0:
        clean = {target: 1.0 for target in prior.target_categories}
        total = float(len(prior.target_categories))
    return {target: clean[target] / total for target in prior.target_categories}


def _local_pi_crosswalk(*, prior: RaceBridgePrior, local_pi: dict[str, float]) -> dict[str, dict[str, float]]:
    """Return W[source][target] ∝ C[source,target] * local pi[target]."""
    output: dict[str, dict[str, float]] = {}
    for source in prior.source_categories:
        weights = {
            target: max(float(prior.matrix[source].get(target, 0.0)), 0.0) * max(float(local_pi.get(target, 0.0)), 0.0)
            for target in prior.target_categories
        }
        total = sum(weights.values())
        if total <= 0:
            output[source] = dict(local_pi)
        else:
            output[source] = {target: weights[target] / total for target in prior.target_categories}
    return output


def _bridge_with_crosswalk(
    raw_counts: dict[str, int],
    crosswalk: dict[str, dict[str, float]],
    prior: RaceBridgePrior,
) -> dict[str, float]:
    posterior = {target: 0.0 for target in prior.target_categories}
    for source, n in raw_counts.items():
        row = crosswalk[source]
        for target, weight in row.items():
            posterior[target] += float(n) * weight
    return posterior


def _bridge_once(raw_counts: dict[str, int], prior: RaceBridgePrior) -> dict[str, float]:
    return _bridge_with_crosswalk(raw_counts, prior.matrix, prior)


def _bootstrap_bridge_cv(*, counts: RaceBridgeCounts, prior: RaceBridgePrior) -> float:
    observed_total = sum(max(int(v), 0) for v in counts.raw_admin_counts.values())
    if observed_total <= 0:
        return 0.0
    source_categories = list(counts.raw_admin_counts)
    probs = np.array([counts.raw_admin_counts[source] / observed_total for source in source_categories], dtype=float)
    rng = np.random.default_rng(20260627)
    totals_by_target = {target: [] for target in prior.target_categories}
    for _ in range(RACE_BRIDGE_BOOTSTRAP_REPLICATES):
        draw = rng.multinomial(observed_total, probs)
        raw = {source: int(draw[idx]) for idx, source in enumerate(source_categories)}
        posterior = _bridge_once(raw, prior)
        for target, value in posterior.items():
            totals_by_target[target].append(float(value))
    cvs: list[float] = []
    for values in totals_by_target.values():
        if not values:
            continue
        mean = float(np.mean(values))
        if mean <= 0:
            continue
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        cvs.append(sd / mean)
    return max(cvs) if cvs else 0.0


def _dirichlet(alpha: list[float], rng: np.random.Generator) -> np.ndarray:
    safe = np.array([max(float(x), 1e-6) for x in alpha], dtype=float)
    return rng.dirichlet(safe)


def _draw_prior_matrix(prior: RaceBridgePrior, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    concentration = float(prior.metadata.get("matrix_concentration", RACE_BRIDGE_MATRIX_CONCENTRATION))
    matrix: dict[str, dict[str, float]] = {}
    for source in prior.source_categories:
        base = [max(float(prior.matrix[source].get(target, 0.0)), 0.0) for target in prior.target_categories]
        draw = _dirichlet([concentration * value + 1e-3 for value in base], rng)
        matrix[source] = {target: float(draw[idx]) for idx, target in enumerate(prior.target_categories)}
    return matrix


def _draw_local_pi(local_pi: dict[str, float], prior: RaceBridgePrior, rng: np.random.Generator) -> dict[str, float]:
    concentration = float(prior.metadata.get("local_pi_concentration", RACE_BRIDGE_PI_CONCENTRATION))
    draw = _dirichlet([concentration * local_pi[target] + 1e-3 for target in prior.target_categories], rng)
    return {target: float(draw[idx]) for idx, target in enumerate(prior.target_categories)}


def _posterior_bridge_draws(
    *,
    counts: RaceBridgeCounts,
    prior: RaceBridgePrior,
    local_pi: dict[str, float],
) -> dict[str, list[float]]:
    observed_total = sum(max(int(value), 0) for value in counts.raw_admin_counts.values())
    rng = np.random.default_rng(20260628)
    source_categories = list(prior.source_categories)
    source_probs = np.array(
        [max(int(counts.raw_admin_counts.get(source, 0)), 0) / observed_total for source in source_categories],
        dtype=float,
    ) if observed_total > 0 else np.array([1.0 / len(source_categories)] * len(source_categories), dtype=float)
    draws = {target: [] for target in prior.target_categories}
    for _ in range(RACE_BRIDGE_BOOTSTRAP_REPLICATES):
        matrix_draw = _draw_prior_matrix(prior, rng)
        pi_draw = _draw_local_pi(local_pi, prior, rng)
        crosswalk = _local_pi_crosswalk(
            prior=RaceBridgePrior(
                bridge_id=prior.bridge_id,
                mode=prior.mode,
                source_axis=prior.source_axis,
                target_axis=prior.target_axis,
                source_categories=prior.source_categories,
                target_categories=prior.target_categories,
                matrix=matrix_draw,
                sensitivity_width=prior.sensitivity_width,
                metadata=prior.metadata,
                prior_hash=prior.prior_hash,
            ),
            local_pi=pi_draw,
        )
        if observed_total > 0:
            observed_draw = rng.multinomial(observed_total, source_probs)
            raw_draw = {source: int(observed_draw[idx]) for idx, source in enumerate(source_categories)}
        else:
            raw_draw = {source: 0 for source in source_categories}
        posterior = _bridge_with_crosswalk(raw_draw, crosswalk, prior)
        if counts.missing_count:
            missing_draw = rng.multinomial(int(counts.missing_count), np.array([pi_draw[target] for target in prior.target_categories]))
            for idx, target in enumerate(prior.target_categories):
                posterior[target] += float(missing_draw[idx])
        for target, value in posterior.items():
            draws[target].append(float(value))
    return draws


def _posterior_intervals(
    *,
    draws: dict[str, list[float]],
    posterior: dict[str, float],
    prior: RaceBridgePrior,
) -> tuple[dict[str, float], dict[str, float]]:
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    width = max(float(prior.sensitivity_width), 0.0)
    for target in prior.target_categories:
        values = draws.get(target) or [posterior[target]]
        lo = float(np.quantile(values, 0.025))
        hi = float(np.quantile(values, 0.975))
        # Conservative partial-identification widening over the declared credible set.
        lower[target] = max(0.0, min(lo, posterior[target] * (1.0 - width)))
        upper[target] = max(hi, posterior[target] * (1.0 + width))
    return lower, upper


def _posterior_draw_cv(*, draws: dict[str, list[float]], prior: RaceBridgePrior) -> float:
    cvs: list[float] = []
    for target in prior.target_categories:
        values = draws.get(target) or []
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        if mean <= 0:
            continue
        sd = float(np.std(values, ddof=1))
        cvs.append(sd / mean)
    return max(cvs) if cvs else 0.0


def _sensitivity_width(
    *,
    posterior: dict[str, float],
    lower: dict[str, float],
    upper: dict[str, float],
    prior: RaceBridgePrior,
    missing_share: float,
) -> float:
    widths: list[float] = [max(float(prior.sensitivity_width), float(missing_share))]
    for target in prior.target_categories:
        value = posterior.get(target, 0.0)
        if value <= 0:
            continue
        widths.append(max(value - lower.get(target, value), upper.get(target, value) - value) / value)
    return float(min(1.0, max(widths)))
