from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import content_hash, sha256_file

ADMIN_RACE_LABELS = {
    "1": "branca",
    "2": "preta",
    "3": "amarela",
    "4": "parda",
    "5": "indigena",
}

DEFAULT_TARGET_CATEGORIES = ["branca", "preta", "amarela", "parda", "indigena"]


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

    def metadata(self) -> dict[str, Any]:
        return {
            "numerator_axis_source": self.prior.source_axis,
            "denominator_axis_target": self.prior.target_axis,
            "bridge_operator": "Bridge_R_fixedC_dynamic_weight",
            "emission_matrix_registry_version": self.prior.bridge_id,
            "bridge_mode": self.prior.mode,
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
    if mode != "fixedC_dynamic_weight":
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
        if code in raw_counts and state == "valid_admin_race":
            raw_counts[code] += 1
        else:
            missing += 1
    return RaceBridgeCounts(raw_admin_counts=raw_counts, missing_count=missing, total_count=int(df.height), support=support)


def fixedc_dynamic_weight_bridge(counts: RaceBridgeCounts, prior: RaceBridgePrior) -> RaceBridgePosterior:
    for source in counts.raw_admin_counts:
        if source not in prior.source_categories:
            raise RaceBridgeValidationError(f"Raw administrative race category not supported by prior: {source}")

    posterior = {target: 0.0 for target in prior.target_categories}
    for source, n in counts.raw_admin_counts.items():
        row = prior.matrix[source]
        for target, weight in row.items():
            posterior[target] += float(n) * weight

    # Width is deliberately conservative: explicit prior width plus unallocated missing-race share.
    width = min(1.0, max(prior.sensitivity_width, counts.missing_share))
    lower = {target: max(0.0, value * (1.0 - width)) for target, value in posterior.items()}
    upper = {target: value * (1.0 + width) for target, value in posterior.items()}
    mean = sum(posterior.values()) / len(posterior) if posterior else 0.0
    variance = sum((value - mean) ** 2 for value in posterior.values()) / len(posterior) if posterior else 0.0
    cv = math.sqrt(variance) / mean if mean > 0 else 0.0

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
        support=counts.support,
    )
