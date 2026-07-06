"""Synthetic ground truth — planted-structure generation + recovery scoring (VAL-03, §IX.2).

The only source of *exact* ground truth: generate an LDO field from a KNOWN dependency
structure (planted directed-lagged edges, shared latent factors, spatial structure) and
measure how well the engine recovers it (precision/recall of the edge set, lag error,
factor attribution). This is the required test for the hardest claims — separability
adequacy and certification power — that cannot be settled by argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pegasus.ldo.assemble import LDOField


@dataclass(frozen=True)
class PlantedEdge:
    source: str
    target: str
    lag: int
    coef: float


@dataclass
class PlantedTruth:
    variables: tuple[str, ...]
    edges: tuple[PlantedEdge, ...] = ()
    factors: tuple[tuple[tuple[str, ...], float], ...] = ()   # ((members...), loading)

    def undirected_pairs(self) -> set[frozenset[str]]:
        return {frozenset((e.source, e.target)) for e in self.edges}


def generate_planted_field(
    truth: PlantedTruth, *, S: int = 24, T: int = 48, noise: float = 0.4, seed: int = 0,
) -> LDOField:
    """Generate an ``LDOField`` realizing ``truth``.

    Base noise per variable; shared factors add a common wave to their members (→ the
    low-rank layer should attribute these as ``latent_shared``); each planted edge adds
    ``coef · source(t-lag)`` to the target. Edges are applied in listed order (author a DAG).
    """
    rng = np.random.default_rng(seed)
    variables = truth.variables
    idx = {v: i for i, v in enumerate(variables)}
    p = len(variables)
    X = noise * rng.standard_normal((p, S, T))

    for members, loading in truth.factors:
        wave = rng.standard_normal((S, T))
        for v in members:
            X[idx[v]] += loading * wave

    for e in truth.edges:
        si, ti = idx[e.source], idx[e.target]
        if e.lag <= 0:
            X[ti] += e.coef * X[si]
        else:
            X[ti, :, e.lag:] += e.coef * X[si, :, : T - e.lag]

    return LDOField(
        variables=variables,
        space_ids=tuple(str(270000 + i) for i in range(S)),
        time_ids=tuple(range(T)),
        X=X,
        W=np.ones((p, S, T), dtype=np.float64),
        resolution="month",
    )


@dataclass
class RecoveryScore:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    lag_mae: float | None                 # mean |recovered_lag − planted_lag| over matched directed edges
    factor_attribution: float             # fraction of planted factor pairs flagged latent_shared
    details: dict = field(default_factory=dict)


def recovery_score(
    discovered, truth: PlantedTruth, *, min_weight: float = 0.05, selected_only: bool = True,
) -> RecoveryScore:
    """Score edge-set recovery of ``discovered`` LinkRecords against ``truth``.

    An edge counts as "found" when it is certified ``selected`` (or, if ``selected_only``
    is False, exceeds ``min_weight``) and is not a ``mechanical_overlap`` artefact.
    Precision/recall are over undirected variable pairs; lag error is over matched planted
    directed edges; factor attribution is the share of planted co-factor pairs the engine
    flagged ``latent_shared``.
    """
    def _keep(r) -> bool:
        # A latent_shared record is a *confounding* claim, not a direct-edge claim — the
        # engine correctly declaring E,F,G share a driver must not be scored as three
        # false-positive direct edges. It is scored separately as factor_attribution below.
        if r.edge_type in ("mechanical_overlap", "latent_shared"):
            return False
        if selected_only and r.certification_status != "selected":
            return False
        return abs(r.weight) >= min_weight

    kept = [r for r in discovered if _keep(r)]
    found_pairs = {frozenset((r.source_var, r.target_var)) for r in kept}
    planted_pairs = truth.undirected_pairs()

    tp = len(planted_pairs & found_pairs)
    fp = len(found_pairs - planted_pairs)
    fn = len(planted_pairs - found_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # lag error over matched directed planted edges
    lagged = {(r.source_var, r.target_var): r.lag_k for r in kept if r.edge_type == "lagged_directed"}
    lag_errs = [
        abs(lagged[(e.source, e.target)] - e.lag)
        for e in truth.edges if e.lag > 0 and (e.source, e.target) in lagged
    ]
    lag_mae = float(np.mean(lag_errs)) if lag_errs else None

    # factor attribution: planted co-factor pairs the engine flagged latent_shared
    shared = {frozenset((r.source_var, r.target_var)) for r in discovered if r.edge_type == "latent_shared"}
    planted_factor_pairs: set[frozenset[str]] = set()
    for members, _ in truth.factors:
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                planted_factor_pairs.add(frozenset((members[a], members[b])))
    factor_attr = (
        len(planted_factor_pairs & shared) / len(planted_factor_pairs)
        if planted_factor_pairs else 1.0
    )

    return RecoveryScore(
        precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn,
        lag_mae=lag_mae, factor_attribution=factor_attr,
        details={"found": sorted(tuple(sorted(pr)) for pr in found_pairs)},
    )


__all__ = ["PlantedEdge", "PlantedTruth", "generate_planted_field", "RecoveryScore", "recovery_score"]
