"""O4 / §III.8 — the two certification conjuncts the original audit missed.

(3) Regularization-path agreement: an edge present at only one λ₁ operating point is a threshold
artefact and must not certify. (4) Latent-vs-lag separability (§IX.2): a directed lag whose
endpoints also share a contemporaneous latent factor may be a phase-offset shared-wave artefact.
Both are folded into the gate as warnings certify_link downgrades on.
"""

from __future__ import annotations

from pegasus.ldo.certgates import (
    apply_certification_gates,
    latent_vs_lag_confounds,
)
from pegasus.ldo.certify import certify_link
from pegasus.ldo.lags import LaggedLink
from pegasus.ldo.records import LinkRecord


class _FakeLagged:
    def __init__(self, lagged_links, latent_shared):
        self.lagged_links = lagged_links
        self.latent_shared = latent_shared


def test_latent_vs_lag_confound_flags_shared_factor_directed_edge():
    lagged = _FakeLagged(
        lagged_links=[
            LaggedLink(source="A", target="B", peak_lag=2, peak_partial_correlation=0.3,
                       response_curve=[0.0, 0.1, 0.3]),
            LaggedLink(source="C", target="D", peak_lag=1, peak_partial_correlation=0.4,
                       response_curve=[0.0, 0.4]),
        ],
        latent_shared=[("A", "B", 0.6)],   # A,B ALSO ride a shared contemporaneous factor
    )
    flags = latent_vs_lag_confounds(lagged)
    assert ("A", "B", 2) in flags          # the A→B lag is a possible phase-offset artefact
    assert ("C", "D", 1) not in flags      # C→D has no shared factor → genuine lagged edge


def test_gates_downgrade_a_single_lambda_edge_and_a_confounded_lag():
    records = [
        LinkRecord(source_var="A", target_var="B", edge_type="lagged_directed", lag_k=2,
                   weight=0.3, stability=0.9, uncertainty=0.05, certification_status="selected"),
        LinkRecord(source_var="E", target_var="F", edge_type="contemporaneous", lag_k=0,
                   weight=0.4, stability=0.9, uncertainty=0.05, certification_status="selected"),
    ]
    # E~F appears on only 1/3 of the λ grid → path disagreement; A→B is latent-lag confounded.
    path_agreement = {(frozenset(("E", "F")), 0): 0.33, (frozenset(("A", "B")), 2): 1.0}
    gated = apply_certification_gates(
        records, path_agreement=path_agreement, latent_flags={("A", "B", 2)})
    by = {r.source_var: r for r in gated}
    assert any(w.startswith("low_regularization_path_agreement") for w in by["E"].warnings)
    assert "possible_latent_lag_confound" in by["A"].warnings
    # certify_link turns both warnings into a descriptive (non-certified) verdict
    assert certify_link(by["A"]).certification_status == "descriptive"
    assert certify_link(by["E"]).certification_status == "descriptive"


def test_clean_edge_still_certifies():
    r = LinkRecord(source_var="A", target_var="B", edge_type="contemporaneous", lag_k=0,
                   weight=0.4, stability=0.9, uncertainty=0.05, certification_status="selected")
    gated = apply_certification_gates([r], path_agreement={(frozenset(("A", "B")), 0): 1.0},
                                      latent_flags=set())
    assert certify_link(gated[0]).certification_status == "selected"
