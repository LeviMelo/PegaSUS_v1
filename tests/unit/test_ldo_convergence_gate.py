"""LDO §V.6 convergence gate — an unconverged low-rank ADMM solve must not certify.

A stability-passing edge is normally promoted to ``selected``; but if the low-rank
ADMM did not converge, the S/L split it was read from is unreliable, so the edge is
downgraded to ``descriptive`` (surfaced, never presented as exact). The orchestrator
stamps ``lowrank_unconverged_descriptive_only`` on every record when the fit did not
converge; this pins the gate that consumes it.
"""

from __future__ import annotations

from pegasus.ldo.certify import certify_link
from pegasus.ldo.records import LinkRecord


def _edge(**kw) -> LinkRecord:
    base = dict(source_var="A", target_var="B", edge_type="contemporaneous",
                weight=0.5, stability=0.9)
    base.update(kw)
    return LinkRecord(**base)


def test_converged_stable_edge_is_selected():
    assert certify_link(_edge()).certification_status == "selected"


def test_unconverged_edge_is_downgraded_to_descriptive():
    rec = _edge(warnings=("lowrank_unconverged_descriptive_only",))
    assert certify_link(rec).certification_status == "descriptive", \
        "an unconverged-ADMM edge must not be certified selected (§V.6)"


def test_unconverged_gate_overrides_high_stability():
    # even with stability well above threshold, non-convergence wins.
    rec = _edge(stability=0.99, warnings=("lowrank_unconverged_descriptive_only",))
    assert certify_link(rec).certification_status == "descriptive"
