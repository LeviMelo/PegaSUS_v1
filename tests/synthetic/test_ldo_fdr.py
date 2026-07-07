"""Theme-5 — dependence-aware FDR over LDO edges.

BH on Fisher-z p-values at the dependence-corrected effective-n: strong edges (high |pcorr|,
high n_eff) get small q, null-ish edges (tiny pcorr) get large q. Annotate-only is the default
(certification_status untouched); fdr_enforce downgrades a selected null edge to descriptive.
"""

from __future__ import annotations

from pegasus.ldo.certify import LDOCertificationPolicy, certify_links
from pegasus.ldo.multiplicity import benjamini_hochberg, fisher_z_pvalue
from pegasus.ldo.records import LinkRecord


def _edge(name, pcorr, n_eff):
    # pre-gated as selected (stability + uncertainty present) so only FDR can move it
    return LinkRecord(
        source_var=name, target_var=name + "'", edge_type="contemporaneous",
        weight=pcorr, partial_correlation=pcorr, stability=0.9, uncertainty=0.05,
        n_eff=n_eff, certification_status="selected",
    )


def test_bh_separates_strong_from_null_and_enforce_downgrades():
    strong = [_edge("S%d" % i, 0.45, 900.0) for i in range(3)]
    nulls = [_edge("N%d" % i, 0.01, 900.0) for i in range(6)]
    records = strong + nulls

    # annotate-only default: q computed, certification_status untouched
    annotated = certify_links(records)
    by = {r.source_var: r for r in annotated}
    assert all(r.certification_status == "selected" for r in annotated)  # no-op on status
    assert all(r.fdr_method == "benjamini_hochberg_fisherz_neff" for r in annotated)
    assert max(by["S%d" % i].fdr_qvalue for i in range(3)) < 0.05   # strong → small q
    assert min(by["N%d" % i].fdr_qvalue for i in range(6)) > 0.5    # null → large q

    # enforce: null edges that were selected fall to descriptive; strong stay selected
    enforced = certify_links(records, policy=LDOCertificationPolicy(fdr_enforce=True))
    by = {r.source_var: r for r in enforced}
    assert all(by["S%d" % i].certification_status == "selected" for i in range(3))
    assert all(by["N%d" % i].certification_status == "descriptive" for i in range(6))
    assert "fdr_not_significant_descriptive_only" in by["N0"].warnings


def test_untestable_edges_skipped_and_default_is_byte_identical():
    # no n_eff / no pcorr → skipped, q stays None, record unchanged under the default
    r = LinkRecord(source_var="A", target_var="B", edge_type="contemporaneous",
                   weight=0.3, partial_correlation=0.3, stability=0.9, uncertainty=0.05,
                   certification_status="selected")  # n_eff is None → untestable
    out = certify_links([r])
    assert out[0] == r                      # default leaves an untestable edge byte-identical
    assert out[0].fdr_qvalue is None

    # BH primitive sanity: monotone, ordered
    rej, q = benjamini_hochberg([0.001, 0.5, 0.9], 0.1)
    assert rej[0] and not rej[1] and not rej[2]
    assert q[0] < q[1] <= q[2]
    assert fisher_z_pvalue(0.45, 900.0) < fisher_z_pvalue(0.01, 900.0)
