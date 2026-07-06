"""run_investigate → LDO disease-axis wiring is LIVE (refactor §4.1, MII-OUT-01/DIS-04).

Before this wiring, ``run_investigate`` called ``run_ldo`` with ``disease_graph`` /
``variable_meta`` left ``None``, so four built capabilities were inert on every real run:
the DIS-04 disease ``L_D`` prior, the §III.8 mechanical-overlap Jaccard guard, disease
provenance annotation, and the §5.1 disease variable grammar. These tests build a small
synthetic *compiled* run whose analytical variables are σ_C ICD restriction counts (exactly
the shape a real compile produces — code set in ``support.restrict_conditions``) and prove:

  1. ``variable_meta`` reaches the LDO: an edge between two code-overlapping variables is
     re-typed ``mechanical_overlap`` and carries an ``overlap_jaccard`` — and this label is
     ABSENT when ``variable_meta`` is suppressed. (overlap guard + provenance are live.)
  2. ``disease_graph`` reaches ``disease_penalty_matrix``: it is non-None, structural, and
     the related-disease penalty is strictly discounted vs. the scalar baseline — so the
     precision solve receives a genuinely different (disease-informed) penalty.

If the wiring in ``run_investigate`` is reverted (disease_graph / variable_meta back to
None), assertion 1's ``mechanical_overlap`` label disappears and the test FAILS.

Exposure (the count-with-exposure Poisson offset) is intentionally NOT exercised here — see
the module note in ``run_investigate``: a correct ``(p,S,T)`` exposure requires aligning the
population denominator onto the LDO's internally-assembled variable axis, which this
field-level fixture cannot supply without duplicating the assembler; it is reported as
deferred rather than faked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from pegasus.workflows.investigate import (
    _disease_graph_from_meta,
    disease_variable_meta,
    run_investigate,
)


# Four σ_C ICD restriction count variables. Dengue's codes are a strict subset of
# Arbovirus's (mechanical overlap); Zika shares the same block (structural, no overlap);
# the digestive count is a different chapter entirely (unrelated control).
_VARS = {
    "SIH.DengueHospitalAdmissions": {
        "codes": ["A90", "A91"],
        "base": "arbo",       # co-moves with the arbovirus aggregate
    },
    "SIH.ArbovirusHospitalAdmissions": {
        "codes": ["A90", "A91", "A92", "U06"],
        "base": "arbo",
    },
    "SIH.ZikaHospitalAdmissions": {
        "codes": ["A92", "U06"],
        "base": "arbo2",      # correlated arbovirus sibling, distinct code set
    },
    "SIH.DigestiveHospitalAdmissions": {
        "codes": ["K35", "K37"],
        "base": "digest",     # different chapter, independent signal
    },
}


def _write_synthetic_run(tmp: Path, *, seed: int = 0) -> Path:
    run = tmp / "run"
    tensors = run / "tensors"
    tensors.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    munis = [str(270000 + i) for i in range(30)]
    years = list(range(2008, 2023))

    # Shared latent intensities so the disease variables co-vary and edges actually form.
    bases: dict[str, dict[tuple[str, int], float]] = {}
    for name in ("arbo", "arbo2", "digest"):
        bases[name] = {}
    for m in munis:
        for y in years:
            local = rng.gamma(3.0, 2.0)
            bases["arbo"][(m, y)] = local + rng.gamma(1.0, 1.0)
            bases["arbo2"][(m, y)] = 0.6 * local + rng.gamma(1.0, 1.0)   # correlated sibling
            bases["digest"][(m, y)] = rng.gamma(3.0, 2.0)                # independent

    v_rows = []
    for fid, spec in _VARS.items():
        base = bases[spec["base"]]
        rows = []
        for m in munis:
            for y in years:
                lam = max(base[(m, y)] + rng.normal(0, 0.5), 0.1)
                rows.append({"municipality_cod6": m, "year": y, "value": float(rng.poisson(lam))})
        path = tensors / f"{fid.replace('.', '_')}.parquet"
        pl.DataFrame(rows).write_parquet(path)
        support = {
            "support_kind": "source_artifact_event_count_restricted",
            "restrict_conditions": [
                {"column": "principal_icd_norm", "op": "starts_with_any", "value": spec["codes"]}
            ],
        }
        v_rows.append({
            "field_id": fid,
            "name": fid,
            "path": str(path),
            "provenance": json.dumps(["official"]),
            "state": "materialized",
            "role": json.dumps(["source_event_count", "restricted_count"]),
            "support_json": json.dumps(support),
            "axes_json": json.dumps({"geography": "mun_residence_cod6", "time": "admission_year"}),
        })

    pl.DataFrame(v_rows).write_parquet(run / "V_fields.parquet")
    return run


def _edges(records):
    return {(r.source_var, r.target_var, r.edge_type): r for r in records}


def test_variable_meta_and_disease_graph_are_derived_from_the_compiled_run(tmp_path: Path) -> None:
    """The helpers the wiring uses recover code sets + a coupling graph from V_fields alone."""
    run = _write_synthetic_run(tmp_path)
    meta = disease_variable_meta(run)

    assert set(meta) == set(_VARS)
    assert meta["SIH.DengueHospitalAdmissions"]["code_set"] == frozenset({"A90", "A91"})
    assert meta["SIH.ArbovirusHospitalAdmissions"]["code_set"] == frozenset({"A90", "A91", "A92", "U06"})

    graph = _disease_graph_from_meta(meta)
    assert graph is not None and graph.legality_class == "structural"
    dense = graph.adjacency().toarray()
    codes = list(graph.codes)
    di, ai = codes.index("SIH.DengueHospitalAdmissions"), codes.index("SIH.ArbovirusHospitalAdmissions")
    gi = codes.index("SIH.DigestiveHospitalAdmissions")
    assert dense[di, ai] > 0.0                       # arbovirus siblings are coupled
    assert dense[di, gi] == 0.0                       # digestive (chapter XI) is not


def test_overlap_guard_is_live_through_run_investigate_and_absent_when_suppressed(tmp_path: Path) -> None:
    """The §III.8 mechanical-overlap guard fires via ``run_investigate`` and vanishes without it.

    Both branches call the REAL entrypoint over the same panel/data; the only difference is
    whether the disease wiring is threaded. If the wiring in ``run_investigate`` is reverted,
    the ``wired`` branch loses its mechanical-overlap edges and disease provenance and this
    test fails — i.e. it is anchored to the entrypoint, not to a hand-built LDO call.
    """
    run = _write_synthetic_run(tmp_path)

    wired = run_investigate(
        run, K=1, n_subsamples=6, run_residual_scan=False, write=False,
        stability_threshold=0.0, seed=0,
    )
    # Suppress the disease wiring by passing empty meta + None graph explicitly — same panel,
    # same data, everything else identical (this is exactly the pre-wiring behaviour).
    suppressed = run_investigate(
        run, K=1, n_subsamples=6, run_residual_scan=False, write=False,
        stability_threshold=0.0, seed=0,
        variable_meta={}, disease_graph=None,
    )

    # Through the real entrypoint: the wired run's diagnostics record the disease variables,
    # the applied L_D prior, disease provenance, and at least one mechanical-overlap re-typing.
    assert wired.diagnostics["disease_variables"] == 4
    assert wired.diagnostics["disease_graph_applied"] is True
    assert wired.diagnostics["disease_prior_applied"] is True
    assert wired.diagnostics["n_disease_provenanced"] > 0
    assert wired.diagnostics["n_mechanical_overlap"] >= 1, (
        "overlap guard did not fire through run_investigate"
    )

    # The suppressed (pre-refactor) run does NONE of it — the guard and prior are inert.
    assert suppressed.diagnostics["disease_variables"] == 0
    assert suppressed.diagnostics["disease_graph_applied"] is False
    assert suppressed.diagnostics["disease_prior_applied"] is False
    assert suppressed.diagnostics["n_mechanical_overlap"] == 0
    assert suppressed.diagnostics["n_disease_provenanced"] == 0

    # Detailed corroboration on the re-typed record itself (Dengue codes ⊂ Arbovirus codes):
    # the overlap is demoted to descriptive, carries the Jaccard + shared-code warning, and
    # is stamped with the CID-10 code system by the provenance step.
    from pegasus.ldo.orchestrator import run_ldo
    from pegasus.she.panel import compile_common_panel

    panel = compile_common_panel(run, resolution="year")
    meta = disease_variable_meta(run)
    graph = _disease_graph_from_meta(meta)
    wired_run = run_ldo(
        panel, K=1, n_subsamples=6, run_residual_scan=False, seed=0,
        stability_threshold=0.0, variable_meta=meta, disease_graph=graph,
    )
    dengue, arbo = "SIH.DengueHospitalAdmissions", "SIH.ArbovirusHospitalAdmissions"
    overlap = [
        r for r in wired_run.link_records
        if r.edge_type == "mechanical_overlap" and {r.source_var, r.target_var} == {dengue, arbo}
    ]
    assert overlap, "expected a Dengue↔Arbovirus mechanical_overlap edge"
    r = overlap[0]
    assert r.overlap_jaccard is not None and r.overlap_jaccard >= 0.5
    assert "mechanical_overlap_shared_codes" in r.warnings
    assert r.certification_status == "descriptive"   # demoted, never a discovery
    assert r.code_system == "CID-10"                 # disease provenance annotation ran


def test_disease_graph_gives_the_precision_solve_a_discounted_penalty(tmp_path: Path) -> None:
    """disease_graph reaches disease_penalty_matrix: related-disease penalty < scalar λ1."""
    run = _write_synthetic_run(tmp_path)
    meta = disease_variable_meta(run)
    graph = _disease_graph_from_meta(meta)
    assert graph is not None

    from pegasus.ldo.disease_prior import disease_penalty_matrix

    lam = 0.1
    penalty = disease_penalty_matrix(graph.codes, graph, lambda1=lam)
    codes = list(graph.codes)
    di, ai = codes.index("SIH.DengueHospitalAdmissions"), codes.index("SIH.ArbovirusHospitalAdmissions")
    gi = codes.index("SIH.DigestiveHospitalAdmissions")

    # The related arbovirus pair is discounted below the scalar baseline (the L_D prior
    # lets their sparse edge survive); the unrelated digestive pair stays at full λ1.
    assert penalty[di, ai] < lam
    assert penalty[di, gi] == lam
    # And the matrix is genuinely non-scalar — i.e. NOT what the estimator sees with no graph.
    assert not np.allclose(penalty, lam)
