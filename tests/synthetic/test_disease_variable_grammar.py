"""Disease variable grammar (§5.1) — carriers × disease concepts become LDO variables.

Proves the concept grammar instantiates the LDO variable set from raw ICD-bearing
events: the disease axis *generates* what gets modeled, end-to-end into `run_ldo`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pegasus.disease.variable_grammar import assemble_disease_field, stratify_events
from pegasus.pirs.ldo.orchestrator import run_ldo


def test_stratify_by_chapter_generates_typed_variables() -> None:
    events = pl.DataFrame({
        "underlying_icd_norm": ["I219", "I22", "A90", "J18", "R99"],
        "municipality_cod6": ["270430", "270430", "270430", "270630", "270630"],
        "time": [1, 1, 2, 1, 2],
    })
    strat = stratify_events(
        events, carrier="Deaths", topology_role="underlying_cause",
        resolution="chapter", code_col="underlying_icd_norm",
    )
    by_id = {v.concept_id: v for v in strat.variables}

    # cardiac I21/I22 → chapter IX; WHO-absent dengue A90 → chapter I (not dropped/coerced)
    assert set(by_id["chapter_IX"].member_codes) == {"I219", "I22"}
    assert by_id["chapter_I"].member_codes == ("A90",)
    # topology_role is part of the variable identity (§5.1: underlying_cause ≠ mention)
    assert by_id["chapter_IX"].variable_id == "Deaths|underlying_cause|chapter_IX"
    # the cardiac cell count is 2 (I21+I22) at muni 270430, month 1
    ix = strat.counts.filter(
        (pl.col("variable_id") == "Deaths|underlying_cause|chapter_IX")
        & (pl.col("municipality_cod6") == "270430") & (pl.col("time") == 1)
    )
    assert ix["count"][0] == 2


def test_multilabel_preserved_at_concept_resolution() -> None:
    events = pl.DataFrame({
        "underlying_icd_norm": ["I219"], "municipality_cod6": ["270430"], "time": [1],
    })
    strat = stratify_events(
        events, carrier="Deaths", topology_role="underlying_cause",
        resolution="concept:body_system", code_col="underlying_icd_norm",
    )
    concepts = {v.concept_id for v in strat.variables}
    # one code feeds BOTH its chapter and its block variable — multi-label, never forced to one
    assert "chapter_IX" in concepts and "block_I20-I25" in concepts


def test_disease_variables_feed_the_ldo() -> None:
    """Generated disease-stratified variables drive run_ldo and recover their co-movement."""
    rng = np.random.default_rng(0)
    munis = [str(270000 + i) for i in range(15)]
    rows: list[dict] = []
    for t in range(24):
        for m in munis:
            base = rng.poisson(8)                     # shared local intensity → cardiac & resp co-vary
            n_cardiac = base + rng.poisson(2)
            n_resp = base + rng.poisson(2)
            n_other = rng.poisson(5)                  # an independent third chapter
            rows += [{"underlying_icd_norm": "I219", "municipality_cod6": m, "time": t}] * int(n_cardiac)
            rows += [{"underlying_icd_norm": "J18", "municipality_cod6": m, "time": t}] * int(n_resp)
            rows += [{"underlying_icd_norm": "K35", "municipality_cod6": m, "time": t}] * int(n_other)
    events = pl.DataFrame(rows)

    strat = stratify_events(
        events, carrier="Deaths", topology_role="underlying_cause",
        resolution="chapter", code_col="underlying_icd_norm",
    )
    field = assemble_disease_field(strat)
    assert "Deaths|underlying_cause|chapter_IX" in field.variables    # cardiac
    assert "Deaths|underlying_cause|chapter_X" in field.variables     # respiratory

    run = run_ldo(field, K=1, n_subsamples=6, run_residual_scan=False, seed=0)
    # the planted co-movement between the two disease variables surfaces as an edge
    linked = {
        frozenset((r.source_var, r.target_var))
        for r in run.link_records if abs(r.weight) > 0.05
    }
    assert frozenset(("Deaths|underlying_cause|chapter_IX", "Deaths|underlying_cause|chapter_X")) in linked
