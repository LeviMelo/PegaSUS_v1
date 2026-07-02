"""Disease Semantic Axis foundation (MSD-II §II.13) — DIS-01/02/03."""

import numpy as np
import pytest

from pegasus.disease import concept_registry as R
from pegasus.disease import icd_adapter as A
from pegasus.disease.graph import DiseaseGraph, DiseaseGraphCircularityError


# --- DIS-02: ICD/CID adapter -------------------------------------------------

def test_adapter_dot_and_hierarchy():
    assert A.add_dot("I219") == "I21.9"
    assert A.remove_dot("I21.9") == "I219"
    assert A.is_valid_who("I21.9") is True
    assert "I21" in A.ancestors("I21.9")
    assert A.nearest_common_ancestor("I21.0", "I22") == "I20-I25"


def test_adapter_code_info_who():
    info = A.code_info("I21.9")
    assert info.status == "who_icd10"
    assert info.chapter == "IX" and info.block == "I20-I25" and info.category == "I21"
    assert info.is_leaf is True


def test_adapter_cid10_only_code_not_coerced():
    # Dengue A90 is authoritative CID-10 but absent from simple-icd-10's WHO tree.
    info = A.code_info("A90")
    assert info.status == "source_system_specific"      # NOT coerced to a WHO subcode
    assert info.chapter == "I"                          # still resolved via CID-10 range table
    assert A.cid10_chapter("A90") == "I"


# --- DIS-01: Disease Concept Registry ----------------------------------------

def test_registry_multilabel_and_provenance():
    concepts = {a.concept_id: a for a in R.assertions_for_code("E11.9")}  # diabetes
    # Structural WHO concepts are exact; icd-mappings groupers are approximate on CID-10.
    assert concepts["chapter_IV"].projection_status == "exact"
    assert concepts["chapter_IV"].source == "who_icd10"
    ccsr = next(a for a in concepts.values() if a.concept_id.startswith("ccsr_"))
    assert ccsr.projection_status == "approximate"
    assert ccsr.multi_label is True
    assert "icd10cm_grouper_on_cid10" in ccsr.warnings
    # Brazilian ICSAP list is exact, and diabetes is chronic.
    assert concepts["brazilian_icsap_diabetes"].projection_status == "exact"
    assert concepts["brazilian_icsap_diabetes"].chronic == "chronic"
    # A code holds MULTIPLE concepts (multi-label preserved, never forced to one).
    assert len(concepts) >= 4


def test_registry_chronicity_from_cci():
    concepts = {a.concept_id for a in R.assertions_for_code("E11.9")}
    assert "chronicity_chronic" in concepts


def test_registry_who_absent_code_still_usable():
    # Dengue must not be dropped: it keeps its CID-10 chapter and any grouper matches.
    concepts = {a.concept_id: a for a in R.assertions_for_code("A90")}
    assert concepts["chapter_I"].projection_status == "exact"
    assert concepts["chapter_I"].source == "cid10_catalog"
    assert any(cid.startswith("source_specific_") for cid in concepts)


def test_registry_malformed_typed_not_silent():
    concepts = R.assertions_for_code("!!!")
    assert len(concepts) == 1 and concepts[0].projection_status == "unmappable"


# --- DIS-03: DiseaseGraph ----------------------------------------------------

def test_disease_graph_hierarchy_laplacian_is_sparse_symmetric():
    codes = ["I21.0", "I21.9", "I22", "E11.9", "J45"]  # cardio cluster + diabetes + asthma
    g = DiseaseGraph.hierarchy(codes)
    assert g.legality_class == "structural"
    L = g.laplacian()
    dense = L.toarray()
    assert np.allclose(dense, dense.T)                 # symmetric
    assert np.allclose(dense.sum(axis=1), 0.0)         # Laplacian rows sum to zero
    # The two I21 subcodes (same category) are strongly coupled; I21<->E11 (diff chapter) not.
    i210, i219 = g.codes.index("I21.0"), g.codes.index("I21.9")
    e11 = g.codes.index("E11.9")
    assert g.adjacency()[i210, i219] == pytest.approx(1.0)
    assert g.adjacency()[i210, e11] == 0.0


def test_disease_graph_distance_and_groups():
    g = DiseaseGraph.hierarchy(["I21.0", "I21.9", "E11.9"])
    assert g.distance("I21.0", "I21.9") == 0.0         # same category
    assert g.distance("I21.0", "E11.9") is None or g.distance("I21.0", "E11.9") > 0
    groups = g.groups()
    assert any("diabetes" in cid for cid in groups)     # ICSAP diabetes group present


def test_disease_graph_context_derived_refused_as_prior():
    import scipy.sparse as sp
    ctx = DiseaseGraph(codes=("A90", "A91"), weights=sp.csr_matrix((2, 2)),
                       legality_class="context_derived", provenance=("datasus_events",))
    with pytest.raises(DiseaseGraphCircularityError):
        ctx.as_prior()
