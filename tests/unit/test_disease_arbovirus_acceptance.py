"""Arbovirus/microcephaly ICD acceptance gate (MSD-III §XI.4; refactor §4.3 + ICD review).

The program's definition-of-done for the Brazilian arbovirus/microcephaly epidemiology it
targets did not exist as a regression (`grep zika|microcephaly tests/` was empty). This pins:

  1. The WHO-2019 catalogue GAP is handled, not silently swallowed: A90/A91 (dengue) and U06
     (Zika 2016 emergency code) are absent from simple-icd-10's WHO tree but are Brazil's
     highest-burden arbovirus codes. They must resolve as ``source_system_specific`` with a
     CID-10 chapter AND block (the gap-table fix), NEVER coerced to a WHO subcode (§II.13.2).
  2. The WHO-present arbovirus/microcephaly codes (chikungunya A92, yellow fever A95, Zika-in-
     WHO A92.x, microcephaly Q02, COVID U07.1) stay valid with the correct chapter — a
     catalogue-diff guard so a silent simple-icd-10 upgrade that moves them is caught here.
  3. The DiseaseGraph structural prior couples same-block arbovirus codes (the whole point of
     restoring the block for A90/A91).
"""

from __future__ import annotations

from pegasus.disease import icd_adapter
from pegasus.disease.graph import _structural_weight


def test_who_absent_cid10_codes_resolve_without_coercion() -> None:
    """A90/A91/U06 are WHO-absent but CID-10-real: chapter + block resolved, code NOT coerced."""
    a90 = icd_adapter.code_info("A90")
    assert a90.status == "source_system_specific"
    assert a90.chapter == "I"                       # A00-B99, infectious
    assert a90.block == "A90-A99"                    # gap-table fix (was None -> collapsed prior)
    assert a90.category == "A90"
    assert a90.normalized == "A90"                   # NEVER coerced to a WHO subcode (A97.x)

    a91 = icd_adapter.code_info("A91")
    assert a91.status == "source_system_specific" and a91.block == "A90-A99"

    u06 = icd_adapter.code_info("U06")               # Zika 2016 emergency code, WHO-retired
    assert u06.status == "source_system_specific"
    assert u06.chapter == "XXII" and u06.block == "U00-U49"


def test_who_present_arbovirus_and_microcephaly_catalogue_is_stable() -> None:
    """Catalogue-diff guard: these must stay WHO-valid with the pinned chapter."""
    present = {
        "A92": "I",     # other mosquito-borne viral fevers (chikungunya A92.0, Zika A92.5/.8)
        "A95": "I",     # yellow fever
        "A97": "I",     # dengue (the WHO-2019 relocation of A90/A91)
        "Q02": "XVII",  # microcephaly (congenital malformations)
        "U07.1": "XXII",# COVID-19, virus identified
    }
    for code, chapter in present.items():
        info = icd_adapter.code_info(code)
        assert info.status == "who_icd10", f"{code} unexpectedly not in WHO tree"
        assert info.chapter == chapter, f"{code} chapter drifted to {info.chapter}"
        assert info.block is not None, f"{code} lost its WHO block"


def test_disease_graph_couples_same_block_arbovirus() -> None:
    """Restored block makes dengue's two categories same-block (0.5); same code is same-category (1.0)."""
    assert _structural_weight("A90", "A90") == 1.0        # same category
    assert _structural_weight("A90", "A91") == 0.5        # dengue + dengue-haemorrhagic, same block
    # A different chapter is uncoupled (microcephaly vs dengue).
    assert _structural_weight("A90", "Q02") == 0.0
