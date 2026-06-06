from pegasus.datasus.icd_groups import block_for_icd, chapter_for_icd


def test_icd_chapter_mapping_for_fixture_codes():
    assert chapter_for_icd("A419").id == "CHAPTER_01_A00_B99"
    assert chapter_for_icd("P369").id == "CHAPTER_16_P00_P96"
    assert chapter_for_icd("Q249").id == "CHAPTER_17_Q00_Q99"


def test_icd_block_mapping_for_fixture_codes():
    assert block_for_icd("A419").id == "BLOCK_A30_A49"
    assert block_for_icd("P369").id == "BLOCK_P35_P39"
    assert block_for_icd("Q249").id == "BLOCK_Q20_Q28"
