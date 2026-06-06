from pegasus.datasus.icd_parser import parse_icd


def test_valid_icd():
    result = parse_icd("A419", topology_role="underlying_cause", source_field="CAUSABAS")
    assert result.parse_state == "valid"
    assert result.normalized == "A419"


def test_asterisk_preserved_as_marker():
    result = parse_icd("*A419", topology_role="terminal_chain", source_field="LINHAA", position="A")
    assert result.parse_state == "valid"
    assert result.normalized == "A419"
    assert result.raw_marker == "*"


def test_r_code_is_ill_defined():
    result = parse_icd("R99", topology_role="underlying_cause", source_field="CAUSABAS")
    assert result.parse_state == "ill-defined"


def test_blank_not_coerced_to_r99():
    result = parse_icd("", topology_role="underlying_cause", source_field="CAUSABAS")
    assert result.parse_state == "blank"
    assert result.normalized is None
