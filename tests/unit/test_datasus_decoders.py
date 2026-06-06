from pegasus.datasus.decoders import (
    clamp_bool,
    decode_count2,
    decode_physical_scalar,
    decode_sim_idade,
    filter_cnpj,
)


def test_decode_sim_idade_years():
    decoded = decode_sim_idade("474")
    assert decoded.state == "valid"
    assert decoded.age_years == 74
    assert decoded.age_unit == "years"


def test_clamp_bool_outlier_invalid_not_true():
    decoded = clamp_bool("131")
    assert decoded.value is None
    assert decoded.state == "InvalidFlagState"
    assert decoded.warning == "boolean_flag_outlier"


def test_filter_cnpj_zero_nullified():
    decoded = filter_cnpj("00000000000000")
    assert decoded.cnpj is None
    assert decoded.state == "NullifiedZeroCNPJ"


def test_decode_count2_preserves_semantics():
    decoded = decode_count2("03", sentinels={"99"})
    assert decoded.value == 3
    assert decoded.raw_value == "03"


def test_physical_scalar_sentinel():
    decoded = decode_physical_scalar("9999", unit="grams", lower=300, upper=6500, sentinels={"9999"})
    assert decoded.value is None
    assert decoded.state == "InvalidScalar"
