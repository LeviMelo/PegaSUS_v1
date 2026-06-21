from __future__ import annotations

from pegasus.datasus.sinasc_normalize import decode_anomaly_flag, normalize_anomaly_icd
from pegasus.geo.state_panel import clean_datasus_municipalities, is_valid_datasus_municipality_cod6


def test_alagoas_cod6_support_excludes_state_pseudocode() -> None:
    valid, invalid = clean_datasus_municipalities(["270000", "270010", "270430", None, "foo"], uf_prefix="27")
    assert valid == ["270010", "270430"]
    assert "270000" in invalid
    assert not is_valid_datasus_municipality_cod6("270000", uf_prefix="27")


def test_sinasc_idanomal_codebook_is_not_inverted() -> None:
    present, present_state = decode_anomaly_flag("1")
    absent, absent_state = decode_anomaly_flag("2")
    ignored, ignored_state = decode_anomaly_flag("9")

    assert present is True
    assert absent is False
    assert ignored is None
    assert present_state == "valid_present"
    assert absent_state == "valid_absent"
    assert ignored_state == "sentinel"


def test_sinasc_idanomal_accepts_microdatasus_labels() -> None:
    present, present_state = decode_anomaly_flag("Sim")
    absent, absent_state = decode_anomaly_flag("Não")

    assert present is True
    assert absent is False
    assert present_state == "valid_present"
    assert absent_state == "valid_absent"


def test_sinasc_anomaly_icd_numerator_policy() -> None:
    assert normalize_anomaly_icd(None, False) == (None, "absent", False)
    assert normalize_anomaly_icd(None, True) == (None, "flag_present_code_missing", True)
    assert normalize_anomaly_icd("Q359", False) == ("Q359", "valid_q_anomaly", True)
    assert normalize_anomaly_icd("Z000", False) == ("Z000", "valid_non_q_absent", False)
    assert normalize_anomaly_icd("Z000", None) == ("Z000", "valid_non_q_not_anomaly", False)
    assert normalize_anomaly_icd("not-an-icd", None) == (None, "invalid", False)
