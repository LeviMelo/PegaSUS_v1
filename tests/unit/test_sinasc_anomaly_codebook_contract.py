from __future__ import annotations

from pegasus.datasus.sinasc_normalize import decode_anomaly_flag, normalize_anomaly_icd


def test_idanomal_official_codebook() -> None:
    assert decode_anomaly_flag("1") == (True, "valid_present")
    assert decode_anomaly_flag("2") == (False, "valid_absent")
    assert decode_anomaly_flag("9") == (None, "sentinel")


def test_idanomal_text_labels() -> None:
    assert decode_anomaly_flag("Sim") == (True, "valid_present")
    assert decode_anomaly_flag("Não") == (False, "valid_absent")


def test_cod_anomal_does_not_promote_absent_births() -> None:
    assert normalize_anomaly_icd(None, False) == (None, "absent", False)
    assert normalize_anomaly_icd("0000", False)[2] is False
    assert normalize_anomaly_icd("Q999", False)[2] is True
