import pytest

from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7, ibge_cod7_to_datasus_cod6
from pegasus.geo.support import SupportAlignmentError, assert_municipality_year_support_aligned


def test_datasus_to_ibge_smoke_crosswalk_is_explicit():
    assert datasus_cod6_to_ibge_cod7("270430") == "2704302"
    assert datasus_cod6_to_ibge_cod7("270030") == "2700300"
    assert ibge_cod7_to_datasus_cod6("2704302") == "270430"


def test_support_alignment_accepts_cod6_numerator_and_cod7_denominator():
    result = assert_municipality_year_support_aligned(
        numerator_support={"years": [2022], "municipalities": ["270430"]},
        numerator_axes={"geography": "mun_residence_cod6"},
        denominator_support={"years": [2022], "municipalities": ["2704302"]},
        denominator_axes={"geography": "municipality"},
    )
    assert result.aligned is True
    assert result.numerator_municipalities_ibge_cod7 == ["2704302"]
    assert result.denominator_municipalities_ibge_cod7 == ["2704302"]


def test_support_alignment_rejects_unmatched_municipality_sets():
    with pytest.raises(SupportAlignmentError, match="municipality_support_mismatch"):
        assert_municipality_year_support_aligned(
            numerator_support={"years": [2022], "municipalities": ["270030", "270430"]},
            numerator_axes={"geography": "mun_residence_cod6"},
            denominator_support={"years": [2022], "municipalities": ["2704302"]},
            denominator_axes={"geography": "municipality"},
        )
