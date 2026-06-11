import pytest
from pegasus.registries.cnes_capacity import CNESCapacityRegistryError, get_capacity_component, require_vector_index
from pegasus.registries.sih_cost import SIHCostRegistryError, get_cost_component, require_component_specific_cost


def test_cnes_capacity_registry_blocks_generic_beds():
    assert get_capacity_component("QTLEITP3").component_id == "obstetric_bed_capacity"
    with pytest.raises(CNESCapacityRegistryError):
        require_vector_index("Beds")


def test_sih_cost_registry_blocks_generic_cost_and_preserves_components():
    assert get_cost_component("VAL_SH").unit == "reais_hospital_services"
    assert get_cost_component("VAL_SP").unit == "reais_professional_services"
    assert get_cost_component("VAL_UTI").unit == "reais_icu_services"
    assert get_cost_component("VAL_TOT").unit == "reais_total_billing"
    with pytest.raises(SIHCostRegistryError):
        require_component_specific_cost("cost")
