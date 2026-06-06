from pegasus.sidra.registry import request_from_view


def test_sidra_view_registry_population_smoke_view():
    request = request_from_view("population_9606_total_2022_alagoas_smoke")
    assert request.table_id == "9606"
    assert request.variables == ["93"]
    assert request.periods == ["2022"]
    assert request.locality_level == "N6"
    assert request.localities == ["2704302"]
    assert request.classifications["86"] == ["95251"]
