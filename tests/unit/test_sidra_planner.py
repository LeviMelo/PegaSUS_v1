from pegasus.sidra.metadata import fixture_sidra_metadata
from pegasus.sidra.plan import estimate_cells, plan_sidra_chunks
from pegasus.sidra.schemas import SIDRARequest


def test_estimate_cells_multiplies_axes():
    assert estimate_cells(
        localities=["1", "2"],
        periods=["2022"],
        variables=["93"],
        classifications={"2": ["0", "1"], "58": ["0", "4"]},
    ) == 8


def test_small_request_does_not_require_view_definition():
    metadata = fixture_sidra_metadata()
    request = SIDRARequest(
        table_id="9606",
        variables=["93"],
        periods=["2022"],
        locality_level="N6",
        localities=["270430"],
        classifications={"2": ["0"], "58": ["0"], "287": ["0"]},
    )
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=49900)
    assert len(chunks) == 1
    assert chunks[0].estimated_cells == 1


def test_large_request_splits_by_localities_first():
    metadata = fixture_sidra_metadata()
    table = metadata.tables["9606"]
    request = SIDRARequest(
        table_id="9606",
        variables=["93"],
        periods=["2022", "2023"],
        locality_level="N6",
        localities=[f"{i:06d}" for i in range(100)],
        classifications={"2": ["0", "1", "2"], "58": ["0", "4", "5"], "287": [str(i) for i in range(100)]},
    )

    # Extend fixture metadata so test is about split behavior, not validation.
    table.localities_by_level["N6"] = request.localities
    table.classifications["287"] = request.classifications["287"]

    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=49900)
    assert len(chunks) > 1
    assert all(c.estimated_cells <= 49900 for c in chunks)
    assert sum(c.estimated_cells for c in chunks) == 100 * 2 * 1 * 3 * 3 * 100
