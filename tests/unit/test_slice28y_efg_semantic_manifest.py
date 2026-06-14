from __future__ import annotations

from types import SimpleNamespace

import pegasus.efg.dag as dag


def _lineage():
    return SimpleNamespace(registry_versions={"test_registry": "v1"}, source_manifest_hashes=["hash_a"])


def _field(field_id: str, *, carrier: str, unit: str, source: str):
    return SimpleNamespace(
        id=field_id,
        field_id=field_id,
        name=field_id,
        kind="extensive_measure",
        carrier=carrier,
        unit=unit,
        aggregation="additive",
        role=[],
        source=[source],
        lineage=_lineage(),
        warnings=[],
    )


class _DummyEFGResult:
    def __init__(self):
        self.fields = (
            _field("sim_deaths", carrier="Deaths", unit="counts", source="SIM"),
            _field("sidra_population", carrier="Population", unit="persons", source="SIDRA"),
        )

    def as_manifest(self):
        return {"schema_version": "test"}


def test_slice28y_build_efg_wrapper_attaches_semantic_summaries(monkeypatch):
    monkeypatch.setattr(dag, "_build_efg_base", lambda *args, **kwargs: _DummyEFGResult())
    result = dag.build_efg(substrate=object(), registry_root="config/registries")

    core = getattr(result, "_slice28y_core_seed_summary")
    bridges = getattr(result, "_slice28y_bridge_plan_summary")

    assert core["seed_count"] == 2
    assert core["role_counts"]["death_event_seed"] == 1
    assert core["role_counts"]["population_denominator_seed"] == 1
    assert bridges["bridge_type_counts"]["mortality_rate_bridge"] == 1


def test_slice28y_efgresult_manifest_method_is_patched() -> None:
    code = dag.EFGResult.as_manifest.__code__
    source_names = set(code.co_names)
    string_constants = {value for value in code.co_consts if isinstance(value, str)}

    # ``core_seed_summary`` and ``bridge_plan_summary`` are payload keys passed
    # to ``dict.setdefault``.  In CPython bytecode they are string constants, not
    # attribute/function names.  The contract is that the patched manifest method
    # injects those keys and uses the Slice 28Y default-summary helpers.
    assert "setdefault" in source_names
    assert "core_seed_summary" in string_constants
    assert "bridge_plan_summary" in string_constants
    assert "semantic_manifest_schema" in string_constants
    assert "_slice28y_empty_core_seed_summary" in source_names
    assert "_slice28y_empty_bridge_plan_summary" in source_names

