"""P3b — edge query + export (FEAT-P3 vertical slice).

Query the LDO's certified hypotheses from a bundle (with row filters), export the frame + a
provenance sidecar. The typed LinkRecord fields ARE the edge provenance (pass-through).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from pegasus.ldo.records import LINK_RECORD_COLUMNS
from pegasus.output.query import QuerySpec, QueryError, materialize_query, write_dataset


def _write_fixture_bundle(bundle_dir) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    base = {c: None for c in LINK_RECORD_COLUMNS}
    rows = [
        {**base, "source_var": "A", "target_var": "B", "edge_type": "contemporaneous",
         "weight": 0.5, "certification_status": "selected", "uncertainty": 0.05, "stability": 0.9,
         "code_system": "CID-10", "fdr_qvalue": 0.001},
        {**base, "source_var": "C", "target_var": "D", "edge_type": "lagged_directed",
         "weight": 0.3, "certification_status": "descriptive", "uncertainty": 0.10, "stability": 0.4,
         "code_system": "CID-10", "fdr_qvalue": 0.30},
    ]
    pl.DataFrame(rows).write_parquet(bundle_dir / "Hypotheses.parquet")


def test_edge_query_filters_and_carries_provenance(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    _write_fixture_bundle(bundle)
    spec = QuerySpec(quantity="mortality_all_cause", kind="edge",
                     filters={"certification_status": "selected"}, fmt="csv")
    ds = materialize_query(bundle, spec)
    assert ds.kind == "edge"
    assert ds.frame.height == 1 and ds.frame["source_var"][0] == "A"
    # the typed provenance fields ride on the row (pass-through)
    assert ds.frame["uncertainty"][0] == 0.05 and ds.frame["fdr_qvalue"][0] == 0.001
    assert ds.provenance["n_rows"] == 1 and ds.provenance["query"]["quantity"] == "mortality_all_cause"
    assert str(ds.provenance["run"]["bundle_dir"]).endswith("bundle")


def test_edge_export_writes_frame_and_provenance_sidecar(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    _write_fixture_bundle(bundle)
    ds = materialize_query(bundle, QuerySpec(quantity="incidence_c25", kind="edge", fmt="csv"))
    paths = write_dataset(ds, tmp_path / "out", name="edges")
    assert paths["data"].exists() and paths["data"].suffix == ".csv"
    assert paths["provenance"].exists()
    prov = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert prov["kind"] == "edge" and prov["query"]["quantity"] == "incidence_c25"
    # both edges present (no filter), re-readable
    assert pl.read_csv(paths["data"]).height == 2


def test_unknown_filter_column_refuses(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    _write_fixture_bundle(bundle)
    with pytest.raises(QueryError, match="filter column"):
        materialize_query(bundle, QuerySpec(quantity="mortality_all_cause", kind="edge",
                                            filters={"not_a_column": 1}))


def test_rate_kind_is_wired_and_requires_a_materialized_numerator(tmp_path) -> None:
    # rate is now wired (P3d) as a SELECTION of the EFG's materialized RN field; a bare registry
    # quantity id with no corresponding numerator field in the bundle refuses at numerator resolution
    # (proving the read path runs), never a silent guess. Full rate behavior: test_output_query_rate.py.
    bundle = tmp_path / "bundle"
    _write_fixture_bundle(bundle)
    with pytest.raises(QueryError, match="rate numerator"):
        materialize_query(bundle, QuerySpec(quantity="mortality_all_cause", kind="rate"))


def _write_variable_dictionary(bundle_dir) -> None:
    rows = [
        {"field_id": "A", "name": "Pancreatic cancer deaths", "carrier": "Deaths", "unit": "counts",
         "diagnostic_role": "underlying_cause", "icd_group_id": "C25", "icd_group_kind": "category"},
        {"field_id": "B", "name": "Diabetes admissions", "carrier": "Admissions", "unit": "counts",
         "diagnostic_role": "comorbidity", "icd_group_id": "E11", "icd_group_kind": "category"},
        {"field_id": "C", "name": "Smoking context", "carrier": "Context", "unit": "rate",
         "diagnostic_role": None, "icd_group_id": None, "icd_group_kind": None},
        {"field_id": "D", "name": "Obesity context", "carrier": "Context", "unit": "rate",
         "diagnostic_role": None, "icd_group_id": None, "icd_group_kind": None},
    ]
    pl.DataFrame(rows).write_parquet(bundle_dir / "VariableDictionary.parquet")


def test_edge_enrichment_makes_hypotheses_self_describing(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    _write_fixture_bundle(bundle)
    _write_variable_dictionary(bundle)
    ds = materialize_query(bundle, QuerySpec(quantity="incidence_c25", kind="edge"))  # enrich default True
    assert ds.provenance["enriched"] is True
    assert {"source_name", "target_name", "source_icd_group_id"}.issubset(ds.frame.columns)
    row = ds.frame.filter(pl.col("source_var") == "A").row(0, named=True)
    assert row["source_name"] == "Pancreatic cancer deaths" and row["source_icd_group_id"] == "C25"
    assert row["target_name"] == "Diabetes admissions"


def test_enrichment_absent_dictionary_warns_but_still_serves_edges(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    _write_fixture_bundle(bundle)  # no VariableDictionary
    ds = materialize_query(bundle, QuerySpec(quantity="incidence_c25", kind="edge"))
    assert "variable_metadata_unavailable_edges_unenriched" in ds.warnings
    assert ds.provenance["enriched"] is False
    assert ds.frame.height == 2  # edges still returned, just unenriched
