"""Curated-cause reporter: describe-only over a run bundle (SCALE-01 study deliverable)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pegasus.workflows.report.curated_cause_report import (
    CuratedCauseReportError,
    build_curated_cause_report,
    render_curated_cause_html,
    write_curated_cause_report,
)

_COUNT = "SIM-DO.SIM-DO__processed_events.count.curated_cause_group"
_RATE = "CuratedCauseMortality"


def _synthetic_bundle(tmp: Path) -> Path:
    run = tmp / "run"
    tensors = run / "Tables" / "efg_tensors"
    tensors.mkdir(parents=True)
    count_fid, rate_fid = "cfid0001", "rfid0001"
    pl.DataFrame(
        {
            "field_id": [count_fid, rate_fid, "sfid"],
            "name": [_COUNT, _RATE, "IllDefinedCauseShare"],
        }
    ).write_parquet(run / "V_fields.parquet")

    # Two SP munis + one MG muni, two years, pancreas + one other cause.
    pl.DataFrame(
        {
            "year": [2021, 2021, 2021, 2022, 2022, 2022],
            "municipality_cod6": ["355030", "354990", "310620", "355030", "354990", "310620"],
            "curated_cause_group": ["neoplasm_pancreas"] * 3 + ["neoplasm_pancreas"] * 3,
            "value": [10.0, 5.0, 4.0, 12.0, 6.0, 5.0],
        }
    ).write_parquet(tensors / f"{count_fid}.parquet")

    pl.DataFrame(
        {
            "year": [2021, 2021, 2021, 2022, 2022, 2022],
            "municipality_cod6": ["355030", "354990", "310620", "355030", "354990", "310620"],
            "curated_cause_group": ["neoplasm_pancreas"] * 6,
            "numerator_count": [10.0, 5.0, 4.0, 12.0, 6.0, 5.0],
            "exposure": [100000.0, 50000.0, 40000.0, 100000.0, 50000.0, 40000.0],
        }
    ).write_parquet(tensors / f"{rate_fid}.measured_quantity.parquet")

    pl.DataFrame(
        {
            "field_id": [count_fid, rate_fid],
            "moran_i": [0.30, 0.31],
            "spatial_entropy": [0.5, 0.5],
            "cov_S": [1.0, 1.0],
            "cv": [0.4, 0.45],
            "n_eff": [900.0, 900.0],
            "provenance_risk": [0.0, 0.0],
            "state": ["verified", "verified"],
        }
    ).write_parquet(run / "Q_tensor.parquet")

    pl.DataFrame({"value": [0.05, 0.09]}).write_parquet(tensors / "sfid.parquet")

    (run / "UserIntent.json").write_text(
        json.dumps({"execution_scale": "national", "execution_stage": "compile", "population_mode": "official_sidra_anchor"}),
        encoding="utf-8",
    )
    return run


def test_report_totals_and_rates(tmp_path: Path) -> None:
    run = _synthetic_bundle(tmp_path)
    rep = build_curated_cause_report(run, cause_group="neoplasm_pancreas")

    assert rep.total_deaths == 42.0  # 10+5+4+12+6+5
    assert dict(rep.deaths_by_year) == {2021: 19.0, 2022: 23.0}
    assert rep.municipalities_with_deaths == 3

    # Pooled rate 2022 = (12+6+5)/(190000) * 1e5 = 12.105...
    rate_by_year = dict(rep.national_rate_per_100k_by_year)
    assert rate_by_year[2022] == pytest.approx(23.0 / 190000.0 * 1e5, rel=1e-6)

    # SP aggregates both 355030 + 354990; MG is 310620.
    by_uf = {u["uf"]: u for u in rep.by_uf}
    assert by_uf["SP"]["deaths"] == 33.0  # 10+5+12+6
    assert by_uf["MG"]["deaths"] == 9.0   # 4+5
    assert rep.spatial["interpretation"] == "spatially clustered (positive autocorrelation)"
    assert rep.quality["field_state"] == "verified"
    assert rep.scope.startswith("National")


def test_report_missing_cause_raises(tmp_path: Path) -> None:
    run = _synthetic_bundle(tmp_path)
    with pytest.raises(CuratedCauseReportError):
        build_curated_cause_report(run, cause_group="neoplasm_stomach")


def test_report_html_and_files(tmp_path: Path) -> None:
    run = _synthetic_bundle(tmp_path)
    out_html = tmp_path / "r.html"
    out_json = tmp_path / "r.json"
    rep = write_curated_cause_report(run, out_html=out_html, out_json=out_json)
    html = render_curated_cause_html(rep)
    assert "Malignant neoplasm of pancreas" in html
    assert "Moran" in html
    assert out_html.exists() and out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["total_deaths"] == 42.0
