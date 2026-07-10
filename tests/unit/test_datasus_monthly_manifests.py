from __future__ import annotations

from pegasus.datasus.manifests import build_datasus_manifests


def test_cnes_st_builds_monthly_manifests() -> None:
    manifests = build_datasus_manifests(system="CNES-ST", uf="AL", years="2022", config={"rscript_path": "Rscript"})
    assert len(manifests) == 12
    assert [m.month_start for m in manifests] == list(range(1, 13))
    assert [m.month_end for m in manifests] == list(range(1, 13))
    assert all(m.system == "CNES-ST" for m in manifests)


def test_sih_rd_builds_monthly_manifests() -> None:
    manifests = build_datasus_manifests(system="SIH-RD", uf="AL", years="2022", config={"rscript_path": "Rscript"})
    assert len(manifests) == 12
    assert [m.month_start for m in manifests] == list(range(1, 13))
    assert [m.month_end for m in manifests] == list(range(1, 13))
    assert all(m.system == "SIH-RD" for m in manifests)


def test_known_uf_month_source_gaps_are_not_planned_as_failed_fetches() -> None:
    rr = build_datasus_manifests(system="SIH-RD", uf="RR", years="2000", config={"rscript_path": "Rscript"})
    assert [manifest.month_start for manifest in rr] == list(range(6, 13))
    ro = build_datasus_manifests(system="SIH-RD", uf="RO", years="2000", config={"rscript_path": "Rscript"})
    assert [manifest.month_start for manifest in ro] == list(range(1, 13))
    ap = build_datasus_manifests(system="SIH-RD", uf="AP", years="2007", config={"rscript_path": "Rscript"})
    assert 10 not in [manifest.month_start for manifest in ap]


def test_sim_sinasc_remain_annual_manifests() -> None:
    sim = build_datasus_manifests(system="SIM-DO", uf="AL", years="2022", config={"rscript_path": "Rscript"})
    sinasc = build_datasus_manifests(system="SINASC", uf="AL", years="2022", config={"rscript_path": "Rscript"})
    assert len(sim) == 1
    assert len(sinasc) == 1
    assert sim[0].month_start is None
    assert sinasc[0].month_start is None


def test_publication_ceiling_bounds_not_yet_published_years() -> None:
    # SINASC final data (NOV/DNRES) ends 2022 — 2023/2024 are not published, so a strict run must not
    # plan them (verified against the DATASUS FTP listing 2026-07-10). SIM-DO/SIH-RD publish through 2024.
    assert build_datasus_manifests(system="SINASC", uf="AL", years="2022", config={"rscript_path": "Rscript"})
    assert build_datasus_manifests(system="SINASC", uf="AL", years="2023", config={"rscript_path": "Rscript"}) == []
    assert build_datasus_manifests(system="SINASC", uf="AL", years="2024", config={"rscript_path": "Rscript"}) == []
    assert len(build_datasus_manifests(system="SIM-DO", uf="AL", years="2024", config={"rscript_path": "Rscript"})) == 1
    assert len(build_datasus_manifests(system="SIH-RD", uf="AL", years="2024", config={"rscript_path": "Rscript"})) == 12
