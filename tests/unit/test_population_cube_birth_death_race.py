"""Birth/death/race data ingestion into the population tensor (MSD §2.8.1/§2.8.5/
§2.8.6/§2.8.8). Prior to this, ``solve_population_tensor_from_sidra_strata`` never
requested SINASC births or race-bridged SIM deaths at all, and the birth/race
objective weights were hardcoded to zero regardless of data availability -- these
tests pin the fix: real data in, nonzero objective terms out.
"""
from pathlib import Path

import polars as pl
import pytest

from pegasus.sidra.facts import normalize_flat_records_to_facts, write_facts_parquet
from pegasus.sidra.population_cube.build import solve_population_tensor_from_sidra_strata

MUNI = "2704302"
MUNI_COD6 = "270430"
RACE_CATS = {"branca": "2776", "preta": "2777", "amarela": "2778", "parda": "2779", "indigena": "2780"}
SEX_CATS = {"male": "4", "female": "5"}
AGE_CATS = {"age_0": "6557", "age_1": "6558"}
PRIOR_PATH = Path("config/priors/race_bridge/fixedC_sim_admin_to_ibge_selfdeclared_v1.json")


def _census_records(year: str, counts_by_age_sex_race: dict) -> list[dict]:
    out = []
    for age_label, age_code in AGE_CATS.items():
        for sex_label, sex_code in SEX_CATS.items():
            for race_label, race_code in RACE_CATS.items():
                value = counts_by_age_sex_race[age_label][sex_label][race_label]
                out.append({
                    "table_id": "9606", "variable_id": "93", "period": year, "locality_level": "N6",
                    "locality_id": MUNI,
                    "classification_tuple": [["2", "Sexo"], ["86", "Cor ou raça"], ["287", "Idade"]],
                    "category_tuple": [["2", sex_code], ["86", race_code], ["287", age_code]],
                    "value": str(value), "unit": "Pessoas",
                })
    return out


_COUNTS = {
    "age_0": {"male": {"branca": 40, "preta": 10, "amarela": 1, "parda": 30, "indigena": 1},
              "female": {"branca": 42, "preta": 11, "amarela": 1, "parda": 31, "indigena": 1}},
    "age_1": {"male": {"branca": 400, "preta": 100, "amarela": 10, "parda": 300, "indigena": 5},
              "female": {"branca": 420, "preta": 110, "amarela": 12, "parda": 310, "indigena": 6}},
}


def _build(tmp_path: Path, *, sim_year: int, mode: str = "sim_informed_denominator"):
    records = _census_records("2010", _COUNTS) + _census_records("2022", _COUNTS)
    strata_path = tmp_path / "strata.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts(records, table_id="9606", request_hash="r1", metadata_hash="m1"),
        output_path=strata_path,
    )
    totals_path = tmp_path / "totals.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts(records, table_id="mixed", request_hash="r2", metadata_hash="m2"),
        output_path=totals_path,
    )

    sim = pl.DataFrame({
        "mun_residence_cod6": [MUNI_COD6] * 6,
        "year": [sim_year] * 6,
        "sex": ["male", "male", "female", "female", "male", "female"],
        "age_years": [1, 1, 1, 1, 1, 1],
        "race_color_admin": ["1", "2", "4", "1", "3", "5"],
        "race_missingness_state": ["valid_admin_race"] * 6,
    })
    sim_path = tmp_path / "sim_events.parquet"
    sim.write_parquet(sim_path)

    sinasc = pl.DataFrame({
        "mun_residence_cod6": [MUNI_COD6] * 8,
        "year": [2010] * 8,
        "newborn_sex": ["male", "female", "male", "female", "male", "female", "male", "female"],
        "newborn_race_admin": ["1", "1", "2", "4", "4", "1", "3", "5"],
        "newborn_race_state": ["valid_admin_race"] * 8,
    })
    sinasc_path = tmp_path / "sinasc_events.parquet"
    sinasc.write_parquet(sinasc_path)

    return solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_path,
        total_anchor_path=totals_path,
        output_path=tmp_path / "tensor.parquet",
        mode=mode,
        sim_events_path=sim_path,
        sinasc_events_path=sinasc_path,
        race_bridge_prior_path=PRIOR_PATH,
        max_iterations=200,
    )


def test_sinasc_births_activate_the_birth_loss(tmp_path: Path):
    build = _build(tmp_path, sim_year=2022)
    terms = build.as_manifest()["diagnostics"]["objective_terms"]
    assert terms["birth"] > 0.0


def test_sim_deaths_bridge_to_race_and_activate_death_loss_at_a_census_year(tmp_path: Path):
    build = _build(tmp_path, sim_year=2022)
    terms = build.as_manifest()["diagnostics"]["objective_terms"]
    assert terms["death"] > 0.0

    out = pl.read_parquet(build.output_path)
    death_rows = out.filter((pl.col("year") == 2022) & pl.col("sim_deaths").is_not_null() & (pl.col("sim_deaths") > 0))
    # 6 raw admin-race deaths bridged across the 5 canonical race categories must
    # conserve total count (Bridge_R redistributes, it does not fabricate or drop).
    assert death_rows["sim_deaths"].sum() == pytest.approx(6.0, abs=1e-6)
    assert set(death_rows["race"].unique().to_list()) <= set(RACE_CATS)


def test_sim_deaths_in_a_non_census_year_do_not_activate_death_rate(tmp_path: Path):
    """Known limitation (MSD §2.8.6 implementation note): delta is estimated as
    deaths/anchor, and the anchor only exists at census years -- deaths in an
    intercensal year cannot inform the death loss yet, and must not silently
    fabricate a rate either. independent_denominator mode is used here because
    sim_informed_denominator hard-requires lambda_D>0 (loss.py's own validation) --
    it is not a mode that can gracefully degrade when death data can't be placed."""
    build = _build(tmp_path, sim_year=2015, mode="independent_denominator")
    terms = build.as_manifest()["diagnostics"]["objective_terms"]
    assert terms["death"] == 0.0


def test_census_race_composition_prior_activates_the_race_loss(tmp_path: Path):
    build = _build(tmp_path, sim_year=2022)
    terms = build.as_manifest()["diagnostics"]["objective_terms"]
    assert terms["race"] > 0.0


def test_no_bridge_prior_leaves_race_stratified_priors_unwired_not_fabricated(tmp_path: Path):
    """Without a configured Bridge_R prior, SIM/SINASC admin race cannot be honestly
    placed on the tensor's self-declared race axis (MSD §2.8.1 axis-identity note) --
    birth/death priors must be entirely absent, not silently collapsed onto one race."""
    records = _census_records("2010", _COUNTS) + _census_records("2022", _COUNTS)
    strata_path = tmp_path / "strata.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts(records, table_id="9606", request_hash="r1", metadata_hash="m1"),
        output_path=strata_path,
    )
    totals_path = tmp_path / "totals.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts(records, table_id="mixed", request_hash="r2", metadata_hash="m2"),
        output_path=totals_path,
    )
    sim = pl.DataFrame({
        "mun_residence_cod6": [MUNI_COD6] * 6,
        "year": [2022] * 6,
        "sex": ["male", "male", "female", "female", "male", "female"],
        "age_years": [1, 1, 1, 1, 1, 1],
        "race_color_admin": ["1", "2", "4", "1", "3", "5"],
        "race_missingness_state": ["valid_admin_race"] * 6,
    })
    sim_path = tmp_path / "sim_events.parquet"
    sim.write_parquet(sim_path)

    build = solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_path,
        total_anchor_path=totals_path,
        output_path=tmp_path / "tensor.parquet",
        mode="independent_denominator",
        sim_events_path=sim_path,
        race_bridge_prior_path=None,
        max_iterations=200,
    )
    manifest = build.as_manifest()
    assert manifest["diagnostics"]["objective_terms"]["death"] == 0.0
    assert "sim_death_race_stratification_unavailable_without_bridge" in manifest["warnings"]


def test_age_group_axis_is_ordered_numerically_not_alphabetically(tmp_path: Path):
    """Regression for the age-axis sort bug found while stress-testing births/deaths:
    tuple(sorted(...)) on the raw "age_N" labels is alphabetical ("age_10" < "age_2"),
    which would silently scramble the tensor's chronological age adjacency."""
    build = _build(tmp_path, sim_year=2022)
    manifest = build.as_manifest()
    assert manifest["age_groups"] == ["age_0", "age_1"]
