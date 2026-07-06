"""Net-migration residual + SINASC maternal-race fallback (MSD §2.8.7 / §2.8.5).

Migration is inferred by exclusion from the demographic balancing equation
NetMig(s,t) = E(s,t) - E(s,t-1) - Births(s,t) + Deaths(s,t), using SIDRA
civil-registry vital totals (tables 2609 births / 2683 deaths) preferred, DATASUS
SIM/SINASC counts as fallback. These tests use synthetic facts (no network).
"""
from pathlib import Path

import polars as pl

from pegasus.denominators.reconstruction.loss import evaluate_population_loss
from pegasus.denominators.reconstruction.schema import PopulationObjectiveWeights, PopulationTensorProblem
from pegasus.sidra.facts import normalize_flat_records_to_facts, write_facts_parquet
from pegasus.denominators.population.build import (
    _migration_residual_totals,
    _sidra_vital_totals,
    solve_population_tensor_from_sidra_strata,
)

MUNI = "2704302"
MUNI_COD6 = "270430"


def _total_row(table, var, year, val, disagg=False):
    return {
        "table_id": table, "variable_id": var, "period": year, "locality_level": "N6",
        "locality_id": MUNI,
        "classification_tuple": ([["2", "S"], ["86", "R"], ["287", "A"]] if disagg else []),
        "category_tuple": ([["2", "6794"], ["86", "95251"], ["287", "100362"]] if disagg else []),
        "value": str(val), "unit": "P",
    }


def _civ(table, var, year, val):
    return {"table_id": table, "variable_id": var, "period": year, "locality_level": "N6",
            "locality_id": MUNI, "classification_tuple": [], "category_tuple": [], "value": str(val), "unit": "P"}


def test_migration_loss_term_gradient_is_exact():
    import numpy as np
    shape, n = (2, 3, 1, 1, 1), 6
    rng = np.random.default_rng(1)
    P, M = rng.random(n) + 1, rng.random(n) * 2 - 1
    prob = PopulationTensorProblem(
        shape=shape, anchors=(None,) * n, hard_anchor_mask=(False,) * n,
        mode="independent_denominator",
        migration_locality_totals=(0.5, None, -0.3, 0.1, None, 0.2),
        weights=PopulationObjectiveWeights(anchor=0, aging=0, birth=0, death=0, migration=0, migration_total=0.7, race=0, age_smooth=0),
    )
    ev = evaluate_population_loss(prob, tuple(P), tuple(M))
    gm = np.array(ev.migration_gradient)
    num = np.zeros(n)
    eps = 1e-6
    for i in range(n):
        Mp, Mm = M.copy(), M.copy()
        Mp[i] += eps
        Mm[i] -= eps
        num[i] = (evaluate_population_loss(prob, tuple(P), tuple(Mp)).total - evaluate_population_loss(prob, tuple(P), tuple(Mm)).total) / (2 * eps)
    assert np.max(np.abs(gm - num)) < 1e-5
    assert ev.terms["migration_total"] > 0


def test_sidra_civil_registry_vital_totals_reads_per_muni_year(tmp_path: Path):
    path = tmp_path / "births.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts([_civ("2609", "217", "2016", 30), _civ("2609", "217", "2017", 32)], table_id="2609", request_hash="r", metadata_hash="m"),
        output_path=path,
    )
    totals = _sidra_vital_totals(path, table_id="2609", variable_id="217")
    assert totals == {(MUNI_COD6, "2016"): 30.0, (MUNI_COD6, "2017"): 32.0}


def test_residual_only_for_consecutive_years_with_both_closures():
    # closure over 4 periods: 2010 gap then 2015,2016,2017 consecutive.
    shape = (1, 4, 1, 1, 1)
    periods = ("2010", "2015", "2016", "2017")
    closure = [1000.0, 2000.0, 2100.0, 2150.0]
    births = {(MUNI_COD6, "2016"): 32.0, (MUNI_COD6, "2017"): 31.0}
    deaths = {(MUNI_COD6, "2016"): 22.0, (MUNI_COD6, "2017"): 19.0}
    residual, warnings = _migration_residual_totals(
        closure=closure, births_by_st=births, deaths_by_st=deaths,
        localities=(MUNI_COD6,), periods=periods, shape=shape,
    )
    # 2015: predecessor 2010 is a 5-year gap -> None. 2016: 2100-2000-32+22=90. 2017: 2150-2100-31+19=38.
    assert residual == (None, None, 90.0, 38.0)


def test_end_to_end_migration_recovers_closure_and_directional_residual(tmp_path: Path):
    strata_p = tmp_path / "strata.parquet"
    write_facts_parquet(normalize_flat_records_to_facts([_total_row("9606", "93", "2015", 2000, disagg=True)], table_id="9606", request_hash="r", metadata_hash="m"), output_path=strata_p)
    totals = [_total_row("9606", "93", "2015", 2000, disagg=True)] + [
        {"table_id": "6579", "variable_id": "9324", "period": y, "locality_level": "N6", "locality_id": MUNI,
         "classification_tuple": [], "category_tuple": [], "value": str(v), "unit": "P"}
        for y, v in [("2016", 2100), ("2017", 2150)]
    ]
    totals_p = tmp_path / "totals.parquet"
    write_facts_parquet(normalize_flat_records_to_facts(totals, table_id="mix", request_hash="r", metadata_hash="m"), output_path=totals_p)
    births_p, deaths_p = tmp_path / "b.parquet", tmp_path / "d.parquet"
    write_facts_parquet(normalize_flat_records_to_facts([_civ("2609", "217", "2016", 30), _civ("2609", "217", "2017", 32)], table_id="2609", request_hash="r", metadata_hash="m"), output_path=births_p)
    write_facts_parquet(normalize_flat_records_to_facts([_civ("2683", "343", "2016", 20), _civ("2683", "343", "2017", 22)], table_id="2683", request_hash="r", metadata_hash="m"), output_path=deaths_p)

    build = solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_p, total_anchor_path=totals_p, output_path=tmp_path / "tensor.parquet",
        mode="independent_denominator", civil_registry_births_path=births_p, civil_registry_deaths_path=deaths_p,
        max_iterations=5000,
    )
    terms = build.as_manifest()["diagnostics"]["objective_terms"]
    assert terms["migration_total"] > 0.0
    out = pl.read_parquet(build.output_path)
    # closure constraint holds exactly for every year.
    pops = {r["year"]: r["pop"] for r in out.group_by("year").agg(pl.col("value").sum().round(1).alias("pop")).iter_rows(named=True)}
    assert pops == {2015: 2000.0, 2016: 2100.0, 2017: 2150.0}
    # 2016 expected residual = 2100-2000-30+20 = 90 (positive); migration pulled positive.
    net_2016 = out.filter(pl.col("year") == 2016)["migration"].sum()
    assert net_2016 > 30.0


def test_maternal_race_fills_in_for_missing_newborn_race(tmp_path: Path):
    """A birth whose newborn race is missing but whose mother's race is declared
    must still be race-placeable via the maternal fallback (MSD §2.8.5 r_n|r_m),
    not dropped as unbridgeable."""
    from pegasus.denominators.population.build import _sinasc_birth_priors
    from pegasus.measurement.race import load_race_bridge_prior

    prior = load_race_bridge_prior("config/priors/race_bridge/fixedC_sim_admin_to_ibge_selfdeclared_v1.json")
    sinasc = pl.DataFrame({
        "mun_residence_cod6": [MUNI_COD6] * 4,
        "birth_year": [2016] * 4,
        "newborn_sex": ["male", "female", "male", "female"],
        # newborn race all missing; maternal race present for all.
        "newborn_race_admin": [None, None, None, None],
        "newborn_race_state": ["missing"] * 4,
        "maternal_race_admin": ["1", "4", "2", "4"],
        "maternal_race_state": ["valid_admin_race"] * 4,
    })
    sinasc_p = tmp_path / "sinasc.parquet"
    sinasc.write_parquet(sinasc_p)

    locality_index = {MUNI_COD6: 0}
    period_index = {"2016": 0}
    sex_index = {"male": 0, "female": 1}
    race_index = {"branca": 0, "preta": 1, "amarela": 2, "parda": 3, "indigena": 4}
    births, warnings = _sinasc_birth_priors(
        sinasc_events_path=sinasc_p, locality_index=locality_index, period_index=period_index,
        sex_index=sex_index, race_index=race_index, t_count=1, x_count=2, r_count=5,
        race_bridge_prior=prior,
    )
    assert births is not None, "maternal fallback should keep the births bridgeable"
    # 4 births conserved across the bridged race categories.
    assert sum(v for v in births if v is not None) == 4.0
